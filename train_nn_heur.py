from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from model_utils import Transformer, get_model, get_saved_model
from multiprocessing import cpu_count
import torch.nn.functional as F

import numpy as np
import random, torch, time

#torch.set_num_threads(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
N_CORES = cpu_count()
print('Done importing modules', device)

GOAL = 'ABCDEFGHIJKLMNO.'

swap_letters = lambda s, i0, i1: f'{s[:i0]}{s[i1]}{s[i0+1:i1]}{s[i0]}{s[i1+1:]}'
board_str_to_id = lambda board: [15 if c == '.' else ord(c)-65 for c in board]
str_to_arr = lambda board: np.array(board_str_to_id(board), dtype=np.int64)

def print_board(board):
    for i in range(4):
        print(board[4*i:4*(i+1)])
    print()


GOAL_VEC = torch.tensor(str_to_arr(GOAL)).to(device)

def get_possible_moves(board):
    # 0123 -> UDLR, move_idx represents correct move from this position
    dot_idx = board.index('.')
    dot_row, dot_col = dot_idx // 4, dot_idx % 4
    ret = []
    if dot_row > 0: # move dot up
        ret.append(0)
    if dot_row < 3: # move dot down
        ret.append(1)
    if dot_col > 0: # move dot left
        ret.append(2)
    if dot_col < 3: # move dot right
        ret.append(3)
    return ret

def make_move(board, move):
    # 0123 -> udlr, move_idx represents correct move from this position
    dot_idx = board.index('.')
    dot_row, dot_col = dot_idx // 4, dot_idx % 4
    if move == 0: # move dot up
        assert dot_row > 0
        s, e = dot_idx-4, dot_idx
    elif move == 1: # move dot down
        assert dot_row < 3
        s, e = dot_idx, dot_idx+4
    elif move == 2: # move dot left
        assert dot_col > 0
        s, e = dot_idx-1, dot_idx
    elif move == 3: # move dot right
        assert dot_col < 3
        s, e = dot_idx, dot_idx+1
    return swap_letters(board, s, e)

move_pairs = {0:1, 1:0, 2:3, 3:2}
def make_new_board(max_num_reset_moves):
    board = GOAL
    num_reset_moves = random.randint(0, max_num_reset_moves)
    if num_reset_moves == 0:
        return GOAL
    move = round(random.random()) * 2 # either 0 (U) or 2 (L)
    board = make_move(board, move)
    past_move = move 
    for _ in range(num_reset_moves-1):
        moves = get_possible_moves(board)
        moves.remove(move_pairs[past_move])
        move = random.choice(moves)
        board = make_move(board, move)
        past_move = move
    return board

class SliderDataset(Dataset):
    def __init__(self, k=500):
        self.k = k

    def __len__(self):
        return int(1e9)

    def __getitem__(self, idx):
        board = make_new_board(self.k)
        board_actions = get_possible_moves(board)
        num_actions = len(board_actions)
        neighbors = [str_to_arr(make_move(board, action)) for action in board_actions] + [np.zeros((16,), dtype=np.int64) for _ in range(4-num_actions)]
        return str_to_arr(board), neighbors, len(board_actions)

def convert_vec_to_str(vecs):
    ret = [] 
    for v in vecs:
        if np.max(v) == 0:
            ret.append(None)
            continue
        board_str = ''
        for i in range(16):
            chr_idx = np.argmax(v[16*i:16*(i+1)])
            if chr_idx == 15:
               board_str += '.'
            else:
                board_str += chr(65+chr_idx)
        ret.append(board_str)
    return ret

BATCH_SIZE = 2048
dataloader = DataLoader(SliderDataset(), batch_size=BATCH_SIZE, num_workers=N_CORES)

model = get_saved_model().to(device)

def j_approx_fn(states, neighbors, num_actions, model):
    ret = torch.ones((states.shape[0], 1)).to(device)
    with torch.autocast(device, enabled=True, dtype=torch.bfloat16):
        possible_j = model(neighbors)
    s_i = 0
    for state_idx, num_neighbors in enumerate(num_actions):
        e_i = s_i + num_neighbors
        ret[state_idx] += possible_j[s_i:e_i].min()
        s_i = e_i
    ret[(states == GOAL_VEC).all(dim=1)] = 0
    return ret

def train(model, dataloader, opt, loss_fn=F.mse_loss, loss_thresh=0.1, lr_decay_gamma=0.9999993):
    original_model = Transformer().eval().to(device)
    original_model.load_state_dict(model.state_dict())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, lr_decay_gamma)

    try:
        tic = time.time()
        for batch_idx, data in enumerate(dataloader):
            with torch.no_grad():
                boards, neighbors, num_actions = data
                neighbors = torch.cat(neighbors, axis=1).reshape(BATCH_SIZE*4, 16)
                neighbors = neighbors[(neighbors != torch.zeros(16,)).any(dim=1)].long()
                boards = boards.to(device)
                neighbors = neighbors.to(device)
                num_actions = num_actions.to(device)

                y = j_approx_fn(boards, neighbors, num_actions, original_model)

            for _ in range(1):
                opt.zero_grad()
                with torch.autocast(device, enabled=True, dtype=torch.bfloat16):
                    y_pred = model(boards.detach())
                    loss = loss_fn(y_pred, y.detach())
                loss.backward()
                opt.step()
                scheduler.step()

            if batch_idx % 10 == 0:
                j = y.cpu().detach().numpy()
                print(f'J min/mean/max: {j.min()}/{j.mean()}/{j.max()}')
                recorded_loss = loss.cpu().detach().numpy()
                print('Loss:', recorded_loss)
                print(f'Duration: {time.time() - tic}')
                tic = time.time()

                if recorded_loss < loss_thresh:
                    original_model = Transformer().eval().to(device)
                    original_model.load_state_dict(model.state_dict())

    except KeyboardInterrupt:
        save_input = input('Save? [Y/n] ')
        if save_input != 'n':
            torch.save(model.state_dict(), 'slider_model_h.pth')

train(model, dataloader, Adam(model.parameters(), 1e-4))
