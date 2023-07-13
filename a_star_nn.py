from typing import List, Tuple, Set, Callable, Dict
from model_utils import get_saved_model
from collections import deque
from os import stat
import numpy as np
import time, sys, heapq, torch

# Note: all Tuples[str, str] instances represent a tuple (<slider_state>, <path_to_state>) and str instances (in func args) is <slider_state> itself

if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = '15_puzzles.txt'
    #filename = 'slide_puzzle_tests_2.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

puzzles = [line.strip() for line in open(filename)]

calc_board_size = lambda state: int(len(state) ** 0.5)
get_idx = lambda size, y, x: y * size + x
get_coords = lambda size, ind: (ind // size, ind % size)
# swaps string characters based on index; assumes that ind1 < ind2
swap = lambda text, ind1, ind2: f'{text[:ind1]}{text[ind2]}{text[ind1+1:ind2]}{text[ind1]}{text[ind2+1:]}'
sign = lambda x: 1 if x >= 0 else -1 # incorrect sign function at x=0 but the case when x=0 won't be used in this program
get_row = lambda size, vals, idx: vals[size*idx: size*(idx+1)]
get_col = lambda size, vals, idx: vals[idx::size]

board_str_to_id = lambda board: [15 if c == '.' else ord(c)-65 for c in board]
str_to_arr = lambda boards: np.array([board_str_to_id(board) for board in boards], dtype=np.int64)

# returns a class-like structure for a state without all the overhead of classes
def gen_state(vals: str, mode='astar') -> Tuple:
    return (0, vals, '', 0)

def print_puzzle(state: str) -> None:
    size = calc_board_size(state)
    assert len(state) == size ** 2

    for i in range(size):
        for j in range(size):
            print(state[i * size + j], end=' ')
        print()

def find_goal(state: str) -> str:
    sorted_state = sorted(state)
    return ''.join(sorted_state[1:]) + sorted_state[0] # . is before any number/letter in ASCII so it's always first in sorted_state

# swap function but not assuming that ind0 < ind1
def swap_no_assume(text: str, ind0: int, ind1: int) -> str:
    if ind0 < ind1:
        return swap(text, ind0, ind1)
    return swap(text, ind1, ind0)

# takes in a word, returns its neighbors
def get_children(vals: str) -> List[str]:
    ret = []
    board_size = calc_board_size(vals)
    empty_ind = vals.index('.')
    empty_coord = get_coords(board_size, empty_ind)
    eyd, ey, eyi = empty_coord[0] - 1, empty_coord[0], empty_coord[0] + 1 # empty coord y decrement/increment
    exd, ex, exi = empty_coord[1] - 1, empty_coord[1], empty_coord[1] + 1

    ret = [swap_no_assume(vals, empty_ind, get_idx(board_size, *char_coord)) for condition, char_coord in (
        (ey > 0, (eyd, ex)),
        (ex > 0, (ey, exd)),
        (ey < board_size - 1, (eyi, ex)), # -1 since that's the max coord that can be indexed within a list
        (ex < board_size - 1, (ey, exi)),
    ) if condition]

    return ret

# checks if a board is solvable based on parity
def parity_check(state: str) -> bool:
    size = calc_board_size(state)
    num_ooop = 0 # number of out-of-order-pairs
    dot_idx = state.index('.')

    s = state.replace('.', '')
    for i in range(len(s)):
        char1 = s[i]
        for j in range(i+1, len(s)):
            if char1 > s[j]:
                num_ooop += 1
    if size % 2 == 1: # odd-sized boards - only even parity boards are solvable
        return num_ooop % 2 == 0

    dot_y = dot_idx // size # y-val of empty space in board
    return (num_ooop + dot_y) % 2 == 1 # returns true if num_ooop is even/odd and the dot is on an odd/even-indexed row (last row is odd-indexed)

#device = 'cpu'
USE_AUTOCAST = device == 'cuda'
model = get_saved_model().to(device)

def nn_heur(boards):
    with torch.autocast(device, enabled=USE_AUTOCAST):
        with torch.no_grad():
            x_pred = torch.from_numpy(str_to_arr(boards)).to(device)
            j = model(x_pred)
    return j[:, 0].round().cpu().numpy().tolist()

import matplotlib.pyplot as plt

def j_approx(h_parents, h_children, num_actions):
    ret = h_children.copy()
    s_i = 0
    for state_idx, num_neighbors in enumerate(num_actions):
        if num_neighbors == 0:
            continue
        e_i = s_i + num_neighbors
        j_gain = abs(min(h_children[s_i:e_i]) - h_parents[state_idx] + 1)
        for i in range(s_i, e_i):
            ret[i] = j_gain
        s_i = e_i
    return ret

def a_star(state: Tuple[float, str, str, int], goal: str, batch_size: int = 128) -> Tuple[int, Tuple[int, str, str, int]]:
    num_nodes_visited = 0
    closed = set()
    _f, s, p, depth = state
    h = nn_heur([s])[0]
    state = (h, s, p, depth)
    fringe = [state]

    while fringe:
        unvisited_children = []
        unvisited_children_depths = []
        children_popped = 0
        
        #f_vals = []
        while fringe and children_popped < batch_size:
            state = heapq.heappop(fringe)
            children_popped += 1
            num_nodes_visited += 1
            f, s, p, depth = state

            if s == goal:
                return depth, state, num_nodes_visited
            
            if s not in closed:
                closed.add(s)
                children = get_children(s)
                valid_children = [child for child in children if child not in closed]
                unvisited_children += valid_children
                unvisited_children_depths += [depth for _ in range(len(valid_children))]

        if unvisited_children == []:
            continue
        h_values = nn_heur(unvisited_children)
        #h_errs = j_approx(h_parents, nn_heur(unvisited_children), num_actions)
        updated_path = f'{s} {p}'
        for child, h, depth in zip(unvisited_children, h_values, unvisited_children_depths):
            f_value = (depth + 1) + h # depth + 1 represents the depth of the child node
            heapq.heappush(fringe, (f_value, child, updated_path, depth+1))

    return -1, None

def print_solution(terminal_state: Tuple[str, str, int, Set[str]]) -> None:
    state, path, _depth, _ancestors = terminal_state
    list(map(print, reversed([state, *path.split()])))

#import cProfile, pstats, io
#from pstats import SortKey
#pr = cProfile.Profile()
#pr.enable()

def transpose_board(board_str, flip_x=False, flip_y=False):
    board_size = int(len(board_str) ** 0.5)
    if not (flip_x or flip_y):
        return board_str
    if flip_x and flip_y:
        transposed_board_str = board_str[::-1]
    elif flip_x:
        rows = [board_str[board_size*i:board_size*(i+1)] for i in range(board_size)]
        transposed_board_str = ''.join([row[::-1] for row in rows])
    elif flip_y:
        #cols = [board_str[i::board_size] for i in range(board_size)]
        rows = [board_str[board_size*i:board_size*(i+1)] for i in range(board_size)]
        transposed_board_str = ''.join(rows[::-1])
        #return ''.join([col[::-1] for col in cols])
    return transposed_board_str

REF_BOARD_STR = puzzles[0]
BOARD_SIZE = int(len(REF_BOARD_STR) ** 0.5)
FLIP_X = FLIP_Y = False
if REF_BOARD_STR.index('.') == 0:
    FLIP_X = FLIP_Y = True
elif REF_BOARD_STR.index('.') == BOARD_SIZE-1:
    FLIP_Y = True
elif REF_BOARD_STR.index('.') == len(REF_BOARD_STR) - BOARD_SIZE:
    FLIP_X = True

CHAR_TRANSPOSE_PAIRS = {c1: c2 for c1, c2 in zip(transpose_board(REF_BOARD_STR, FLIP_X, FLIP_Y), find_goal(REF_BOARD_STR))}

very_start = time.perf_counter()
for idx, line in enumerate(puzzles):
    start = time.perf_counter()
    tokens = line.split()
    #state = transpose_board(tokens[0], FLIP_X, FLIP_Y)
    state = ''.join([CHAR_TRANSPOSE_PAIRS[c] for c in transpose_board(tokens[0], FLIP_X, FLIP_Y)])
    goal_state = find_goal(state)

    if not parity_check(state):
        print(f'Line {idx}: No possible solution found in {time.perf_counter() - start} seconds\n')
        continue

    start = time.perf_counter()
    solution_depth, solution, num_nodes_visited = a_star(gen_state(state), goal_state)
    duration = time.perf_counter() - start
    print(f'Line {idx}: {state}, Neural Network A* - {solution_depth} moves found in {duration:.3f} seconds')
    print(f'{num_nodes_visited} nodes visited, {round(num_nodes_visited/duration, 3)} nodes/s')
    #start = time.perf_counter()
    #solution_depth, solution, num_nodes_visited = a_star(gen_state(state), goal_state, heuristic=better_taxicab)
    #duration = time.perf_counter() - start
    #print(f'Manhattan A* - {solution_depth} moves found in {duration:.3f} seconds')
    #print(f'{num_nodes_visited} nodes visited, {round(num_nodes_visited/duration, 3)} nodes/s')
    
    print()

    #pr.disable()
    #s = io.StringIO()
    #sortby = SortKey.CUMULATIVE
    #ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    #ps.print_stats()
    #print(s.getvalue())
    #raise

print(f'Time to run all tests: {time.perf_counter() - very_start}')
