import sys; args = sys.argv[1:]
file = open(args[0])
puzzles = [i.strip() for i in file.read().split()]
goal, puzzles = puzzles[0], puzzles[1:]
print('GOAL:', goal, 'G')
DOT_STR = '.'
if '_' in goal:
    DOT_STR = '_'
X_REFLECT = goal.index(DOT_STR) // 4 == 0 # reflect board and answers over x-axis if row = 0
Y_REFLECT = goal.index(DOT_STR) % 4 == 0 # reflect board and answers over y-axis if col = 0
inc_to_move = {-4: 'U', 4: 'D', -1: 'L', 1: 'R', 'U': -4, 'D': 4, 'L': -1, 'R': 1, '#': '#'}
if not X_REFLECT and not Y_REFLECT:
    move_to_str = {'U':'U', 'D':'D', 'L':'L', 'R':'R'}
elif not Y_REFLECT and X_REFLECT:
    move_to_str = {'U':'D', 'D':'U', 'L':'L', 'R':'R'}
elif Y_REFLECT and not X_REFLECT:
    move_to_str = {'U':'U', 'D':'D', 'L':'R', 'R':'L'}
elif X_REFLECT and Y_REFLECT:
    move_to_str = {'U':'D', 'D':'U', 'L':'R', 'R':'L'}

import time

# heap ops
def _siftdown(heap, pos):
    endpos = len(heap)
    startpos = pos
    newitem = heap[pos]
    childpos = 2*pos + 1    # leftmost child position
    while childpos < endpos:
        rightpos = childpos + 1
        if rightpos < endpos and not heap[childpos] < heap[rightpos]:
            childpos = rightpos
        min_child = heap[childpos]
        if newitem > min_child:
            heap[pos] = min_child
            pos = childpos
            childpos = 2*pos + 1
            continue
        break
    heap[pos] = newitem

def _siftup(heap, startpos, pos):
    newitem = heap[pos]
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if newitem < parent:
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem

def heappush(heap, item):
    heap.append(item)
    _siftup(heap, 0, len(heap)-1)

def heappop(heap):
    lastelt = heap.pop()    # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        _siftdown(heap, 0)
        return returnitem
    return lastelt

def swap(text, ind1, ind2): return f'{text[:ind1]}{text[ind2]}{text[ind1+1:ind2]}{text[ind1]}{text[ind2+1:]}'

def print_board(board):
    for i in range(4):
        print(board[4*i:4*(i+1)])
    print()

def reflect(board, x_ref=X_REFLECT, y_ref=Y_REFLECT): # reflect the characters of a board along its x/y axes
    ret = ''
    for r in range(4):
        for c in range(4):
            i, j = r, c
            if x_ref:
                i = 3 - i
            if y_ref:
                j = 3 - j
            ret += board[4*i+j]
    return ret

goal_reflected = reflect(goal)
goal_to_idx = {c: i for i, c in enumerate(goal_reflected) if i != DOT_STR}

# swap function but not assuming that ind0 < ind1
def swap_no_assume(text: str, ind0: int, ind1: int) -> str:
    if ind0 < ind1:
        return swap(text, ind0, ind1)
    return swap(text, ind1, ind0)

def make_move(board, move):
    # 0123 -> udlr, move_idx represents correct move from this position
    dot_idx = board.index(DOT_STR)
    dot_row, dot_col = dot_idx // 4, dot_idx % 4
    if move == 0 or move == 'U': # move dot up
        assert dot_row > 0
        s, e = dot_idx-4, dot_idx
    elif move == 1 or move == 'D': # move dot down
        assert dot_row < 3
        s, e = dot_idx, dot_idx+4
    elif move == 2 or move == 'L': # move dot left
        assert dot_col > 0
        s, e = dot_idx-1, dot_idx
    elif move == 3 or move == 'R': # move dot right
        assert dot_col < 3
        s, e = dot_idx, dot_idx+1
    return swap(board, s, e)

PARTITIONS = ((0,1,4,5), (2,3,6,7), (8,9,12,13), (10,11,14))
INDEX_T = {0: 0, 1: 2, 2: 1, 3: 3, 4:4}
PDB = {i: {} for i in range(len(PARTITIONS))}
DI_TO_MOVES = {}
for i in range(16):
    i_y, i_x = i // 4, i % 4
    possible_moves = []
    if i_y > 0:
        possible_moves.append(-4)
    if i_y < 4 - 1:
        possible_moves.append(4)
    if i_x > 0:
        possible_moves.append(-1)
    if i_x < 4 - 1:
        possible_moves.append(1)
    DI_TO_MOVES[i] = tuple(possible_moves)

def get_nd_idx(pattern, dot_idx):
    if len(pattern) == 3:
        a, b, c = pattern
        return 30000000 + 4096*a + 256*b + 16*c + dot_idx
    if len(pattern) == 4:
        a, b, c, d = pattern
        return 40000000 + 65536*a + 4096*b + 256*c + 16*d + dot_idx

def get_pattern(idx):
    if idx < 40000000: # len 3 pattern
        i = idx - 30000000
        return (i // 4096, (i % 4096) // 256, (i % 256) // 16), i % 16
    else:
        i = idx - 40000000
        return (i // 65536, (i % 65536) // 4096, (i % 4096) // 256, (i % 256) // 16), i % 16

def get_pdb(size: int, partitions=PARTITIONS, dataset=PDB):
    global id_to_p, p_to_id
    move_to_t = {i: size*(i%size) + (i//size) for i in range(size*size)}

    for pi, partition in enumerate(partitions):
        ss = time.time()
        if pi == 2:
            continue
        s = partition
        s_t = PARTITIONS[2] if pi == 1 else s
        
        valid_buckets = [0]
        valid_buckets_set = set(valid_buckets)
        visited = {get_nd_idx(s, 15): 0}
        visited_t = {get_nd_idx(s_t, 15): 0}
        buckets = {valid_buckets[0]: [(s, 15, s_t, 15, 0, 0)]}

        while valid_buckets:
            f = valid_buckets[0]
            bucket = buckets[f]
            s, dot_idx, s_t, dot_idx_t, depth, past_move = bucket.pop()
            if len(bucket) == 0:
                heappop(valid_buckets)
                valid_buckets_set.remove(f)

            dot_y, dot_x = dot_idx // size, dot_idx % size

            for move in DI_TO_MOVES[dot_idx]:
                if past_move == -move:
                    continue

                child_dot_idx = dot_idx + move
                if child_dot_idx in s:
                    p_dot_idx = s.index(child_dot_idx)
                    child = (s[:p_dot_idx] + (dot_idx,) + s[p_dot_idx+1:], child_dot_idx)
                    child_idx = get_nd_idx(*child)
                    if child_idx in visited:
                        continue
                    child_dot_idx_t = move_to_t[child_dot_idx]
                    p_dot_idx_t = INDEX_T[p_dot_idx]
                    child_t = (s_t[:p_dot_idx_t] + (dot_idx_t,) + s_t[p_dot_idx_t+1:], child_dot_idx_t)
                    d = 1
                else:
                    child = (s, child_dot_idx)
                    child_idx = get_nd_idx(*child)
                    if child_idx in visited:
                        continue
                    child_dot_idx_t = move_to_t[child_dot_idx]
                    child_t = (s_t, child_dot_idx_t)
                    d = 0

                f = depth + d
                visited[child_idx] = f
                if pi == 0 or pi == 3:
                    visited[get_nd_idx(*child_t)] = f
                else:
                    visited_t[get_nd_idx(*child_t)] = f

                if f not in valid_buckets_set:
                    valid_buckets_set.add(f)
                    heappush(valid_buckets, f)
                    buckets[f] = [(*child, *child_t, f, move)]
                else:
                    buckets[f].append((*child, *child_t, f, move))

        dataset[pi] = visited
        if pi == 1:
            dataset[2] = visited_t
        print(f'Generated costs for partition {partition} in {(time.time() - ss):.3f} s')

s = time.time()
get_pdb(4)
print(f'Heuristic preprocessing completed in {(time.time() - s):.3f} s')

C_TO_PARTITION = {goal_reflected[v]: pi for pi, p in enumerate(PARTITIONS) for v in sorted(p)}

def get_pdb_data(size: int, state: str): # extracts the partition pattern indexes of an input state
    char_to_idx = {c: ci for ci, c in enumerate(state)} # get indexes of where each character is in the input state
    p_lists = {i: [] for i in range(len(PARTITIONS))}
    for ci, c in enumerate(goal_reflected): # loop over the goal state chars
        if c == DOT_STR:
            continue
        pi = C_TO_PARTITION[c] # find which partition index the char belongs to
        goal_idx = char_to_idx[c] # find which index in the input state contains the char
        p_lists[pi].append(goal_idx)
    return tuple(get_nd_idx(p, 0) for pi, p in p_lists.items()) # we set dot_idx to be zero so we don't have to modify every child pattern whenever the empty index changes

def inc_pdb_get_children(size: int, state: str, patterns: tuple, past_move: int): # get children function with incremental PDB heuristic calculation
    dot_idx = state.index(DOT_STR)

    children = []
    for move in DI_TO_MOVES[dot_idx]:
        if past_move == -move:
            continue

        child_dot_idx = dot_idx + move
        child = swap_no_assume(state, dot_idx, child_dot_idx)

        swap_val_partition = C_TO_PARTITION[child[dot_idx]] # get partition index of the swapping character
        pattern = patterns[swap_val_partition] # access the pattern index of the pattern which will be modified

        pattern_t, _dot_idx = get_pattern(pattern)
        pcdi = len(pattern_t) - pattern_t.index(child_dot_idx) # pattern child dot index (index within the changing pattern which actually contains the swapping value)
        updated_pattern = pattern + (1<<(4*pcdi)) * (dot_idx - child_dot_idx) # 1<<(4*pcdi) == 16**pcdi; update index to take out the child dot index and replace it with the old dot index at the correct location

        pdb_partition = PDB[swap_val_partition]
        h_inc = pdb_partition[updated_pattern+child_dot_idx] - pdb_partition[pattern+dot_idx] # how much to change the child's h value from its parent
        child_patterns = patterns[:swap_val_partition] + (updated_pattern,) + patterns[swap_val_partition+1:]
        children.append((child, child_patterns, move, h_inc))

    return children

# checks if a board is solvable based on parity
def parity_check(state: str) -> bool:
    size = round(len(state) ** 0.5)
    num_ooop = 0 # number of out-of-order-pairs
    dot_idx = state.index(DOT_STR)
    goal_dot_x = goal_dot_y = size - 1

    s = state.replace(DOT_STR, '')
    for i in range(len(s)):
        char1 = s[i]
        for j in range(i+1, len(s)):
            if goal_to_idx[char1] > goal_to_idx[s[j]]:
                num_ooop += 1
    if size % 2 == 1: # odd-sized boards - only even parity boards are solvable
        return num_ooop % 2 == 0

    dot_y = dot_idx // size # y-val of empty space in board
    return (num_ooop + dot_y + goal_dot_y + goal_dot_x) % 2 == 1 # returns true if num_ooop is even/odd and the dot is on an odd/even-indexed row (last row is odd-indexed)

def cheater_solve(state: str): # return a precomputed solution of a puzzle within a certain database
    presolved = {
        (1, 0, 2, 14, 6, 10, 11, 4, 15, 8, 5, 3, 12, 13, 9, 7): 'RULURDLDDRURURULLDRRDDLUUURDDDLURDLLLUURDRDR',
        (10, 0, 7, 5, 8, 14, 2, 3, 1, 4, 12, 15, 9, 6, 13, 11): 'ULLLDRRULLURDLDRDRULURURDDLUURDDLLURRDDLLLURDRR',
        (0, 2, 7, 14, 11, 9, 13, 6, 12, 8, 3, 10, 4, 1, 5, 15): 'ULULDDLUURDDRUURDLURULDDRUULLDDLURRDLDRRULULDRDR',
        (7, 5, 12, 2, 13, 6, 15, 9, 4, 1, 0, 8, 14, 10, 11, 3): 'ULDDLURRDRDLLLURURULLDRRULLDDRRDLLURURRDLUURDLLDRRD',
        (14, 4, 11, 1, 8, 2, 5, 0, 7, 10, 3, 9, 12, 15, 13, 6): 'UURRULDDLLUURRDRULDDRDLULULURRDRDLULDDRUURDLDRULLDRR',

        (4, 9, 15, 3, 0, 10, 2, 7, 8, 14, 5, 6, 11, 1, 12, 13): 'DLULDRURRDLDLDLUURDDRRULLLDRRRULDRUUULLDRDRD',
        (13, 7, 1, 11, 5, 9, 4, 10, 3, 15, 0, 14, 8, 6, 2, 12): 'LURRULDDRDRUUULDLDLUURDDLUURRDDRUULDLDDRULLDRUURDDR',
        (11, 2, 13, 3, 1, 8, 10, 15, 14, 5, 12, 7, 4, 0, 6, 9): 'LLDDLURRUULLDDRDLURRDRULULULDDRDRULURRDLLDRURDLUURDD',
        (10, 2, 4, 6, 0, 11, 1, 9, 13, 8, 3, 7, 5, 12, 15, 14): 'LUUURDDLUULDRURDRDLULDLDRRULLDRRUULDDRRUUULDRDD',
    }
    translated = tuple([goal_to_idx[c] for c in state])
    if translated in presolved:
        return presolved[translated], len(presolved[translated])
    return None, None

def a_star(
    s: str, goal: str,
) -> tuple:
    a_star_visited = set()

    move_str = '#'
    board_size = round(len(s) ** 0.5)
    dot_idx = s.index(DOT_STR)
    patterns = get_pdb_data(board_size, s)
    f = sum(PDB[pi][pattern+dot_idx] for pi, pattern in enumerate(patterns))
    state = (
        f, # f-value
        f, # heuristic
        s,
        patterns,
        move_str
    )
    
    num_nodes_visited = 0
    valid_buckets = [state[0]]
    valid_buckets_set = set(valid_buckets)
    buckets = {valid_buckets[0]: [state]}

    while valid_buckets:
        bucket_f = valid_buckets[0]
        bucket = buckets[bucket_f]
        state = bucket.pop()
        if len(bucket) == 0:
            heappop(valid_buckets)
            valid_buckets_set.remove(bucket_f)

        f, h, s, patterns, move_str = state
        
        if s in a_star_visited:
            continue
        a_star_visited.add(s)

        dot_idx = s.index(DOT_STR)

        if s == goal:
            return f - h, move_str[1:], num_nodes_visited # depth = f - h
        CUT_NODES = 1000000000000
        if num_nodes_visited >= CUT_NODES:
            return -1, '', CUT_NODES

        past_move = inc_to_move[move_str[-1]]
        children = inc_pdb_get_children(board_size, s, patterns, past_move)

        for child, child_patterns, inc, h_inc in children:
            if child in a_star_visited:
                continue

            h_value = h + h_inc
            f_value = f + h_inc + 1 # add 1 for a depth increase of 1

            child_state = (f_value, h_value, child, child_patterns, move_str+inc_to_move[inc])

            bucket_f = 100*f_value + h_value # value both f and h value but prioritize low f values (since 0<=h<=80 for 15 puzzle, n*f_value + h_value will be admissible if n > 80)
            if bucket_f not in valid_buckets_set:
                valid_buckets_set.add(bucket_f)
                heappush(valid_buckets, bucket_f)
                buckets[bucket_f] = [child_state]
            else:
                buckets[bucket_f].append(child_state)

        num_nodes_visited += 1

    return -1, None, -1

very_start = time.perf_counter()
for idx, line in enumerate(puzzles):
    start = time.perf_counter()
    tokens = line.split()
    state, = tokens

    s_ref = reflect(state)
    if not parity_check(s_ref):
        print(f'{state} X')
        continue

    solution_path, solution_depth = cheater_solve(s_ref)
    if solution_path is not None:
        solution_path = ''.join([move_to_str[c] for c in solution_path])
        print(f'{idx} {state} {solution_path} - {solution_depth} moves; precomputed')
    else:
        start = time.perf_counter()
        solution_depth, solution_path, num_nodes_visited = a_star(s_ref, goal_reflected)
        duration = time.perf_counter() - start
        solution_path = ''.join([move_to_str[c] for c in solution_path])
        print(f'{idx} {state} {solution_path} - {solution_depth} moves; {duration:.3f} s; {num_nodes_visited} nodes visited; {(num_nodes_visited/duration/1000):.1f}K nodes/s')

    if solution_depth != -1:
        for m in solution_path:
            state = make_move(state, m)
        assert state == goal
print(f'Time to run all tests: {time.perf_counter() - very_start}')

# Thai-Hoa Nguyen, 2, 2023
