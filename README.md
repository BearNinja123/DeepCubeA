# DeepCubeA
My implementation of Forest Agostinelli et. al's DeepCubeA algorithm on 15-puzzles. DeepCubeA aims to solve puzzles using A* search where the heuristic function is a neural network trained using value iteration. While this means that the search isn't guaranteed to be optimal (as the NN heuristic is not restricted to always meet the conditions for such a heuristic), it can find optimal solutions most of the time while searching far less nodes than other heuristics like pattern databases (PDBs).
Currently, my implementation uses a transformer rather than an MLP as described in the paper because it just works better...
Requires PyTorch (GPU highly recommended).

To train the network, run 
```python train_nn_heur.py```

to test the network (saved as slider_model_h.pth) on a set of slider puzzles (e.g. korf100.txt), run
```python a_star_nn.py <file>```

and to test the PDB heuristic program, run
```python pdb.py <file>```

Benchmarks (Tested on Dell Inspiron 16 Plus, i7-12700H CPU, 32 GB DDR5 RAM, 3060 Max-Q GPU)
15_puzzle.txt
|File|Metric|4-4-4-3 PDB|DeepCubeA|DeepCubeA improvement|
----------------------------------------
|15_puzzle.txt|Completion time (s)|11.2|8.9|1.26|
|korf100.txt|Completion time (s)|445.5|31.1|14.3|
|N/A|# nodes searched/s|~200K|~20K|0.1|

[DeepCubeA paper](https://deepcube.igb.uci.edu/static/files/SolvingTheRubiksCubeWithDeepReinforcementLearningAndSearch_Final.pdf)
