from __future__ import annotations

import numpy as np
import torch
import random

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from common import Position

def reset_random_seed(seed=(47, 57, 67)):
    torch.manual_seed(seed[0])
    np.random.seed(seed[1])
    random.seed(seed[2])

def get_epsilon(x, eps_range, eps_deacy):
    return eps_range[1] + \
        (eps_range[0] - eps_range[1]) * np.exp(-x / eps_deacy)

def get_ema(np_in, avg_alpha):
    np_out = np.array(np_in)
    for i in range(1, np_out.shape[0]):
        np_out[i] = np_out[i-1] * (1-avg_alpha) + np_out[i] * avg_alpha
    return np_out

def sorted_pairs(lst_key, lst_value):
    # How to sort two lists together in Python
    # https://www.adamsmith.haus/python/answers/how-to-sort-two-lists-together-in-python
    # https://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list
    return tuple(list(tup) for tup in zip(*sorted(zip(lst_key, lst_value), key=lambda x: x[0])))

# https://leetcode.com/problems/shortest-path-in-binary-matrix/discuss/312827/Python-Concise-BFS
# Awsome, modified, numpy
def shortestPathBinaryMatrix(grid:np.ndarray, a:Position, b:Position) -> int:
    grid = np.array(grid)
    a, b = tuple(a), tuple(b)
    if grid[a[0], a[1]] or grid[b[0], b[1]]: return -1
    q = [(*a, 0)]
    grid[a[0], a[1]] = 1
    for i, j, d in q:
        if (i,j) == b: return d
        for x, y in ((i-1,j), (i,j+1), (i+1,j), (i,j-1)):
            if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] and not grid[x, y]:
                grid[x, y] = 1
                q.append((x, y, d+1))
    return -1

def get_saved_suffix(is_rand:bool, is_large:bool):
    suffix = ""
    if is_rand: suffix += "_rand"
    if is_large: suffix += "_large"
    return suffix
