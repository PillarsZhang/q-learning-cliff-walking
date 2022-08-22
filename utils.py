from __future__ import annotations

import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import ticker

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from common import Env, Position, PositionList

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

def get_reward_and_epsilon_fig(np_episode, np_reward, np_epsilon, avg_alpha, alpha=0.2):
    np_reward_ema = get_ema(np_reward, avg_alpha)

    # fig, ax = plt.subplots(figsize=(4, 2), constrained_layout=True)
    # l1, = ax.plot(np_episode, np_reward, color="b", linewidth=0.4, alpha=0.5, label="Reward (raw)")
    fig, ax = plt.subplots(figsize=(6, 2), constrained_layout=True)
    l1 = ax.scatter(np_episode, np_reward, color="b", alpha=alpha, s=1, label="Reward (raw)")
    l2, = ax.plot(np_episode, np_reward_ema, color="r", linewidth=1.5, label=f"Reward ($\\alpha_{{EMA}}={avg_alpha}$)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")

    axl = ax.twinx()
    l3, = axl.plot(np_episode, np_epsilon, linestyle="--", color="y", label="$\\epsilon$")
    axl.set_ylabel("$\\epsilon$")

    ax.legend(handles=[l1, l2, l3], loc='center left', bbox_to_anchor=(1.2, 0.5))
    return fig

def get_visual_q_fig(np_q:np.ndarray, env:Env, track_list:PositionList=None):

    fig, ax = plt.subplots(figsize=(5, 2.5), constrained_layout=True)
    np_value = np_q.max(axis=2)

    cmap = plt.cm.get_cmap()
    im = ax.imshow(np_value, cmap=cmap)

    cbar = ax.figure.colorbar(im, ax=ax, location='bottom')
    cbar.ax.spines[:].set_visible(False)
    cbar.ax.text(0.5, 0.5, 
        r"$\blacksquare$ $V(s_t)=\max_{a}Q(s_t, a)$   $\bullet$ $Q(s_t, a)$   " +
        r"$\emptyset$ cliff   $\bigstar$ start   $\clubsuit$ end", 
        horizontalalignment='center', verticalalignment='center', transform=cbar.ax.transAxes, 
        color="white")

    # Turn spines off and create white grid (Possible zorder=2.5)
    ax.xaxis.tick_top()
    ax.set_axisbelow(False)
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(np_value.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(np_value.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", top=False, left=False)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(np_value.shape[1]), labels=np.arange(np_value.shape[1]))
    ax.set_yticks(np.arange(np_value.shape[0]), labels=np.arange(np_value.shape[0]))

    # Mark cliff and starting point (zorder=2.1, 2.8)
    for i, j in env.cliff_list:
        ax.add_patch(
            Rectangle(
                (j - 0.5, i - 0.5), 
                1, 1, alpha=1, edgecolor="red", facecolor="white", linewidth=6, zorder=2.1
            )
        )
        ax.text(
            j, i, r"$\emptyset$", 
            ha="center", va="center", color="black", fontsize=12, zorder=2.8
        )

    i, j = env.start_pos
    ax.add_patch(
        Rectangle(
            (j - 0.5, i - 0.5), 
            1, 1, alpha=1, edgecolor="green", facecolor="green", linewidth=8, zorder=2.1
        )
    )
    ax.text(
        j, i, r"$\bigstar$", 
        ha="center", va="center", color="w", fontsize=12, zorder=2.8
    )

    i, j = env.end_pos
    ax.add_patch(
        Rectangle(
            (j - 0.5, i - 0.5), 
            1, 1, alpha=1, edgecolor="blue", facecolor="blue", linewidth=8, zorder=2.1
        )
    )
    ax.text(
        j, i, r"$\clubsuit$", 
        ha="center", va="center", color="w", fontsize=12, zorder=2.8
    )

    # Indicates all Q(s_t, a)  (zorder=2.7)
    action_to_offset = np.array([[-1,0], [1,0], [0,-1], [0,1]])
    vmin, vmax = np_value.min(), np_value.max()
    _pad = 0.25
    for i in range(np_q.shape[0]):
        for j in range(np_q.shape[1]):
            _pos = np.array([i, j])
            is_out, is_drop, is_goal = env.inspect(_pos)
            if is_out or is_drop or is_goal: continue
            _np_q_s = np_q[i, j, :]
            _max_action = _np_q_s.argmax()
            for _action in np.arange(4):
                _offset = action_to_offset[_action] * _pad
                if _action == _max_action:
                    ax.scatter(
                        j+_offset[1], i+_offset[0], c=_np_q_s[_action], 
                        s=13, cmap=cmap, vmin=vmin, vmax=vmax, edgecolors="white", zorder=2.7
                    )
                else:
                    ax.scatter(
                        j+_offset[1], i+_offset[0], c=_np_q_s[_action], 
                        s=12, cmap=cmap, vmin=vmin, vmax=vmax, zorder=2.7
                    )

    if track_list is not None:
        # Show tracks  (zorder=2.6)
        i, j = track_list[0]
        ax.text(
            j+0.25, i-0.25, f"{0}", 
            ha="center", va="center", color="w", fontsize=8, zorder=2.61
        )
        i, j = track_list[-1]
        ax.text(
            j+0.25, i-0.25, f"{len(track_list)-1}", 
            ha="center", va="center", color="w", fontsize=8, zorder=2.61
        )
        for n, (i, j) in enumerate(track_list[1:-1]):
            ax.text(
                j, i, f"{n+1}", 
                ha="center", va="center", color="w", fontsize=8, zorder=2.61
            )
        # Line
        np_track = np.array(track_list)
        ax.plot(np_track[:,1], np_track[:,0], linewidth=5, color="r", alpha=0.4, zorder=2.6)

    return fig, ax

def sorted_pairs(lst_key, lst_value):
    # How to sort two lists together in Python
    # https://www.adamsmith.haus/python/answers/how-to-sort-two-lists-together-in-python
    # https://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list
    return tuple(list(tup) for tup in zip(*sorted(zip(lst_key, lst_value), key=lambda x: x[0])))


def get_result_fig(np_episode, np_result, possible_result_list, avg_alpha):
    np_result_onehot = np.eye(len(possible_result_list))[np_result]
    np_result_onehot_ema = get_ema(np_result_onehot, avg_alpha)
    np_result_onehot_ema2 = get_ema(np_result_onehot, avg_alpha*100)

    fig, ax = plt.subplots(figsize=(5, 2), constrained_layout=True)
    l_lst = []
    l_lst2 = []
    # https://stackoverflow.com/questions/4971269/how-to-pick-a-new-color-for-each-plotted-line-within-a-figure-in-matplotlib
    cmap = plt.cm.get_cmap("Set1")
    for _idx, _result in enumerate(possible_result_list):
        l_lst.append(ax.plot(np_episode, np_result_onehot_ema[:, _idx], label=_result.capitalize(), 
            zorder=12, c=cmap.colors[_idx]
        )[0])
        l_lst2.append(ax.plot(np_episode, np_result_onehot_ema2[:, _idx], label=_result.capitalize(), 
            zorder=11, alpha=0.2, linewidth=0.2, c=cmap.colors[_idx]
        )[0])
    ax.set_xlabel(f"Episode")
    ax.set_ylabel(f"Proportion\n($\\alpha_{{EMA}}={avg_alpha}$)")
    ax.legend(handles=l_lst, loc='center left', bbox_to_anchor=(1.05, 0.5))

    # https://atmamani.github.io/cheatsheets/matplotlib/matplotlib_2/#Numbers-on-axes-in-scientific-notation
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True) 
    formatter.set_powerlimits((-1,1)) 
    ax.xaxis.set_major_formatter(formatter)

    return fig

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
