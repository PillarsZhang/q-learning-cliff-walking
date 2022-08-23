from __future__ import annotations
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import ticker
from matplotlib.collections import LineCollection

import scipy.stats as st

from utils import get_ema
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from common import Env, PositionList

def get_reward_and_epsilon_fig(np_episode, np_reward, np_epsilon, avg_alpha, alpha=0.2):
    np_reward_ema = get_ema(np_reward, avg_alpha)

    # fig, ax = plt.subplots(figsize=(4, 2), constrained_layout=True)
    # l1, = ax.plot(np_episode, np_reward, color="b", linewidth=0.4, alpha=0.5, label="Reward (raw)")
    fig, ax = plt.subplots(figsize=(6, 2), constrained_layout=True)
    l1 = ax.scatter(np_episode, np_reward, color="b", alpha=alpha, s=1, label="Reward (raw)", rasterized=True)
    l2, = ax.plot(np_episode, np_reward_ema, color="r", linewidth=1.5, label=f"Reward ($\\alpha_{{EMA}}={avg_alpha}$)", rasterized=True)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")

    axl = ax.twinx()
    l3, = axl.plot(np_episode, np_epsilon, linestyle="--", color="y", label="$\\epsilon$", rasterized=True)
    axl.set_ylabel("$\\epsilon$")

    ax.legend(handles=[l1, l2, l3], loc='center left', bbox_to_anchor=(1.2, 0.5))
    return fig

def get_visual_q_fig(env:Env, np_q:np.ndarray=None, track_list:PositionList=None):

    # Shunt with Q value
    if np_q is not None:
        fig, ax = plt.subplots(figsize=(5, 2.5), constrained_layout=True)
        np_value = np_q.max(axis=2)
        cmap = plt.cm.get_cmap()
        im = ax.imshow(np_value, cmap=cmap)
    else:
        fig, ax = plt.subplots(figsize=(5, 2.0), constrained_layout=True)
        np_value = np.ones(env.map_size) * 0.1
        cmap = plt.cm.get_cmap("Blues")
        im = ax.imshow(np_value, cmap=cmap, vmin=0, vmax=1)

    if np_q is not None:
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
            1, 1, alpha=1, edgecolor="green", facecolor="green", linewidth=6, zorder=2.1
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
            1, 1, alpha=1, edgecolor="blue", facecolor="blue", linewidth=6, zorder=2.1
        )
    )
    ax.text(
        j, i, r"$\clubsuit$", 
        ha="center", va="center", color="w", fontsize=12, zorder=2.8
    )

    if np_q is not None:
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
        track_list = [tuple(pos) for pos in track_list]
        # Show tracks  (zorder=2.6) (text zorder=2.71)
        i, j = track_list[0]
        ax.text(
            j+0.25, i-0.25, f"{0}", 
            ha="center", va="center", color="w", fontsize=8, zorder=2.71
        )
        i, j = track_list[-1]
        ax.text(
            j+0.25, i-0.25, f"{len(track_list)-1}", 
            ha="center", va="center", color="w", fontsize=8, zorder=2.71
        )
        for n, (i, j) in enumerate(track_list[1:-1]):
            if track_list[1:-1].index((i, j)) == n:
                t = f"{n+1}" if track_list[1:-1].count((i, j)) == 1 else f"$\\circlearrowleft${n+1}"
                ax.text(
                    j, i, t, 
                    ha="center", va="center", color="w", fontsize=8, zorder=2.71
                )
        # Line
        np_track = np.array(track_list)
        ax.plot(np_track[:,1], np_track[:,0], linewidth=5, color="r", alpha=0.4, zorder=2.6)
        # plot_line_as_segments(np_track[:,1], np_track[:,0], ax=ax, linewidth=5, color="r", alpha=0.4, zorder=2.6)
        t_line_list = []
        for pos_idx in range(len(track_list)-1):
            t_line_list.append(tuple(sorted((track_list[pos_idx], track_list[pos_idx+1]))))
        c = Counter(t_line_list)
        for t_line, t_count in c.most_common():
            if t_count > 1:
                # ax.plot((t_line[0][1], t_line[1][1]), (t_line[0][0], t_line[1][0]), 
                #     linewidth=5, color="y", alpha=1, zorder=2.6)
                t_middle = (t_line[0][0] + t_line[1][0]) / 2, (t_line[0][1] + t_line[1][1]) / 2
                ax.text(
                    t_middle[1], t_middle[0], f"$\\rightleftharpoons${t_count}", 
                    ha="center", va="center", color="w", fontsize=6, zorder=2.71,
                    rotation=90 if t_line[0][0] == t_line[1][0] else 0,
                    bbox=dict(boxstyle="round",
                        fc="b", ec=None,
                        alpha=0.5
                    )
                )

    return fig, ax


def get_result_fig(np_episode:np.ndarray, np_result:np.ndarray, possible_result_list, avg_alpha):
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
            zorder=12, c=cmap.colors[_idx], 
        rasterized=True)[0])
        l_lst2.append(ax.plot(np_episode, np_result_onehot_ema2[:, _idx], label=_result.capitalize(), 
            zorder=11, alpha=0.2, linewidth=0.2, c=cmap.colors[_idx], 
        rasterized=True)[0])
    ax.set_xlabel(f"Episode")
    ax.set_ylabel(f"Proportion\n($\\alpha_{{EMA}}={avg_alpha}$)")
    ax.legend(handles=l_lst, loc='center left', bbox_to_anchor=(1.05, 0.5))
    ax.set_ylim(0, 1)

    # https://atmamani.github.io/cheatsheets/matplotlib/matplotlib_2/#Numbers-on-axes-in-scientific-notation
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True) 
    formatter.set_powerlimits((-1,1)) 
    ax.xaxis.set_major_formatter(formatter)

    return fig

def get_result_for_bench_fig(np_episode:np.ndarray, np_result_prob:np.ndarray, possible_result_list, avg_alpha):

    np_result_onehot_ema = get_ema(np_result_prob, avg_alpha)

    fig, ax = plt.subplots(figsize=(5, 2), constrained_layout=True)
    l_lst = []
    l_lst2 = []
    # https://stackoverflow.com/questions/4971269/how-to-pick-a-new-color-for-each-plotted-line-within-a-figure-in-matplotlib
    cmap = plt.cm.get_cmap("Set1")
    for _idx, _result in enumerate(possible_result_list):
        l_lst.append(ax.plot(np_episode, np_result_onehot_ema[:, _idx], label=_result.capitalize(), 
            zorder=12, c=cmap.colors[_idx], linewidth=2
        )[0])
        l_lst2.append(ax.plot(np_episode, np_result_prob[:, _idx], label=_result.capitalize(), 
            zorder=11, alpha=0.4, linewidth=1.5, c=cmap.colors[_idx]
        )[0])
    ax.set_xlabel(f"Checkpoint episode")
    ax.set_ylabel(f"Proportion\n($\\alpha_{{EMA}}={avg_alpha}$)")
    ax.legend(handles=l_lst, loc='center left', bbox_to_anchor=(1.05, 0.5))
    ax.set_ylim(0, 1)

    _idx = possible_result_list.index("goal")
    _idx_sca = np_result_prob[:, _idx].argmax()
    ax.scatter(np_episode[_idx_sca], np_result_prob[:, _idx][_idx_sca], c='y')

    # https://atmamani.github.io/cheatsheets/matplotlib/matplotlib_2/#Numbers-on-axes-in-scientific-notation
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True) 
    formatter.set_powerlimits((-1,1)) 
    ax.xaxis.set_major_formatter(formatter)

    return fig

def get_mean_confidence_interval(a: np.ndarray) -> tuple[float, float, float]:
    m = np.mean(a)
    c = st.t.interval(0.95, len(a)-1, loc=m, scale=st.sem(a))
    return m, *c

def get_reward_for_bench_fig(np_weight_episode, 
    np_actual_reward, np_ideal_reward):
    
    fig, ax = plt.subplots(figsize=(4, 2), constrained_layout=True)

    np_episode = np_weight_episode
    np_actual = np.array(list(map(get_mean_confidence_interval, np_actual_reward)))
    np_ideal = np.array(list(map(get_mean_confidence_interval, np_ideal_reward)))

    l1, = ax.plot(np_episode, np_actual[:, 0], color='b', label=r"$\mathrm{Reward}_{\mathrm{actual}}$")
    ax.fill_between(np_episode, np_actual[:, 1], np_actual[:, 2], color='b', alpha=.2)
    _idx_sca = np_actual[:, 0].argmax()
    ax.scatter(np_episode[_idx_sca], np_actual[:, 0][_idx_sca], c='y')
    _idx_sca = np_actual[:, 0].argmin()
    ax.scatter(np_episode[_idx_sca], np_actual[:, 0][_idx_sca], c='y')

    l2, = ax.plot(np_episode, np_ideal[:, 0], color='g', label=r"$\mathrm{Reward}_{\mathrm{ideal}}$")
    ax.fill_between(np_episode, np_ideal[:, 1], np_ideal[:, 2], color='g', alpha=.2)

    ax.plot([np_episode.min(), np_episode.max()], [-50, -50], "--", c="black", linewidth=1)
    ax.text(np_episode.max(), -50, "$\\pm 50$ ", ha="right", va="bottom")

    ax.set_xlim(np_episode.min(), np_episode.max())
    # ax.set_ylim(np_actual[:, 0].min()*2, 0)
    ax.set_ylim(-100, 0)
    ax.set_xlabel(f"Checkpoint episode")
    ax.set_ylabel(f"Reward \n(Mean and 95% \nconfidence interval)")

    np_gap = np.array(list(map(get_mean_confidence_interval, np_ideal_reward - np_actual_reward)))

    axl = ax.twinx()
    l3, = ax.plot(np_episode, np_gap[:, 0], color='r', label=r"$\mathrm{Reward}_{\mathrm{ideal}}-\mathrm{Reward}_{\mathrm{actual}}$")
    ax.fill_between(np_episode, np_gap[:, 1], np_gap[:, 2], color='r', alpha=.4)

    axl.fill_between(np_episode, np_gap[:, 0], color='r', label="Reward gap", alpha=.4)
    # axl.set_ylim(0, np_gap[:, 0].max()*2)
    axl.set_ylim(0, 100)

    # ax.legend(handles=[l1, l2, l3], loc='center left', bbox_to_anchor=(1.2, 0.5))
    ax.legend(handles=[l1, l2, l3])

    return fig

# https://stackoverflow.com/a/70544337/16407115
def plot_line_as_segments(xs, ys=None, ax=None, **kwargs):
    ax = ax or plt.gca()
    if ys is None:
        ys = xs
        xs = np.arange(len(ys))
    segments = np.c_[xs[:-1], ys[:-1], xs[1:], ys[1:]].reshape(-1, 2, 2)
    added_collection = ax.add_collection(LineCollection(segments, **kwargs))
    ax.autoscale()
    return added_collection

def save_figs(fig, fn:Path, suffixs:list[str]=[".pdf", ".png"], **kwargs):
    kwargs = dict(bbox_inches='tight', dpi=300) | kwargs
    for _suffix in suffixs:
        fig.savefig(fn.with_suffix(_suffix), **kwargs)
