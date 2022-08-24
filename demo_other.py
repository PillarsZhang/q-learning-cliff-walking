from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from common import Env
from fig import get_visual_q_fig, save_figs
from utils import reset_random_seed, get_epsilon

if __name__ == "__main__":

    saved_path = Path("saved/demo_other")
    saved_path.mkdir(exist_ok=True, parents=True)

    # Figure: Empty envirment
    env = Env.get_standard()
    fig, _ = get_visual_q_fig(env)
    save_figs(fig, saved_path / "demo_standard_env.pdf")

    reset_random_seed()
    env = Env.get_advanced()
    # fig, _ = get_visual_q_fig(env, np_q=np.random.rand(*env.map_size, 4)*-10)
    fig, _ = get_visual_q_fig(env)
    save_figs(fig, saved_path / "demo_advanced_env.pdf")

    reset_random_seed()
    env = Env.get_advanced(is_rand=True, is_large=True)
    # fig, _ = get_visual_q_fig(env, np_q=np.random.rand(*env.map_size, 4)*-10)
    fig, _ = get_visual_q_fig(env)
    save_figs(fig, saved_path / "demo_advanced_env_rand_large.pdf")

    # Figure: Epsilon
    eps_range = (0.9, 0.05)
    eps_deacy_list = [100, 250, 500]
    num_episodes = int(2.5e3)
    np_episode = np.arange(num_episodes)
    fig, ax = plt.subplots(figsize=(4, 2), constrained_layout=True)

    for _idx in (0, 1):
        ax.plot((np_episode[0], np_episode[-1]), (eps_range[_idx], eps_range[_idx]), 
            '--', linewidth=1, c='black') 
    cmap = plt.cm.get_cmap("Set1")
    for _idx, eps_deacy in enumerate(eps_deacy_list):
        np_epsilon = get_epsilon(np_episode, eps_range, eps_deacy)
        ax.plot(np_episode, np_epsilon, 
            label=f"$\\epsilon_{{\\mathrm{{deacy}}}}={eps_deacy}$", 
            c=cmap.colors[_idx])

    ax.legend()
    ax.set_xlabel("Episode")
    ax.set_ylabel("$\\epsilon$")
    save_figs(fig, saved_path / "demo_epsilon_deacy.pdf")
