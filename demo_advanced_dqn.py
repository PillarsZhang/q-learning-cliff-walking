import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import re

from common import Env, Agent, PositionList
from utils import get_result_fig, get_visual_q_fig, get_reward_and_epsilon_fig, reset_random_seed, sorted_pairs
from advanced_dqn import QNetModel, StatusCounter

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--id", type=str, default="latest")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--rand", action="store_true")
    args = parser.parse_args()

    # For reproducibility
    reset_random_seed((147, 157, 167))
    is_rand_num_ciff = args.rand

    map_size = [4, 12]
    max_steps = np.prod(map_size)
    env = Env.get_advanced(map_size=map_size, is_rand_num_ciff=is_rand_num_ciff)
    model = QNetModel(map_size=map_size, device=torch.device(args.device))
    agent = Agent(env, model)

    if args.id == "latest":
        _p = Path(f"saved/advanced_dqn{'_rand_num_ciff' if is_rand_num_ciff else ''}")
        _id_list = [x.name for x in _p.iterdir() if x.is_dir()]
        assert len(_id_list) > 0, Exception("Checkpoint not found.")
        args.id = max(_id_list, key=float)
    # Restore status
    _q_saved_path = Path(_p / f"{args.id}")
    with open(_q_saved_path / "status_counter_dic_list.json", 'r') as f:
        _json = json.load(f)
        status_counter_list = list(map(lambda dic: StatusCounter(**dic), _json))

    saved_path = Path(f"saved/demo_advanced_dqn{'_rand_num_ciff' if is_rand_num_ciff else ''}")
    saved_path.mkdir(exist_ok=True, parents=True)

    # Get the weight corresponding to episode
    weight_fn_list = list(_q_saved_path.glob(f"episode=*,*.pt"))
    _regexp = re.compile(r"episode=([0-9]+),.*")
    weight_episode_list = list(map(lambda fn: int(_regexp.search(fn.name).group(1)), weight_fn_list))
    weight_episode_list, weight_fn_list = sorted_pairs(weight_episode_list, weight_fn_list)
    print(f"Demo | num_weights: {len(weight_episode_list)}")

    # Exponential moving average
    avg_alpha = 0.5e-3
    np_episode = np.array([x.episode for x in status_counter_list])
    np_reward = np.array([x.reward for x in status_counter_list])
    np_epsilon = np.array([x.epsilon for x in status_counter_list])

    # # Figure: reward_and_epsilon_while_train
    # fig = get_reward_and_epsilon_fig(np_episode, np_reward, np_epsilon, avg_alpha, alpha=0.02)
    # fig.savefig(saved_path / "reward_and_epsilon_while_train.pdf", bbox_inches='tight')
    # fig.savefig(saved_path / "reward_and_epsilon_while_train.png", bbox_inches='tight', dpi=300)
    # plt.close(fig)

    possible_result_list = ["drop", "limit", "goal"]
    np_result = np.array([possible_result_list.index(x.result) for x in status_counter_list])

    # # Figure: result_while_train
    # fig = get_result_fig(np_episode, np_result, possible_result_list, avg_alpha)
    # fig.savefig(saved_path / "result_while_train.pdf", bbox_inches='tight')
    # fig.savefig(saved_path / "result_while_train.png", bbox_inches='tight', dpi=300)
    # plt.close(fig)

    if args.run:
        # Run once environment
        status_counter = StatusCounter()
        track_list: PositionList = []
        epsilon = 0




        agent.reset()
        epsilon = 0
        status_counter.epsilon = epsilon
        s = agent.pos
        r_sum = 0
        for step in range(max_steps):
            track_list.append(s)
            status_counter.step += 1
            a = agent.model.decide(s, epsilon)
            s_new, r, status = agent.step(a)
            r_sum += r
            match status:
                case "drop" | "goal": break
                case _: s = s_new

        track_list.append(s_new)
        if status == "continue":
            status = "limit"
        status_counter.result = status
        status_counter.reward = r_sum

        agent.draw_stdout()
        print(status_counter)

        np_q = np.array(agent.model.q)
        # Figure: q_table_after_train
        fig, _ = get_visual_q_fig(np_q, agent.env, track_list)
        fig.savefig(saved_path / "visual_q_and_run.pdf", bbox_inches='tight')
        plt.close(fig)
