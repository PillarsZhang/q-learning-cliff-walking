import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from common import Env, Agent, PositionList
from utils import get_visual_q_fig, get_reward_and_epsilon_fig, reset_random_seed
from standard_qtable import QTableModel, StatusCounter

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, default="latest")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    # For reproducibility
    reset_random_seed((147, 157, 167))

    env = Env.get_standard()
    model = QTableModel(map_size=env.map_size)
    agent = Agent(env, model)

    max_steps = np.prod(env.map_size)

    if args.id == "latest":
        _p = Path(f"saved/standard_qtable/")
        _id_list = [x.name for x in _p.iterdir() if x.is_dir()]
        assert len(_id_list) > 0, Exception("Checkpoint not found.")
        args.id = max(_id_list, key=float)
    # Restore status
    _q_saved_path = Path(_p / f"{args.id}")
    with open(_q_saved_path / "qtable.npy", 'rb') as f:
        agent.model.q = np.load(f)
    with open(_q_saved_path / "status_counter_dic_list.json", 'r') as f:
        _json = json.load(f)
        status_counter_list = list(map(lambda dic: StatusCounter(**dic), _json))

    saved_path = Path(f"saved/demo_standard_qtable")
    saved_path.mkdir(exist_ok=True, parents=True)

    # Exponential moving average
    avg_alpha = 2.5e-2
    np_episode = np.array([x.episode for x in status_counter_list])
    np_reward = np.array([x.reward for x in status_counter_list])
    np_epsilon = np.array([x.epsilon for x in status_counter_list])

    # Figure: reward_and_epsilon_while_train
    fig = get_reward_and_epsilon_fig(np_episode, np_reward, np_epsilon, avg_alpha)
    fig.savefig(saved_path / "reward_and_epsilon_while_train.pdf", bbox_inches='tight')
    plt.close(fig)

    if args.run:
        # Run once environment
        status_counter = StatusCounter()
        track_list: PositionList = []
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
