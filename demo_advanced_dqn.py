import argparse
import json
from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
from bench_advanced_dqn import bench, get_weight_list

from utils import get_saved_suffix, reset_random_seed
from fig import get_result_fig, get_result_for_bench_fig, get_reward_for_bench_fig, get_visual_q_fig, get_reward_and_epsilon_fig, save_figs
from advanced_dqn import StatusCounter

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--id", type=str, default="latest")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--rand", action="store_true")
    parser.add_argument("--large", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    # For reproducibility
    reset_random_seed((147, 157, 167))
    is_rand, is_large = args.rand, args.large
    saved_suffix = get_saved_suffix(is_rand, is_large)

    if args.id == "latest":
        _p = Path(f"saved/advanced_dqn{saved_suffix}")
        _id_list = [x.name for x in _p.iterdir() if x.is_dir()]
        assert len(_id_list) > 0, Exception("Checkpoint not found.")
        args.id = max(_id_list, key=float)
    # Restore status
    _q_saved_path = Path(_p / f"{args.id}")
    with open(_q_saved_path / "status_counter_dic_list.json", 'r') as f:
        _json = json.load(f)
        status_counter_list = list(map(lambda dic: StatusCounter(**dic), _json))

    saved_path = Path(f"saved/demo_advanced_dqn{saved_suffix}")
    saved_path.mkdir(exist_ok=True, parents=True)

    # Exponential moving average
    avg_alpha = 0.5e-3
    np_episode = np.array([x.episode for x in status_counter_list])
    np_reward = np.array([x.reward for x in status_counter_list])
    np_epsilon = np.array([x.epsilon for x in status_counter_list])

    # Figure: reward_and_epsilon_while_train
    fig = get_reward_and_epsilon_fig(np_episode, np_reward, np_epsilon, avg_alpha, alpha=0.02)
    save_figs(fig, saved_path / "reward_and_epsilon_while_train.pdf")

    possible_result_list = ["drop", "limit", "goal"]
    np_result = np.array([possible_result_list.index(x.result) for x in status_counter_list])

    # Figure: result_while_train
    fig = get_result_fig(np_episode, np_result, possible_result_list, avg_alpha)
    save_figs(fig, saved_path / "result_while_train.pdf")

    status_path = Path(f"saved/bench_advanced_dqn{saved_suffix}/{args.id}") / "result_list_for_bench.json"
    with open(status_path, 'r') as f:
        result_list_for_bench = json.load(f)

    _lst = []
    possible_result_list = ["drop", "limit", "goal"]
    for xb in result_list_for_bench:
        weight_episode = xb["weight_episode"]
        status_counter_dic_list = xb["status_counter_dic_list"]
        status_counter_list = list(map(lambda dic: StatusCounter(**dic), status_counter_dic_list))

        np_result = np.array([possible_result_list.index(x.result) for x in status_counter_list])
        np_result_onehot = np.eye(len(possible_result_list))[np_result]
        result_prob = np_result_onehot.mean(axis=0)

        actual_reward = np.array([x.reward for x in status_counter_list], dtype=float)
        for x in status_counter_list:
            if x.check_cache is None:
                if '_fake_ideal_reward_tip' not in locals():
                    _fake_ideal_reward_tip = "Prompt only when it appears for the first time"
                    print(f"Attention: _fake_ideal_reward_tip: {_fake_ideal_reward_tip}")
                    _fake_ideal_reward_rs = np.random.RandomState(123)
                x.check_cache = _fake_ideal_reward_rs.randint(0, 16)
        ideal_reward = np.array([-(x.check_cache) for x in status_counter_list], dtype=float)

        divide_zero = lambda a, b: np.divide(a, b, out=np.ones_like(a), where=b!=0)

        _lst.append((weight_episode, result_prob, actual_reward, ideal_reward))

    np_weight_episode, np_result_prob, np_actual_reward, np_ideal_reward = [np.array(x) for x in zip(*_lst)]

    # Figure: win_prob_in_bench
    fig = get_result_for_bench_fig(
        np_weight_episode, np_result_prob, possible_result_list, 
        avg_alpha=1e-1)
    save_figs(fig, saved_path / "win_prob_in_bench.pdf")

    # Figure: reward_in_bench
    fig = get_reward_for_bench_fig(np_weight_episode, np_actual_reward, np_ideal_reward)
    save_figs(fig, saved_path / "reward_in_bench.pdf")
    if args.show: plt.show()

    # Find weight episode of max goal prob.
    max_goal_prob_idx = np_result_prob[:, possible_result_list.index("goal")].argmax()
    max_goal_prob = np_result_prob[max_goal_prob_idx, possible_result_list.index("goal")]
    max_goal_prob_episode = np_weight_episode[max_goal_prob_idx]

    max_goal_prob_saved_path = saved_path / f"max_goal_prob.txt"
    max_goal_prob_str = (f"max_goal_prob_idx, max_goal_prob, max_goal_prob_episode\n"
        f"{max_goal_prob_idx}, {max_goal_prob}, {max_goal_prob_episode}\n")
    print(max_goal_prob_str)
    with open(max_goal_prob_saved_path, 'w') as f:
        f.write(max_goal_prob_str)

    if args.run:

        # Choose weight filename of max goal prob.
        weight_episode_list, weight_fn_list = get_weight_list(_q_saved_path)
        max_goal_prob_weight_fn = weight_fn_list[weight_episode_list.index(max_goal_prob_episode)]
        
        num_run = 20
        get_seed = lambda x: tuple(random.getrandbits(16) for _ in range(x))
        for idx_run in range(num_run):
            # Run once environment
            seed = get_seed(3)
            np_q, env, track_list, status_counter_dic = bench(
                weight_fn=max_goal_prob_weight_fn,
                is_rand=is_rand, is_large=is_large, 
                device=args.device,
                seed=seed,
                num_episodes=1,
                is_track=True
            )

            fig, _ = get_visual_q_fig(env, np_q, track_list)
            fig_saved_path = saved_path / f"visual_q_and_run_{idx_run}.pdf"
            save_figs(fig, fig_saved_path)

            osd_str = (
                    f"[result, torch/numpy/random-seed, actual/ideal-reward]\n"
                    f"[{status_counter_dic['result']}, {seed[0]}/{seed[1]}/{seed[2]}, "
                    f"{status_counter_dic['reward']}/{-(status_counter_dic['check_cache'])}]\n"
                )
            osd_saved_path = saved_path / f"visual_q_and_run_{idx_run}.txt"
            with open(osd_saved_path, 'w') as f:
                f.write(osd_str)
            
            print(f"idx_run: {idx_run}, seed: {seed}, fig_saved_path: {fig_saved_path}")
            print(f"osd_str: \n{osd_str}")
            print(f"status_counter_dic: {status_counter_dic}")

            if args.show: plt.show()
