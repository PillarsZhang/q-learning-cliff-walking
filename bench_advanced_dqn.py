import argparse
from dataclasses import asdict
import json
import re
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from advanced_dqn import QNetModel, StatusCounter
from utils import get_saved_suffix, reset_random_seed, sorted_pairs
from common import Position, Env, Agent

def bench(
    weight_fn: Path,
    is_rand: bool, is_large: bool,
    device: torch.device,
    seed: tuple[float, float, float] = (147, 157, 167),
    num_episodes: int = 1000,
    is_track: bool = False
):
    reset_random_seed(seed)
    map_size = Env.get_advanced(is_rand=is_rand, is_large=is_large).map_size
    max_steps = np.prod(map_size)

    model = QNetModel(map_size=map_size, device=device)
    model.policy_net.load_state_dict(torch.load(weight_fn))

    status_counter = StatusCounter()
    status_counter_dic_list = []

    epsilon = 0
    for status_counter.episode in range(num_episodes):
        env = Env.get_advanced(is_rand=is_rand, is_large=is_large)
        agent = Agent(env, model)
        agent.reset()

        status_counter.check_cache = agent.env.check_cache
        pos = agent.pos
        status_counter.reward = 0
        track_list = [pos]
        for step in range(max_steps):
            status_counter.step += 1
            status_counter.epsilon = epsilon

            state = agent.model.get_state(agent.env.vec, pos)
            action = agent.model.decide(state, epsilon)
            pos, reward, result = agent.step(action)
            track_list.append(pos)
            status_counter.reward += reward
            match result:
                case "drop":
                    next_state = None
                    status_counter.drop += 1
                case "goal":
                    next_state = None
                    status_counter.goal += 1
                case _:
                    next_state = agent.model.get_state(agent.env.vec, pos)
            if next_state is None:
                break
            else:
                state = next_state

        if result == "continue":
            status_counter.limit += 1
            status_counter.result = "limit"
        else:
            status_counter.result = result

        status_counter_dic = asdict(status_counter)
        status_counter_dic_list.append(status_counter_dic)

    if is_track:
        np_q = agent.model.get_q(agent.env.vec)
        return np_q, agent.env, track_list, status_counter_dic
    else:
        return status_counter_dic_list

def get_weight_list(_q_saved_path: Path):
    weight_fn_list = list(_q_saved_path.glob(f"episode=*,*.pt"))
    _regexp = re.compile(r"episode=([0-9]+),.*")
    weight_episode_list = list(map(lambda fn: int(_regexp.search(fn.name).group(1)), weight_fn_list))
    weight_episode_list, weight_fn_list = sorted_pairs(weight_episode_list, weight_fn_list)
    return weight_episode_list, weight_fn_list

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--id", type=str, default="latest")
    parser.add_argument("--rand", action="store_true")
    parser.add_argument("--large", action="store_true")
    args = parser.parse_args()

    is_rand, is_large = args.rand, args.large
    saved_suffix = get_saved_suffix(is_rand, is_large)

    random_seed_for_bench = (247, 257, 267)

    _p = Path(f"saved/advanced_dqn{saved_suffix}")
    if args.id == "latest":
        _id_list = [x.name for x in _p.iterdir() if x.is_dir()]
        assert len(_id_list) > 0, Exception("Checkpoint not found.")
        args.id = max(_id_list, key=float)
    # Restore status
    _q_saved_path = Path(_p / f"{args.id}")
    # Get the weight corresponding to episode
    weight_episode_list, weight_fn_list = get_weight_list(_q_saved_path)
    print(f"Bench | num_weights: {len(weight_episode_list)}")

    # Prepare the save directory of bench results
    saved_path = Path(f"saved/bench_advanced_dqn{saved_suffix}/{args.id}")
    saved_path.mkdir(exist_ok=True, parents=True)

    # Bench all weights
    result_list_for_bench = []
    bench_pbar = tqdm(
        list(enumerate(zip(weight_episode_list, weight_fn_list))), 
        desc="Bench"
    )
    for idx, (weight_episode, weight_fn) in bench_pbar:
        status_counter_dic_list = bench(
            weight_fn=weight_fn,
            is_rand=is_rand, is_large=is_large, 
            device=torch.device(args.device),
            seed = random_seed_for_bench,
            num_episodes = 1000,
            is_track = False
        )
        result_list_for_bench.append(dict(
            weight_episode=weight_episode, 
            weight_fn=str(weight_fn), 
            status_counter_dic_list=status_counter_dic_list
        ))
        _result_list_in_dic = [dic["result"] for dic in status_counter_dic_list]
        _goal_prob = _result_list_in_dic.count("goal") / len(_result_list_in_dic)
        bench_pbar.set_postfix(dict(_goal_prob=_goal_prob))
        status_path = saved_path / "result_list_for_bench.json"
        with open(status_path, 'w') as f:
            json.dump(result_list_for_bench, f, indent=4)
