import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
import numpy as np
from tqdm import tqdm
from common import Position, Env, Agent, ReplayMemory, Transition
from utils import get_epsilon, get_saved_suffix, reset_random_seed
from net import MLP, CNN
import torch
from torch import nn, optim

class QNetModel():
    def __init__(self, map_size: Position, device, num_action=4, gamma=0.9, alpha=1e-3, batch_size=128):
        self.device = device
        self.map_size = map_size
        self.num_action = num_action
        self.net_kwargs = dict(input_shape=(*self.map_size, 4), output_shape=(self.num_action,))
        self.policy_net: nn.Module = CNN(**self.net_kwargs).to(self.device)
        self.target_net: nn.Module = CNN(**self.net_kwargs).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.gamma = gamma
        self.alpha = alpha
        self.batch_size = batch_size

        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.alpha)
        self.memory = ReplayMemory(1024*8)

    def get_state(self, map_vec, pos) -> torch.Tensor:
        pos_vec = np.zeros((*self.map_size, 1))
        pos_vec[pos[0], pos[1], 0] = 1
        full_vec = np.concatenate((map_vec, pos_vec), axis=2)
        state = torch.tensor(full_vec).float().unsqueeze(0)
        return state

    def decide(self, state: torch.Tensor, epsilon=0) -> int:
        if np.random.rand() > epsilon:
            self.policy_net.eval()
            with torch.no_grad():
                state = state.to(self.device)
                action = self.policy_net(state).squeeze(0).argmax().item()
        else:
            action = np.random.randint(self.num_action)
        return action

    def update(self) -> float:
        if len(self.memory) < self.batch_size:
            return None
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        final_mask = torch.tensor([
            s is None for s in batch.next_state
        ], dtype=torch.bool)
        next_state_batch = torch.cat([
            x if x is not None else
                torch.zeros((1, *self.net_kwargs["input_shape"]))
            for x in batch.next_state
        ], dim=0).to(self.device)

        state_batch = torch.cat(batch.state, dim=0).to(self.device)
        action_batch = torch.tensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(batch.reward).unsqueeze(1).to(self.device)

        self.policy_net.train()
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = self.target_net(next_state_batch).max(dim=1, keepdim=True)[0].detach()
        next_state_values[final_mask, :] = 0
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = self.criterion(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item() / self.batch_size

    def sync(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_q(self, map_vec) -> np.ndarray:
        state_list = []
        ij_list = []
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                state_list.append(self.get_state(map_vec, (i, j)))
                ij_list.append((i, j))
        states = torch.cat(state_list, dim=0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            _q = self.policy_net(states).cpu().numpy()
        np_q = np.zeros((*self.map_size, 4))
        for n, (i, j) in enumerate(ij_list):
            np_q[i, j, :] = _q[n, :]
        return np_q

@dataclass
class StatusCounter():
    episode: int = 0
    step: int = 0

    drop: int = 0
    limit: int = 0
    goal: int = 0

    epsilon: float = 0
    reward: float = 0
    avg_reward: float = None
    avg_goal_per: float = None
    result: str = "unknow"

    check_cache: float = None

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--rand", action="store_true")
    parser.add_argument("--large", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    reset_random_seed()
    is_rand, is_large = args.rand, args.large
    saved_suffix = get_saved_suffix(is_rand, is_large)

    map_size = Env.get_advanced(is_rand=is_rand, is_large=is_large).map_size
    max_steps = np.prod(map_size)

    num_episodes = int(2e5)+1
    eps_range = (0.9, 0.05)
    eps_deacy = 5e4 # The smaller, the faster drop

    avg_alpha = 1e-3
    sync_interval = 16

    saved_path = Path(f"saved/advanced_dqn{saved_suffix}/{time.time():.3f}")
    saved_path.mkdir(exist_ok=True, parents=True)
    save_episodes = int(1e3)

    model = QNetModel(map_size=map_size, device=torch.device(args.device))

    pbar_episode = tqdm(range(num_episodes), desc="Train")
    status_counter = StatusCounter()
    status_counter_dic_list = []

    for status_counter.episode in pbar_episode:
        env = Env.get_advanced(is_rand=is_rand, is_large=is_large)
        agent = Agent(env, model)
        agent.reset()

        status_counter.check_cache = agent.env.check_cache
        epsilon = get_epsilon(status_counter.episode, eps_range, eps_deacy)
        pos = agent.pos
        status_counter.reward = 0
        for step in range(max_steps):
            status_counter.step += 1
            status_counter.epsilon = epsilon

            state = agent.model.get_state(agent.env.vec, pos)
            action = agent.model.decide(state, epsilon)
            pos, reward, result = agent.step(action)
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
            agent.model.memory.push(state, action, next_state, reward)
            if next_state is None:
                break
            else:
                state = next_state

        status_counter.avg_reward = status_counter.avg_reward*(1-avg_alpha) + \
            status_counter.reward*avg_alpha \
                if status_counter.avg_reward is not None else \
            status_counter.reward

        if result == "continue":
            status_counter.limit += 1
            status_counter.result = "limit"
        else:
            status_counter.result = result

        goal_per = 1 if status_counter.result == "goal" else 0
        status_counter.avg_goal_per = status_counter.avg_goal_per*(1-avg_alpha) + \
            goal_per*avg_alpha \
                if status_counter.avg_goal_per is not None else \
            goal_per

        status_counter.avg_loss = agent.model.update()

        status_counter_dic = asdict(status_counter)
        status_counter_dic_list.append(status_counter_dic)
        pbar_episode.set_postfix(status_counter_dic, refresh=False)

        if status_counter.episode % sync_interval == 0:
            agent.model.sync()

        if status_counter.episode % save_episodes == 0:
            weight_path = saved_path / f"episode={status_counter.episode:08d},avg_reward={status_counter.avg_reward:.3f}.pt"
            status_path = saved_path / "status_counter_dic_list.json"
            torch.save(agent.model.policy_net.state_dict(), weight_path)
            with open(status_path, 'w') as f:
                json.dump(status_counter_dic_list, f, indent=4)
