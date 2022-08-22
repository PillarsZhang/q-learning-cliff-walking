from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
import numpy as np
from tqdm import tqdm
from common import Position, Env, Agent
from utils import get_epsilon, reset_random_seed

class QTableModel():
    def __init__(self, map_size: Position, num_action=4, gamma=0.9, alpha=0.05):
        # Q 表格
        self.q = np.zeros((map_size[0], map_size[1], num_action))
        self.gamma = gamma
        self.alpha = alpha

    def decide(self, s, epsilon=0):
        a = self.q[s[0], s[1], :].argmax() if np.random.rand() > epsilon else np.random.randint(4)
        return a

    def update(self, s, a, r, s_new):
        self.q[s[0], s[1], a] += self.alpha * (r + self.gamma*self.q[s_new[0], s_new[1], :].max() - self.q[s[0], s[1], a])

@dataclass
class StatusCounter():
    episode: int = 0
    step: int = 0

    epsilon: float = 0
    reward: float = 0
    result: str = "unknow"

if __name__ == "__main__":

    reset_random_seed()

    env = Env.get_standard()
    model = QTableModel(map_size=env.map_size)
    agent = Agent(env, model)

    num_episodes = int(2.5e3)
    max_steps = np.prod(env.map_size)

    eps_range = (0.9, 0.05)
    eps_deacy = 2.5e2

    saved_path = Path(f"saved/standard_qtable/{time.time():.3f}")
    saved_path.mkdir(exist_ok=True, parents=True)

    pbar = tqdm(range(num_episodes), desc="Train")
    status_counter = StatusCounter()
    status_counter_dic_list = []
    for status_counter.episode in pbar:
        agent.reset()
        epsilon = get_epsilon(status_counter.episode, eps_range, eps_deacy)
        status_counter.epsilon = epsilon
        s = agent.pos
        r_sum = 0
        for step in range(max_steps):
            status_counter.step += 1
            a = agent.model.decide(s, epsilon)
            s_new, r, status = agent.step(a)
            r_sum += r
            agent.model.update(s, a, r, s_new)
            match status:
                case "drop" | "goal": break
                case _: s = s_new

        if status == "continue":
            status = "limit"
        status_counter.result = status
        status_counter.reward = r_sum

        status_counter_dic = asdict(status_counter)
        status_counter_dic_list.append(status_counter_dic)
        pbar.set_postfix(status_counter_dic, refresh=False)

    agent.draw_stdout()
    with open(saved_path / "qtable.npy", 'wb') as f:
        np.save(f, agent.model.q, allow_pickle=False)
    with open(saved_path / "status_counter_dic_list.json", 'w') as f:
        json.dump(status_counter_dic_list, f, indent=4)
