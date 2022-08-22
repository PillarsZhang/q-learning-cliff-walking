from __future__ import annotations

from collections import deque, namedtuple
import copy
import random
from typing import Union, Sequence
import numpy as np

from utils import shortestPathBinaryMatrix
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from standard_qtable import QTableModel
    from advanced_dqn import QNetModel
    QModel = Union[QTableModel, QNetModel]


Position = Union[Sequence[int], np.ndarray]
PositionList = Union[Sequence[Position], np.ndarray]

class Env():
    @classmethod
    def get_standard(cls) -> "Env":
        map_size = [4, 12]
        start_pos = [map_size[0]-1, 0]
        end_pos = [map_size[0]-1, map_size[1]-1]
        cliff_list = np.array([
            np.ones(map_size[1]-2)*(map_size[0]-1), 
            np.arange(1, map_size[1]-1)
        ]).T

        return cls(
            map_size=map_size,
            start_pos=start_pos,
            end_pos=end_pos,
            cliff_list=cliff_list
        )

    @classmethod
    def get_advanced(cls, map_size=[4, 12], is_rand_num_ciff=False, is_no_block=True) -> "Env":
        while True:
            num_ciff = np.random.randint(0, 11) if is_rand_num_ciff else 10
            signal_idx_list = np.random.permutation(np.prod(map_size))[:2+num_ciff]
            start_pos = np.unravel_index(signal_idx_list[0], map_size)
            end_pos = np.unravel_index(signal_idx_list[1], map_size)
            cliff_list = np.array(np.unravel_index(signal_idx_list[2:], map_size)).T
            env = cls(
                map_size=map_size,
                start_pos=start_pos,
                end_pos=end_pos,
                cliff_list=cliff_list
            )
            if (not is_no_block) or (env.check() >= 0): break
        return env

    def __init__(self, map_size: Position, start_pos: Position, end_pos: Position, cliff_list: PositionList) -> None:
        self.map_size = np.array(map_size, dtype=int)
        self.start_pos = np.array(start_pos, dtype=int)
        self.end_pos = np.array(end_pos, dtype=int)
        self.cliff_list = np.array(cliff_list, dtype=int)

        self.neg_pos = np.array([-1,-1])
        self.vec = self.vectorize()

    def inspect(self, pos_new: Position) -> tuple[bool, bool, bool]:
        is_out = ~((pos_new > self.neg_pos).all() & (pos_new < np.array(self.map_size)).all())
        is_drop = (pos_new == self.cliff_list).all(axis=1).any()
        is_goal = (pos_new == np.array(self.end_pos)).all()
        return is_out, is_drop, is_goal

    def vectorize(self) -> np.ndarray:
        vec = np.zeros((*self.map_size, 3))
        # vec[:,:,0] -> start_pos
        vec[self.start_pos[0], self.start_pos[1], 0] = 1
        # vec[:,:,1] -> end_pos
        vec[self.end_pos[0], self.end_pos[1], 1] = 1
        # vec[:,:,2] -> cliff
        vec[self.cliff_list[:,0], self.cliff_list[:,1], 2] = 1
        return vec

    def pos2idx(self, pos:Position):
        return pos[0]*self.map_size[1] + pos[1]
    def idx2pos(self, idx:int):
        return np.array([idx//self.map_size[1], idx%self.map_size[1]])

    def check(self):
        # Check the connectivity between the start point and the end point
        # https://leetcode.com/problems/shortest-path-in-binary-matrix/
        return shortestPathBinaryMatrix(self.vec[:,:,2], self.start_pos, self.end_pos)

class Agent():
    def __init__(self, env: Env, model: QModel, r_goal:float=0, r_drop:float=-100):
        self.env = env
        self.model = model

        self.r_goal = r_goal
        self.r_drop = r_drop

        self.action_to_offset = np.array([[-1,0], [1,0], [0,-1], [0,1]])
        self.action_to_arrow = ['↑', '↓', '←', '→']
        self.action_history = []
        self.reset()

    def reset(self) -> None:
        self.pos = np.array(self.env.start_pos)
        self.action_history = []

    def step(self, a) -> tuple[Position, float, str]:
        offset = self.action_to_offset[a, :]
        pos_new = self.pos + offset
        is_out, is_drop, is_goal = self.env.inspect(pos_new)
        if not is_out: self.pos = pos_new
        if is_drop:
            r = self.r_drop
            status = "drop"
        elif is_goal:
            r = self.r_goal
            status = "goal"
        else:
            r = -1
            status = "continue"
        self.action_history.append(self.pos)
        return self.pos, r, status

    def draw_stdout(self) -> None:
        if (self.model is not None) and hasattr(self.model, "q"):
            best_action = self.model.q.argmax(axis=2).tolist()
            to_arrow = [[self.action_to_arrow[x] for x in y] for y in best_action]
        else:
            to_arrow = [['+' for _ in range(self.env.map_size[1])] for _ in range(self.env.map_size[0])]
        add_signal = copy.deepcopy(to_arrow)
        for cliff in self.env.cliff_list:
            add_signal[cliff[0]][cliff[1]] = 'x'
        add_signal[self.env.start_pos[0]][self.env.start_pos[1]] = 's'
        add_signal[self.env.end_pos[0]][self.env.end_pos[1]] = 'e'
        output_str = '\n'.join((' '.join(y)) for y in add_signal)
        print(f"start: {self.env.start_pos} ({to_arrow[self.env.start_pos[0]][self.env.start_pos[1]]}), end: {self.env.end_pos}, num_cliff: {len(self.env.cliff_list)}")
        print(output_str)

# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
