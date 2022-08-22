# Q Learning Cliff Walking (Q table and DQN)

## Highlights

## Usage

It is recommended to use VSCode for debugging. I have preset `.vscode/launch.json`.

### Standard Cliff Walking (Solution based on Q table)

The effect should be the same as that of `CliffWalking-v0` of gym.

```
# train
python standard_qtable.py
# demo
python demo_standard_qtable.py --id latest --run
```

### Advanced Cliff Walking (Solution based on DQN)

It uses the same 4x12 map as `CliffWalking-v0`, but will randomly generate 10 cliffs (perhaps better named traps) and start-end points that can ensure connectivity.

```
# train
python advanced_dqn.py --device cuda:0
# bench
python bench_advanced_dqn.py --device cuda:0
# demo
python demo_advanced_dqn.py --device cuda:0 --id latest --run
```

### Advanced Cliff Walking with an indefinite number of cliffs (Solution based on DQN)

Same as above, but the number of cliffs is random between 0-10.

```
# train
python advanced_dqn.py --device cuda:0 --rand
# bench
python bench_advanced_dqn.py --device cuda:0 --rand
# demo
python demo_advanced_dqn.py --device cuda:0 --id latest --run --rand
```

## Note

### gamma ($\gamma$)

gamma is the discount factor. It quantifies how much importance we give for future rewards. It’s also handy to approximate the noise in future rewards. Gamma varies from 0 to 1. If Gamma is closer to zero, the agent will tend to consider only immediate rewards. If Gamma is closer to one, the agent will consider future rewards with greater weight, willing to delay the reward. [[source](https://towardsdatascience.com/practical-reinforcement-learning-02-getting-started-with-q-learning-582f63e4acd9)]