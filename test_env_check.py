from common import Env, Agent
from tqdm import tqdm
import time

from utils import reset_random_seed

reset_random_seed()

print("is_no_block: True")
for i in tqdm(range(10), disable=True):
    env = Env.get_advanced(is_no_block=True)
    agent = Agent(env, None)
    agent.draw_stdout()
    print(agent.env.check())

print("is_no_block: False")
for i in tqdm(range(10), disable=True):
    env = Env.get_advanced(is_no_block=False)
    agent = Agent(env, None)
    agent.draw_stdout()
    print(agent.env.check())

num_test = int(1e6)
num_block = 0
tic = time.time()
for i in tqdm(range(num_test)):
    env = Env.get_advanced(is_no_block=False)
    if env.check() < 0:
        num_block += 1
toc = time.time() - tic

print(
    f"block_prob: {num_block}/{num_test}={num_block/num_test*100:.3f}%, "
    f"speed: {num_test/toc:.3f}it/s"
)

# Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz
# block_prob: 46783/500000=9.357%, speed: 4384.724it/s
# block_prob: 93449/1000000=9.345%, speed: 4443.312it/s
