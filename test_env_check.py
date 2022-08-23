from pathlib import Path
from common import Env, Agent
from tqdm import tqdm
import time
import cpuinfo

from utils import reset_random_seed

reset_random_seed()

print("is_no_block: True")
for i in tqdm(range(10), disable=True):
    env = Env.get_advanced(is_no_block=True)
    agent = Agent(env)
    agent.draw_stdout()
    print(agent.env.check())

print("is_no_block: False")
for i in tqdm(range(10), disable=True):
    env = Env.get_advanced(is_no_block=False)
    agent = Agent(env)
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

info = (
    f"cpu: {cpuinfo.get_cpu_info()['brand_raw']},\n"
    f"block_prob: {num_block}/{num_test}={num_block/num_test*100:.3f}%,\n"
    f"speed: {num_test/toc:.3f}it/s\n"
)

print(info)

saved_path = Path("saved/test_env_check")
saved_path.mkdir(exist_ok=True, parents=True)

with open(saved_path / "info.txt", "w") as f:
    f.write(info)
