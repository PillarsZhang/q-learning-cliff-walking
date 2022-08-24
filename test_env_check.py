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

print("_rand_large")
for i in tqdm(range(10), disable=True):
    env = Env.get_advanced(is_rand=True, is_large=True)
    agent = Agent(env)
    agent.draw_stdout()
    print(agent.env.check())

# Performance test and Monte Carlo block probability test

saved_path = Path("saved/test_env_check")
saved_path.mkdir(exist_ok=True, parents=True)

## default
num_test = int(2e5)
num_block = 0
num_unblock = 0
env_check_sum = 0
tic = time.time()
for i in tqdm(range(num_test)):
    env = Env.get_advanced(is_no_block=False)
    if env.check() < 0:
        num_block += 1
    else:
        env_check_sum += env.check()
        num_unblock += 1
toc = time.time() - tic

info = (
    f"cpu: {cpuinfo.get_cpu_info()['brand_raw']},\n"
    f"mean: {env_check_sum/num_unblock:.3f},\n"
    f"block_prob: {num_block}/{num_test}={num_block/num_test*100:.3f}%,\n"
    f"speed: {num_test/toc:.3f}it/s\n"
)

print("default")
print(info)

with open(saved_path / "info.txt", "w") as f:
    f.write(info)

## rand_large
num_test = int(2e5)
num_block = 0
num_unblock = 0
env_check_sum = 0
tic = time.time()
for i in tqdm(range(num_test)):
    env = Env.get_advanced(is_no_block=False, is_rand=True, is_large=True)
    if env.check() < 0:
        num_block += 1
    else:
        env_check_sum += env.check()
        num_unblock += 1
toc = time.time() - tic

info = (
    f"cpu: {cpuinfo.get_cpu_info()['brand_raw']},\n"
    f"mean: {env_check_sum/num_unblock:.3f},\n"
    f"block_prob: {num_block}/{num_test}={num_block/num_test*100:.3f}%,\n"
    f"speed: {num_test/toc:.3f}it/s\n"
)

print("_rand_large")
print(info)

with open(saved_path / "info_rand_large.txt", "w") as f:
    f.write(info)
