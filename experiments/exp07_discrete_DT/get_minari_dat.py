import gymnasium as gym
import os

os.environ["D3RLPY_DATASETS_PATH"] = "/gpfs/data/fs72297/jklotz/programming_data/d3rlpy_data"
os.makedirs(os.environ["D3RLPY_DATASETS_PATH"], exist_ok=True)
os.environ["MINARI_DATASETS_PATH"] = "/gpfs/data/fs72297/jklotz/programming_data/d3rlpy_data/minari_data"
os.makedirs(os.environ["MINARI_DATASETS_PATH"], exist_ok=True)

import d3rlpy

dataset = d3rlpy.datasets.get_minari('atari/breakout/expert-v0')
dataset = d3rlpy.datasets.get_minari('mujoco/halfcheetah/expert-v0')