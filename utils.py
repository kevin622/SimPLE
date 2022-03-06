import random

import torch
import numpy as np
import gym
from gym.wrappers import FrameStack, ResizeObservation


def set_global_seed(seed, env):
    env.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_resized_stacked_env(env_name: str):
    env = gym.make(env_name)
    env = ResizeObservation(env, (105, 80))
    env = FrameStack(env, 4)
    return env

def to_numpy(tensor: torch.Tensor):
    return tensor.cpu().numpy()

def to_tensor(array: np.ndarray, device: torch.device):
    return torch.FloatTensor(array).to(device)