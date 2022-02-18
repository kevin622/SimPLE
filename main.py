import argparse
import random

import gym
import torch
import numpy as np

from replay_buffer import ReplayBuffer
from models import DeterministicWorldModel, Policy


def main():
    parser = argparse.ArgumentParser(description="SimPLE Args")
    parser.add_argument("--env_name",
                        default="Breakout-v0",
                        help="Name of the environment(default: Breakout-v0)")
    parser.add_argument("--buffer_size",
                        default=1000000,
                        type=int,
                        help="Size of replay buffer(default: 1,000,000)")
    parser.add_argument("--seed", default=123456, type=int, help="Random Seed(default: 123456)")
    args = parser.parse_args()

    env = gym.make(args.env_name)
    env.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    policy = Policy()
    world_model = DeterministicWorldModel(pixel_shape=env.observation_space[0],
                                          action_shape=env.action_space.n)
    replay_buffer = ReplayBuffer(args.buffer_size)

    pass


if __name__ == "__main__":
    main()