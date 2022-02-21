import argparse
import random

import gym
from gym.wrappers import FrameStack, ResizeObservation
import torch
import numpy as np

from replay_buffer import ReplayBuffer
from models import WorldModel, Policy


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
    env = ResizeObservation(env, (105, 80))
    env = FrameStack(env, 4)

    env.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    policy = Policy()
    world_model = WorldModel(in_channel_size=env.observation_space[0] * 4,
                                          action_size=env.action_space.n)
    replay_buffer = ReplayBuffer(args.buffer_size)

    pass


if __name__ == "__main__":
    main()