import random

import numpy as np
import torch

class RealEnvBuffer:
    def __init__(self, capacity):
        self.memory = []
        self.capacity = capacity
        self.change_idx = 0
    
    def __len__(self):
        return len(self.memory)

    def push(self, state, action, next_state, reward, is_done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.change_idx] = [state, action, next_state, reward, is_done]
        self.change_idx = (self.change_idx + 1) % self.capacity
    
    def sample(self, size):
        '''
        states, actions, next_states, rewards, is_dones
        '''
        samples = random.sample(self.memory, k=size)
        states, actions, next_states, rewards, is_dones = map(np.stack, zip(*samples))
        return states, actions, next_states, rewards, is_dones


class RolloutBuffer:
    '''
    Buffer for training PPO policy.
    This buffer is emptied everytime the policy is updated.
    '''
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
