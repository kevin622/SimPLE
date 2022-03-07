# Policy
# PPO is an on-policy algorithm.
# PPO can be used for environments with either discrete or continuous action spaces. => only discrete in ALE
# https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import wandb

from buffer import RolloutBuffer


class ActorCritic(nn.Module):

    def __init__(self, state_dim, action_dim):
        '''
        The SimPLE used PPO hyperparameters from openai baselines
        The PPO paper used policy hyperparameters from Mni+16 arXiv:1602.01783(Asynchronous Methods for Deep Reinforcement Learning)
        Mnih used newtork architecture of below.
        Used convolutional layer with 16 filters of size 8 x 8 with stride 4, 
        followed by a convolutional layer with with 32 filters of size 4 x 4 with stride 2, followed by a fully
        connected layer with 256 hidden units. All three hidden layers were followed by a rectifier nonlinearity(relu)
        '''
        super(ActorCritic, self).__init__()
        ########################## Applied CNN ##########################
        # actor
        self.actor = nn.Sequential(nn.Conv2d(state_dim, 16, 8, 4), nn.ReLU(),
                                   nn.Conv2d(16, 32, 4, 2), nn.ReLU(), nn.Flatten(1),
                                   nn.Linear(2816, 256), nn.ReLU(), nn.Linear(256, action_dim),
                                   nn.Softmax(dim=-1))
        # critic
        self.critic = nn.Sequential(nn.Conv2d(state_dim, 16, 8, 4), nn.ReLU(),
                                    nn.Conv2d(16, 32, 4, 2), nn.ReLU(), nn.Flatten(1),
                                    nn.Linear(2816, 256), nn.ReLU(), nn.Linear(256, 1))
        #################################################################

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy


class PPO:

    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 device):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        # self.buffer = buffer

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam([{
            'params': self.policy.actor.parameters(),
            'lr': lr_actor
        }, {
            'params': self.policy.critic.parameters(),
            'lr': lr_critic
        }])

        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())  # hard copy

        self.MseLoss = nn.MSELoss()

    def select_action(self, state: torch.Tensor):
        with torch.no_grad():
            state = state.reshape(
                (state.shape[0], state.shape[3], state.shape[1], state.shape[2])).to(self.device)
            action, action_logprob = self.policy_old.act(state)

        # self.buffer.states.append(state)
        # self.buffer.actions.append(action)
        # self.buffer.logprobs.append(action_logprob)

        # return action.item()
        return action.detach(), action_logprob.detach()

    def update(self, buffer: RolloutBuffer):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(buffer.rewards), reversed(buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        # rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = torch.stack(rewards).to(self.device)
        # rewards = (rewards - rewards.mean(dim=0)) / (rewards.std(dim=0) + 1e-7)
        # rewards = rewards.reshape(rewards.shape[0] * rewards.shape[1])

        # convert list to tensor
        # old_states = torch.squeeze(torch.stack(buffer.states, dim=0)).detach().to(self.device)
        # old_actions = torch.squeeze(torch.stack(buffer.actions,
        #                                         dim=0)).detach().to(self.device)
        # old_logprobs = torch.squeeze(torch.stack(buffer.logprobs,
        #                                          dim=0)).detach().to(self.device)
        old_states = torch.stack(buffer.states).detach().to(self.device)
        old_states = old_states.reshape([
            old_states.shape[0] * old_states.shape[1], old_states.shape[2], old_states.shape[3],
            old_states.shape[4]
        ])

        old_actions = torch.stack(buffer.actions).detach().to(self.device)
        old_actions = old_actions.flatten()
        # old_actions = old_actions.reshape(old_actions.shape[0] * old_actions.shape[1])
        old_logprobs = torch.stack(buffer.logprobs).detach().to(self.device)
        old_logprobs = old_logprobs.flatten()
        # old_logprobs = old_logprobs.reshape(old_logprobs.shape[0] * old_logprobs.shape[1])

        sum_of_losses = 0
        # Optimize policy for K epochs
        for kth_epoch in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            # Add value evaluation to the last rollout

            new_rewards = rewards.clone()
            new_rewards[-1] += state_values[-16:].detach()
            new_rewards = (new_rewards - new_rewards.mean(dim=0)) / (new_rewards.std(dim=0) + 1e-7)
            new_rewards = new_rewards.flatten()

            # match state_values tensor dimensions with rewards tensor
            # rewards = (rewards - rewards.mean(dim=0)) / (rewards.std(dim=0) + 1e-7)
            # rewards = rewards.reshape(rewards.shape[0] * rewards.shape[1])

            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = new_rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = torch.mean(-torch.min(surr1, surr2) +
                              0.5 * self.MseLoss(state_values, new_rewards) - 0.01 * dist_entropy)

            # take gradient step
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            sum_of_losses += loss.item()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        buffer.clear()
        mean_of_losses = sum_of_losses / self.K_epochs
        return mean_of_losses

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
