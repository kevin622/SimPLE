import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from models import DeterministicModel
from buffer import RealEnvBuffer, RolloutBuffer
from utils import to_tensor
from ppo import PPO


def train_deterministic_model(model: DeterministicModel, lr: float, buffer: RealEnvBuffer,
                              batch_size: int, ith_main_loop: int, device: torch.device):
    optimizer = Adam(model.parameters(), lr)
    # TODO
    # iteration_num = 45000 if ith_main_loop == 1 else 15000
    iteration_num = 150
    for _ in tqdm(range(iteration_num)):
        # get samples from buffer
        states, actions, next_states, rewards, is_dones = buffer.sample(batch_size)
        states = to_tensor(states, device)
        actions = torch.tensor(actions).to(device)
        rewards = to_tensor(rewards, device).reshape(-1, 1)
        next_states = to_tensor(next_states, device)

        # get predictions from model
        predicted_states, predicted_rewards = model.get_output_frame_and_reward(
            states, actions, batch_size, device)
        loss_image = F.mse_loss(next_states, predicted_states)
        loss_reward = F.mse_loss(rewards, predicted_rewards)
        loss = loss_image + loss_reward

        # update the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def train_ppo_policy(ppo_agent: PPO,
                     deterministic_model: DeterministicModel,
                     rollout_buffer: RolloutBuffer,
                     real_env_buffer: RealEnvBuffer,
                     device: torch.device,
                     ppo_epoch: int = 1000,
                     rollout_step_num: int = 50):

    for ith_epoch in range(1, ppo_epoch + 1):
        state = real_env_buffer.sample(1)[0]  # sample one state
        state = to_tensor(state, device)
        for ith_step in range(1, rollout_step_num + 1):
            action, action_logprob = ppo_agent.select_action(state)
            output_frame, reward = deterministic_model.get_output_frame_and_reward(
                state, action, 1, device)
            if ith_step == rollout_step_num:
                state_values = ppo_agent.policy.critic(
                    state.reshape([state.shape[0], state.shape[3], state.shape[1], state.shape[2]]))
                reward += state_values
            reshaped_state = state[0].reshape(state[0].shape[2], state[0].shape[0], state[0].shape[1])
            rollout_buffer.push(reshaped_state, action.item(), action_logprob.item(), reward.item(),
                            False)
            # next_state
            state = torch.cat((state[:, :, :, 3:], output_frame), dim=-1)
        print('updated')
        ppo_agent.update(rollout_buffer)
