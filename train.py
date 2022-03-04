import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from models import DeterministicModel
from buffer import RealEnvBuffer, RolloutBuffer
from utils import to_tensor
from ppo import PPO

def collect_observations_from_real_env(env, ppo_agent: PPO, n_envs: int, real_env_buffer: RealEnvBuffer, device: torch.device):
    print('----------------------------------------------------------------')
    print('Collecting Observations from the Real Environment')
    state = env.reset()
    for ith_step in tqdm(range(6400 // n_envs)):
        action, action_logprob = ppo_agent.select_action(to_tensor(state, device))
        action = action.cpu().numpy()
        next_state, reward, is_done, info = env.step(action)
        for i in range(n_envs):
            real_env_buffer.push(state[i], action[i], next_state[i, :, :, :3], reward[i],
                                    is_done[i])
        state = next_state
    print('Collcecting Done!')

def train_deterministic_model(model: DeterministicModel, lr: float, buffer: RealEnvBuffer,
                              batch_size: int, ith_main_loop: int, device: torch.device):
    print('----------------------------------------------------------------')
    print('Training the Deterministic Model with the RealEnvBuffer')
    optimizer = Adam(model.parameters(), lr)
    iteration_num = 45000 if ith_main_loop == 1 else 15000
    for _ in tqdm(range(iteration_num // batch_size)):
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
    print('Model Trained!')

def train_ppo_policy(ppo_agent: PPO,
                     parallel_agents_num: int,
                     deterministic_model: DeterministicModel,
                     rollout_buffer: RolloutBuffer,
                     real_env_buffer: RealEnvBuffer,
                     device: torch.device,
                     ppo_epoch: int = 1000,
                     rollout_step_num: int = 50):

    print('----------------------------------------------------------------')
    print('Training the policy with PPO algo')
    for ith_epoch in tqdm(range(1, ppo_epoch + 1)):
        state = real_env_buffer.sample(parallel_agents_num)[0]
        state = to_tensor(state, device)
        for ith_step in range(1, rollout_step_num + 1):
            action, action_logprob = ppo_agent.select_action(state)
            output_frame, reward = deterministic_model.get_output_frame_and_reward(
                state, action, parallel_agents_num, device)
            reshaped_state = state.reshape(
                [state.shape[0], state.shape[3], state.shape[1], state.shape[2]])
            rollout_buffer.push(reshaped_state, action, action_logprob, reward, False)
            state = torch.cat((state[:, :, :, 3:], output_frame), dim=-1)
        ppo_agent.update(rollout_buffer)
    print('Policy Trained!')