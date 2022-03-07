import torch
import torch.nn.functional as F
from torch.optim import Adam
from gym.wrappers.time_limit import TimeLimit
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from tqdm import tqdm
import wandb

from models import DeterministicModel
from buffer import RealEnvBuffer, RolloutBuffer
from utils import to_tensor, to_numpy
from ppo import PPO


def collect_observations_from_real_env(env, ppo_agent: PPO, n_envs: int,
                                       real_env_buffer: RealEnvBuffer, device: torch.device):
    print('----------------------------------------------------------------')
    print('Collecting Observations from the Real Environment')
    state = env.reset()
    for ith_step in tqdm(range(6400 // n_envs)):
        action, action_logprob = ppo_agent.select_action(to_tensor(state, device))
        action = to_numpy(action)
        next_state, reward, is_done, info = env.step(action)
        for i in range(n_envs):
            real_env_buffer.push(state[i], action[i], next_state[i, :, :, :3], reward[i],
                                 is_done[i])
        state = next_state
    print('Collcecting Done!')


def train_deterministic_model(model: DeterministicModel, lr: float, buffer: RealEnvBuffer,
                              batch_size: int, ith_main_loop: int, device: torch.device,
                              use_wandb: bool):
    print('----------------------------------------------------------------')
    print('Training the Deterministic Model with the RealEnvBuffer')
    optimizer = Adam(model.parameters(), lr)
    iteration_num = 45000 if ith_main_loop == 1 else 15000
    # for _ in tqdm(range(iteration_num // batch_size)):
    for iter_num in tqdm(range(iteration_num)):
        # get samples from buffer
        states, actions, next_states, rewards, is_dones = buffer.sample(batch_size)
        states = to_tensor(states, device)
        actions = torch.tensor(actions).to(device)
        rewards = to_tensor(rewards, device).reshape(-1, 1)
        next_states = to_tensor(next_states, device)

        # get predictions from model
        predicted_states, predicted_rewards = model.get_output_frame_and_reward(
            states, actions, batch_size, device)
        # TODO Try weighting these losses -> hyperparameters
        # TODO wandb x axis setting
        loss_image = F.mse_loss(next_states, predicted_states)
        loss_reward = F.mse_loss(rewards, predicted_rewards)
        loss = loss_image + loss_reward

        # update the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if use_wandb:
            model_loss_step = iter_num + (ith_main_loop - 1) * iteration_num
            if ith_main_loop > 1:
                model_loss_step += 45000 - 15000
            wandb.log({'Deterministic Model Loss': loss, 'model_loss_step': model_loss_step})
    print('Model Trained!')


def train_ppo_policy(ppo_agent: PPO,
                     parallel_agents_num: int,
                     deterministic_model: DeterministicModel,
                     rollout_buffer: RolloutBuffer,
                     real_env_buffer: RealEnvBuffer,
                     device: torch.device,
                     ppo_epoch: int = 1000,
                     rollout_step_num: int = 50,
                     use_wandb: bool = False,
                     ith_main_loop: int = 1):

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
        ppo_loss = ppo_agent.update(rollout_buffer)
        if use_wandb:
            ppo_loss_step = (ith_main_loop - 1) * ppo_epoch + (ith_epoch - 1)
            wandb.log({
                'PPO loss': ppo_loss,
                'ppo_loss_step': ppo_loss_step
            })
    print('Policy Trained!')


def eval_policy(ppo_agent: PPO,
                env: VecFrameStack,
                device: torch.device,
                iter_nums: int = 10,
                use_wandb: bool = False,
                ith_main_loop: int = 1):

    total_sum_reward = 0
    for iter_num in range(1, iter_nums + 1):
        state = env.reset()
        done = [False]
        sum_of_reward = 0
        while not done[0]:
            action, _ = ppo_agent.select_action(to_tensor(state, device))
            next_state, reward, done, _ = env.step(to_numpy(action))
            breakpoint()
            state = next_state
            sum_of_reward += reward[0]
            if done[0]:
                total_sum_reward += sum_of_reward
                print(f'Episode {iter_num} Sum of Reward: {sum_of_reward}')
    avg_of_reward = total_sum_reward / iter_nums

    print(f'Evaluated for {iter_nums} Episodes, Average Reward: {round(avg_of_reward, 2)}')
    if use_wandb:
        wandb.log({
            f'Average Reward for {iter_nums} Episodes': avg_of_reward,
            'ith_main_loop': ith_main_loop
        })
