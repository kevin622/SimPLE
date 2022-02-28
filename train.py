import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from models import DeterministicModel
from buffer import RealEnvBuffer
from utils import to_tensor


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
