import torch
import numpy as np
from tqdm import tqdm

from buffer import RealEnvBuffer, RolloutBuffer
from models import DeterministicModel
from ppo import PPO
from argument_parser import argument_parse
from utils import set_global_seed, get_resized_stacked_env, to_numpy, to_tensor
from train import train_deterministic_model
from env import make_vec_stack_atari_env


def main():
    args = argument_parse()

    
    env = make_vec_stack_atari_env(args.env_name, args.n_envs)
    state_dim = env.observation_space.shape[2]
    action_dim = env.action_space.n
    # env = get_resized_stacked_env(args.env_name)
    
    set_global_seed(args.seed, env)

    device = torch.device('cuda' if args.cuda else 'cpu')

    real_env_buffer = RealEnvBuffer(args.real_env_buffer_size)
    rollout_buffer = RolloutBuffer()
    ppo_agent = PPO(rollout_buffer, state_dim, action_dim, args.lr_actor, args.lr_critic, args.gamma, args.K_epochs,
                    args.eps_clip, device)
    deterministic_model = DeterministicModel(state_dim, action_dim).to(device)

    for ith_main_loop in range(1, args.main_loop_iter + 1):
        # collect obersvations from real env
        state = env.reset()
        # TODO 200 should be 6400
        for ith_step in tqdm(range(200)):
            action, action_logprob = ppo_agent.select_action(state)
            action = action.cpu().numpy()
            next_state, reward, is_done, info = env.step(action)
            for i in range(args.n_envs):
                # TODO think about the stack order
                pass
            real_env_buffer.push(state, action, next_state[0], reward, is_done)
            state = next_state

        # update model using collected data
        # TODO arbitrary values for hyperparameters
        world_model_batch_size = 32
        world_model_lr = 0.003
        deterministic_model = train_deterministic_model(deterministic_model, world_model_lr,
                                                        real_env_buffer, world_model_batch_size,
                                                        ith_main_loop, device)

        # update policy using world model
        ppo_epoch = 1000
        rollout_step_num = 50
        for ith_epoch in range(1, ppo_epoch + 1):
            breakpoint()
            state = real_env_buffer.sample(1)[0]
            for ith_step in range(1, rollout_step_num + 1):
                action, action_logprob = ppo_agent.select_action(state)
                state = state.unsqueeze(0) # TODO because d_model gets batch_size too. TRY TO MAKE ANOTHER FUNCTION
                output_frame, reward = deterministic_model.get_output_frame_and_reward(state, action, 1, device)
                state = state.squeeze(0)
                state = torch.cat((state[1:], output_frame))
            ppo_agent.update()


if __name__ == "__main__":
    main()