import torch

from buffer import RealEnvBuffer, RolloutBuffer
from models import DeterministicModel
from ppo import PPO
from argument_parser import argument_parse
from utils import set_global_seed
from train import collect_observations_from_real_env, train_deterministic_model, train_ppo_policy
from env import make_vec_stack_atari_env


def main():
    args = argument_parse()

    env = make_vec_stack_atari_env(args.env_name, args.n_envs)
    state_dim = env.observation_space.shape[2]
    action_dim = env.action_space.n

    set_global_seed(args.seed, env)

    device = torch.device('cuda' if args.cuda else 'cpu')

    real_env_buffer = RealEnvBuffer(args.real_env_buffer_size)

    ppo_agent = PPO(state_dim, action_dim, args.lr_actor, args.lr_critic, args.gamma, args.K_epochs,
                    args.eps_clip, device)
    deterministic_model = DeterministicModel(state_dim, action_dim).to(device)

    for ith_main_loop in range(1, args.main_loop_iter + 1):

        collect_observations_from_real_env(env, ppo_agent, args.n_envs, real_env_buffer, device)

        # update model using collected data
        # TODO arbitrary values for hyperparameters
        world_model_batch_size = 64
        world_model_lr = 3e-4
        train_deterministic_model(deterministic_model, world_model_lr, real_env_buffer,
                                  world_model_batch_size, ith_main_loop, device)

        # update policy using world model
        rollout_buffer = RolloutBuffer()
        train_ppo_policy(ppo_agent, args.parallel_agents_num, deterministic_model, rollout_buffer,
                         real_env_buffer, device, args.ppo_epoch, args.rollout_step_num)


if __name__ == "__main__":
    # for ease of debug
    with torch.autograd.set_detect_anomaly(True):
        main()