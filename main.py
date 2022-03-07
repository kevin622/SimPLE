import torch
import wandb

from buffer import RealEnvBuffer, RolloutBuffer
from models import DeterministicModel
from ppo import PPO
from argument_parser import argument_parse
from utils import set_global_seed
from train import collect_observations_from_real_env, train_deterministic_model, train_ppo_policy, eval_policy
from env import make_vec_stack_atari_env


def main():
    args = argument_parse()

    if args.wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_id, config={})
        wandb.config.update(args)

        wandb.define_metric('model_loss_step')
        wandb.define_metric('Deterministic Model Loss', step_metric='model_loss_step')

        wandb.define_metric('ppo_loss_step')
        wandb.define_metric('PPO loss', step_metric='ppo_loss_step')

        wandb.define_metric('ith_main_loop')
        wandb.define_metric(f'Average Reward for {args.eval_iter_num} Episodes',
                            step_metric='ith_main_loop')
        # wandb.config = {i: args.__getattribute__(i) for i in dir(args) if not i.startswith('_')}

    env = make_vec_stack_atari_env(args.env_name, args.n_envs)
    state_dim = env.observation_space.shape[2]
    action_dim = env.action_space.n
    eval_env = make_vec_stack_atari_env("BreakoutDeterministic-v0", 1,
                                        args.seed)  # Environment for evaluation

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
                                  world_model_batch_size, ith_main_loop, device, args.wandb)

        # update policy using world model
        rollout_buffer = RolloutBuffer()
        train_ppo_policy(ppo_agent, args.parallel_agents_num, deterministic_model, rollout_buffer,
                         real_env_buffer, device, args.ppo_epoch, args.rollout_step_num, args.wandb,
                         ith_main_loop)

        # Evaluatate the policy by making rollouts on eval_env
        eval_policy(ppo_agent, eval_env, device, args.eval_iter_num, args.wandb, ith_main_loop)


if __name__ == "__main__":
    # for ease of debug
    with torch.autograd.set_detect_anomaly(True):
        main()