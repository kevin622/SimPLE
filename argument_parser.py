import argparse


# args.lr_actor, args.lr_critic, args.gamma, args.K_epochs, args.eps_clip
def argument_parse():
    parser = argparse.ArgumentParser(description="SimPLE Args")
    parser.add_argument("--env_name",
                        default="BreakoutDeterministic-v0",
                        help="Name of the environment(default: BreakoutDeterministic-v0)")
    parser.add_argument("--n_envs", default=4, help="Number of parallel environments(default: 4)")
    parser.add_argument("--real_env_buffer_size",
                        default=1000000,
                        type=int,
                        help="Size of real env replay buffer(default: 1,000,000)")
    parser.add_argument("--main_loop_iter",
                        default=15,
                        type=int,
                        help="The iteration count of the main loop(default: 15)")
    parser.add_argument("--seed", default=123456, type=int, help="Random Seed(default: 123456)")
    parser.add_argument("--K_epochs",
                        default=80,
                        type=int,
                        help="update policy for K epochs in one PPO(default: 80)")
    parser.add_argument("--eps_clip",
                        default=0.2,
                        type=float,
                        help="clip parameter for PPO(default: 0.2)")
    parser.add_argument("--gamma",
                        default=0.99,
                        type=float,
                        help="PPO discount factor(default: 0.99)")
    parser.add_argument("--lr_actor",
                        default=0.0003,
                        type=float,
                        help="learning rate for PPO actor network(default: 0.0003)")
    parser.add_argument("--lr_critic",
                        default=0.001,
                        type=float,
                        help="learning rate for PPO critic network(default: 0.001)")
    parser.add_argument("--ppo_epoch",
                        default=1000,
                        type=int,
                        help="Epoch for PPO update in one main loop iteration(default: 1000)")
    parser.add_argument("--rollout_step_num",
                        default=50,
                        type=int,
                        help="Length of rollouts in world model(default: 50)")
    parser.add_argument("--parallel_agents_num",
                        default=16,
                        type=int,
                        help="Parallel agents for rollouts in world model(default: 16)")
    parser.add_argument("--cuda", action='store_true', help="whether to use cuda(default: False)")

    args = parser.parse_args()
    return args
