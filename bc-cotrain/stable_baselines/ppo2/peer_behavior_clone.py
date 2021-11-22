import os
import argparse
from stable_baselines import PPO2, logger
from stable_baselines.gail import ExpertDataset
from stable_baselines.common import set_global_seeds
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack

# Using only one expert trajectory
# you can specify `traj_limitation=-1` for using the whole dataset

parser = argparse.ArgumentParser()
parser.add_argument('expert', type=str, default=None,
                    help='Expert model path.')
parser.add_argument('--env', type=str, default='PongNoFrameskip-v4',
                    help='environment ID')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed')
parser.add_argument('--policy', type=str, default='CnnPolicy',
                    choices=['CnnPolicy', 'CnnLstmPolicy', 'CnnLnLstmPolicy',
                             'MlpPolicy', 'MlpLstmPolicy', 'MlpLnLstmPolicy'],
                    help='Policy architecture')
parser.add_argument('--peer', type=float, default=0.,
                    help='Coefficient of the peer term. (default: 0)')
parser.add_argument('--note', type=str, default='test',
                    help='Log path')
parser.add_argument('--val-interval', type=int, default=100)
parser.add_argument('--val-episodes', type=int, default=1)
parser.add_argument('--num-epochs', type=int, default=50)
args = parser.parse_args()

set_global_seeds(args.seed)

logger.configure(os.path.join('logs', args.env, args.note))

dataset = ExpertDataset(
    expert_path=args.expert, batch_size=128, train_fraction=0.99, verbose=1)

if 'NoFrameskip' in args.env:
    env = VecFrameStack(make_atari_env(args.env, 1, args.seed), 4)
else:
    import gym
    env = gym.make(args.env)

model = PPO2(args.policy, env, verbose=1)

# Pretrain the PPO2 model
# Data should be abundant, so train only one epoch
model.pretrain(
    dataset, peer=args.peer, val_interval=args.val_interval,
    val_episodes=args.val_episodes, n_epochs=args.num_epochs)

# As an option, you can train the RL agent
# model.learn(int(1e5))

# Test the pre-trained model
env = model.get_env()
obs = env.reset()
