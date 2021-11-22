import os
import argparse
from stable_baselines import GAIL, logger
from stable_baselines.gail import ExpertDataset
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack


def run_gail():
    parser = argparse.ArgumentParser()
    parser.add_argument('expert', type=str, default=None,
                        help='Expert path (*.npz)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--note', type=str, default='test')
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--num-steps', type=int, default=1000000)
    parser.add_argument('--policy', type=str, default='CnnPolicy',
                        choices=['CnnPolicy', 'CnnLstmPolicy', 'CnnLnLstmPolicy',
                                 'MlpPolicy', 'MlpLstmPolicy', 'MlpLnLstmPolicy'],
                        help='Policy architecture')
    args = parser.parse_args()

    logger.configure(os.path.join('logs', args.env, args.note))
    logger.info(args)

    if 'NoFrameskip' in args.env:
        env = VecFrameStack(make_atari_env(args.env, 1, args.seed), 4)
    else:
        import gym
        env = gym.make(args.env)

    dataset = ExpertDataset(
        expert_path=args.expert, batch_size=128, train_fraction=0.99, verbose=1)
    model = GAIL(args.policy, env, dataset, timesteps_per_batch=1280, verbose=1)
    model.learn(len(dataset.train_loader) * 1280)


if __name__ == '__main__':
    run_gail()
