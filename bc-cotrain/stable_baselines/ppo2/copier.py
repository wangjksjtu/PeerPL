import os
import copy
import numpy as np
import tensorflow as tf

from stable_baselines import PPO2, logger
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines.common.cmd_util import (
    make_atari_env, make_vec_env, atari_arg_parser)
from stable_baselines.common.policies import (
    CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy, MlpPolicy)


class Scheduler:
    def __init__(self, start_step, end_step, decay_type=None):
        self._total_steps = end_step - start_step
        self._start_step = start_step
        self._end_step = end_step
        self._decay_type = decay_type

    def __call__(self, step):
        if step <= self._start_step or step >= self._end_step:
            return 0
        step -= self._start_step
        if self._decay_type == 'incdec':
            return 1 - np.abs(self._total_steps / 2 - step) / (self._total_steps / 2)
        elif self._decay_type == 'inc':
            return step / self._total_steps
        elif self._decay_type == 'dec':
            return 1 - step / self._total_steps
        elif not self._decay_type:
            return 1.
        else:
            raise NotImplementedError


class View:
    def __init__(self, model, peer=0., learning_rate=2.5e-4, epsilon=1e-5):
        self.model = model
        self.peer = peer
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        with self.model.graph.as_default():
            with tf.variable_scope('copier'):
                self.peer_ph = tf.placeholder(tf.float32, (), "peer_ph")
                self.obs_ph, self.actions_ph, self.actions_logits_ph = \
                    self.model._get_pretrain_placeholders()
                actions_ph = tf.expand_dims(self.actions_ph, axis=1)
                one_hot_actions = tf.one_hot(actions_ph, self.model.action_space.n)
                self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=self.actions_logits_ph,
                    labels=tf.stop_gradient(one_hot_actions))
                self.loss = tf.reduce_mean(self.loss)
                # calculate peer term
                self.peer_actions_ph = tf.placeholder(
                    actions_ph.dtype, actions_ph.shape, "peer_action_ph")
                peer_onehot_actions = tf.one_hot(
                    self.peer_actions_ph, self.model.action_space.n)
                # Use clipped softmax instead of the default
                # peer_term = tf.nn.softmax_cross_entropy_with_logits_v2(
                #     logits=self.actions_logits_ph,
                #     labels=tf.stop_gradient(peer_onehot_actions))
                softmax_actions_logits_ph = tf.nn.softmax(
                    self.actions_logits_ph, axis=1) + 1e-8
                self.peer_term = self.peer_ph * tf.reduce_mean(-tf.reduce_sum(
                    tf.stop_gradient(peer_onehot_actions) *
                    tf.log(softmax_actions_logits_ph), axis=-1))
                self.loss -= self.peer_term

            self.optim_op = self.model.trainer.minimize(
                self.loss, var_list=self.model.params)

    def learn(self, obses, actions, learning_rate=None):
        peer_actions = copy.deepcopy(actions)
        np.random.shuffle(peer_actions)
        feed_dict = {
            self.obs_ph: obses,
            self.actions_ph: actions[: None],
            self.peer_actions_ph: peer_actions[:, None],
            self.model.learning_rate_ph: learning_rate or self.learning_rate,
            self.peer_ph: self.peer,
        }
        train_loss, peer_loss, _ = self.model.sess.run(
            [self.loss, self.peer_term, self.optim_op], feed_dict)
        logger.logkv("copier loss", train_loss)
        logger.logkv("peer term", peer_loss)
        logger.dumpkvs()


def train(env_id, num_timesteps, seed, policy, n_envs=8, nminibatches=4,
          n_steps=128, peer=0., scheduler=None, individual=False, repeat=1):
    """
    Train PPO2 model for atari environment, for testing purposes

    :param env_id: (str) the environment id string
    :param num_timesteps: (int) the number of timesteps to run
    :param seed: (int) Used to seed the random generator.
    :param policy: (Object) The policy model to use (MLP, CNN, LSTM, ...)
    :param n_envs: (int) Number of parallel environments
    :param nminibatches: (int) Number of training minibatches per update.
        For recurrent policies, the number of environments run in parallel
        should be a multiple of nminibatches.
    :param n_steps: (int) The number of steps to run for each environment
        per update (i.e. batch size is n_steps * n_env where n_env is
        number of environment copies running in parallel)
    """

    policy = {
        'cnn': CnnPolicy,
        'lstm': CnnLstmPolicy,
        'lnlstm': CnnLnLstmPolicy,
        'mlp': MlpPolicy
    }[policy]

    is_atari = 'NoFrameskip' in env_id
    make_env = lambda: VecFrameStack(make_atari_env(env_id, n_envs, seed), 4) if is_atari \
        else make_vec_env(env_id, n_envs, seed)
    print(make_env)

    models = {
        "A": PPO2(
            policy=policy, policy_kwargs={'view': 'even'}, n_steps=n_steps,
            env=make_env(), nminibatches=nminibatches, lam=0.95, gamma=0.99, 
            noptepochs=4, ent_coef=.01, learning_rate=2.5e-4,
            cliprange=lambda f: f * 0.1, verbose=1),
        "B": PPO2(
            policy=policy, policy_kwargs={'view': 'odd'}, n_steps=n_steps,
            env=make_env(), nminibatches=nminibatches, lam=0.95, gamma=0.99, 
            noptepochs=4, ent_coef=.01, learning_rate=2.5e-4,
            cliprange=lambda f: f * 0.1, verbose=1)}

    views = {view: View(models[view], peer=peer) for view in ("A", "B")}

    n_batch = n_envs * n_steps
    n_updates = num_timesteps // n_batch

    for t in range(n_updates):
        logger.info("current episode:", t)
        for view in "A", "B":
            models[view].learn(n_batch)
        if not individual:
            for view, other_view in zip(("A", "B"), ("B", "A")):
                obses, _, _, actions, _, _, _, _, _ = models[other_view].rollout
                views[view].peer = peer * scheduler(t)
                logger.info("current alpha:", views[view].peer)
                for _ in range(repeat):
                    views[view].learn(
                        obses, actions, views[view].learning_rate / repeat)

    for view in "A", "B":
        models[view].env.close()
        del models[view]  # free memory


def main():
    """
    Runs the test
    """
    parser = atari_arg_parser()
    parser.add_argument('--policy', choices=['cnn', 'lstm', 'lnlstm', 'mlp'],
                        default='cnn', help='Policy architecture')
    parser.add_argument('--peer', type=float, default=0.,
                        help='Coefficient of the peer term. (default: 0)')
    parser.add_argument('--note', type=str, default='test',
                        help='Log path')
    parser.add_argument('--individual', action='store_true', default=False,
                        help='If true, no co-training is applied.')
    parser.add_argument('--start-episode', type=int, default=0,
                        help='Add peer term after this episode.')
    parser.add_argument('--end-episode', type=int, default=10000,
                        help='Remove peer term after this episode.')
    parser.add_argument('--decay-type', type=str, default=None, 
                        choices=[None, 'inc', 'dec', 'inc_dec'],
                        help='Decay type for alpha')
    parser.add_argument('--repeat', type=int, default=1,
                        help='Repeat training on the dataset in one epoch')
    args = parser.parse_args()

    set_global_seeds(args.seed)

    logger.configure(os.path.join('logs', args.env, args.note))
    logger.info(args)
    scheduler = Scheduler(args.start_episode, args.end_episode, decay_type=args.decay_type)
    train(
        args.env,
        num_timesteps=args.num_timesteps,
        seed=args.seed,
        policy=args.policy,
        peer=args.peer,
        scheduler=scheduler,
        individual=args.individual,
        repeat=args.repeat,
    )


if __name__ == '__main__':
    main()
