# FROM: https://raw.githubusercontent.com/shaneshixiang/rllabplusplus/master/sandbox/rocky/tf/algos/ddpg.py
from rllab.algos.base import RLAlgorithm
from rllab.misc.overrides import overrides
from rllab.misc import special
from sandbox.rocky.tf.misc import tensor_utils
from rllab.sampler import parallel_sampler
from rllab.plotter import plotter
from rllab.misc import ext
import rllab.misc.logger as logger
#import pickle as pickle
import numpy as np
import time
import gc
import pyprind
import tensorflow as tf
from tensorflow.contrib.distributions import Bernoulli
from sandbox.rocky.tf.core.network import MLP

from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
#from sandbox.rocky.tf.core.parameterized import suppress_params_loading
from rllab.core.serializable import Serializable
from sampling_utils import SimpleReplayPool

class DDPG(RLAlgorithm):
    """
    Deep Deterministic Policy Gradient.
    """

    def __init__(
            self,
            env,
            policy,
            qf,
            es,
            batch_size=32,
            n_epochs=200,
            epoch_length=1000,
            min_pool_size=10000,
            replay_pool_size=1000000,
            replacement_prob=1.0,
            discount=0.99,
            max_path_length=250,
            qf_weight_decay=0.,
            qf_update_method='adam',
            qf_learning_rate=1e-3,
            policy_weight_decay=0,
            policy_update_method='adam',
            policy_learning_rate=1e-3,
            policy_updates_ratio=1.0,
            eval_samples=10000,
            soft_target=True,
            soft_target_tau=0.001,
            n_updates_per_sample=1,
            scale_reward=1.0,
            include_horizon_terminal_transitions=False,
            plot=False,
            pause_for_plot=False,
            **kwargs):
        """
        :param env: Environment
        :param policy: Policy
        :param qf: Q function
        :param es: Exploration strategy
        :param batch_size: Number of samples for each minibatch.
        :param n_epochs: Number of epochs. Policy will be evaluated after each epoch.
        :param epoch_length: How many timesteps for each epoch.
        :param min_pool_size: Minimum size of the pool to start training.
        :param replay_pool_size: Size of the experience replay pool.
        :param discount: Discount factor for the cumulative return.
        :param max_path_length: Discount factor for the cumulative return.
        :param qf_weight_decay: Weight decay factor for parameters of the Q function.
        :param qf_update_method: Online optimization method for training Q function.
        :param qf_learning_rate: Learning rate for training Q function.
        :param policy_weight_decay: Weight decay factor for parameters of the policy.
        :param policy_update_method: Online optimization method for training the policy.
        :param policy_learning_rate: Learning rate for training the policy.
        :param eval_samples: Number of samples (timesteps) for evaluating the policy.
        :param soft_target_tau: Interpolation parameter for doing the soft target update.
        :param n_updates_per_sample: Number of Q function and policy updates per new sample obtained
        :param scale_reward: The scaling factor applied to the rewards when training
        :param include_horizon_terminal_transitions: whether to include transitions with terminal=True because the
        horizon was reached. This might make the Q value back up less stable for certain tasks.
        :param plot: Whether to visualize the policy performance after each eval_interval.
        :param pause_for_plot: Whether to pause before continuing when plotting.
        :return:
        """
        self.env = env
        self.on_policy_env = Serializable.clone(env)
        self.policy = policy
        self.qf = qf
        self.es = es
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.epoch_length = epoch_length
        self.min_pool_size = min_pool_size
        self.replay_pool_size = replay_pool_size
        self.replacement_prob = replacement_prob
        self.discount = discount
        self.max_path_length = max_path_length
        self.qf_weight_decay = qf_weight_decay
        self.qf_update_method = \
            FirstOrderOptimizer(
                update_method=qf_update_method,
                learning_rate=qf_learning_rate,
            )
        self.qf_learning_rate = qf_learning_rate
        self.policy_weight_decay = policy_weight_decay
        self.policy_update_method = \
            FirstOrderOptimizer(
                update_method=policy_update_method,
                learning_rate=policy_learning_rate,
            )
        self.policy_learning_rate = policy_learning_rate
        self.policy_updates_ratio = policy_updates_ratio
        self.eval_samples = eval_samples
        self.train_step = tf.placeholder(tf.float32, shape=(), name="train_step")
        self.global_train_step = 0.0

        self.soft_target_tau = soft_target_tau
        self.n_updates_per_sample = n_updates_per_sample
        self.include_horizon_terminal_transitions = include_horizon_terminal_transitions
        self.plot = plot
        self.pause_for_plot = pause_for_plot

        self.qf_loss_averages = []
        self.policy_surr_averages = []
        self.q_averages = []
        self.y_averages = []
        self.paths = []
        self.es_path_returns = []
        self.paths_samples_cnt = 0
        self.random_dist = Bernoulli(None, [.5])
        self.sigma_type = kwargs.get('sigma_type', 'gated')

        self.scale_reward = scale_reward

        self.train_policy_itr = 0

        self.opt_info = None

    def start_worker(self):
        parallel_sampler.populate_task(self.env, self.policy)
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    @overrides
    def train(self):
        gc_dump_time = time.time()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # This seems like a rather sequential method
            pool = SimpleReplayPool(
                max_pool_size=self.replay_pool_size,
                observation_dim=self.env.observation_space.flat_dim,
                action_dim=self.env.action_space.flat_dim,
                replacement_prob=self.replacement_prob,
            )


            on_policy_pool = SimpleReplayPool(
                max_pool_size=self.replay_pool_size,
                observation_dim=self.on_policy_env.observation_space.flat_dim,
                action_dim=self.on_policy_env.action_space.flat_dim,
            )


            self.start_worker()

            self.init_opt()
            # This initializes the optimizer parameters
            sess.run(tf.global_variables_initializer())
            itr = 0
            path_length = 0
            path_return = 0
            terminal = False
            initial = False
            observation = self.env.reset()
            on_policy_terminal = False
            on_policy_initial = False
            on_policy_path_length = 0
            on_policy_path_return = 0
            on_policy_observation = self.on_policy_env.reset()


            #with tf.variable_scope("sample_policy"):
                #with suppress_params_loading():
                #sample_policy = pickle.loads(pickle.dumps(self.policy))
            with tf.variable_scope("sample_policy"):
                sample_policy = Serializable.clone(self.policy)

            for epoch in range(self.n_epochs):
                logger.push_prefix('epoch #%d | ' % epoch)
                logger.log("Training started")
                train_qf_itr, train_policy_itr = 0, 0
                for epoch_itr in pyprind.prog_bar(range(self.epoch_length)):
                    # Execute policy
                    if terminal:  # or path_length > self.max_path_length:
                        # Note that if the last time step ends an episode, the very
                        # last state and observation will be ignored and not added
                        # to the replay pool
                        observation = self.env.reset()
                        self.es.reset()
                        sample_policy.reset()
                        self.es_path_returns.append(path_return)
                        path_length = 0
                        path_return = 0
                        initial = True
                    else:
                        initial = False

                    if on_policy_terminal:  # or path_length > self.max_path_length:
                        # Note that if the last time step ends an episode, the very
                        # last state and observation will be ignored and not added
                        # to the replay pool
                        observation = self.on_policy_env.reset()
                        sample_policy.reset()
                        on_policy_path_length = 0
                        on_policy_path_return = 0
                        on_policy_initial = True
                    else:
                        on_policy_initial = False

                    action = self.es.get_action(itr, observation, policy=sample_policy)  # qf=qf)
                    on_policy_action = self.get_action_on_policy(self.on_policy_env, on_policy_observation, policy=sample_policy)

                    next_observation, reward, terminal, _ = self.env.step(action)
                    on_policy_next_observation, on_policy_reward, on_policy_terminal, _ = self.on_policy_env.step(on_policy_action)

                    path_length += 1
                    path_return += reward
                    on_policy_path_length += 1
                    on_policy_path_return += reward

                    if not terminal and path_length >= self.max_path_length:
                        terminal = True
                        # only include the terminal transition in this case if the flag was set
                        if self.include_horizon_terminal_transitions:
                            pool.add_sample(observation, action, reward * self.scale_reward, terminal, initial)
                    else:
                        pool.add_sample(observation, action, reward * self.scale_reward, terminal, initial)

                    if not on_policy_terminal and on_policy_path_length >= self.max_path_length:
                        on_policy_terminal = True
                        # only include the terminal transition in this case if the flag was set
                        if self.include_horizon_terminal_transitions:
                            on_policy_pool.add_sample(on_policy_observation, on_policy_action, on_policy_reward * self.scale_reward, on_policy_terminal, on_policy_initial)
                    else:
                        on_policy_pool.add_sample(on_policy_observation, on_policy_action, on_policy_reward * self.scale_reward,on_policy_terminal, on_policy_initial)


                    on_policy_observation = on_policy_next_observation
                    observation = next_observation

                    if pool.size >= self.min_pool_size:
                        self.global_train_step += 1
                        for update_itr in range(self.n_updates_per_sample):
                            # Train policy
                            batch = pool.random_batch(self.batch_size)
                            on_policy_batch = on_policy_pool.random_batch(self.batch_size)
                            itrs = self.do_training(itr, on_policy_batch, batch)
                            train_qf_itr += itrs[0]
                            train_policy_itr += itrs[1]
                        sample_policy.set_param_values(self.policy.get_param_values())

                    itr += 1
                    if time.time() - gc_dump_time > 100:
                        gc.collect()
                        gc_dump_time = time.time()

                logger.log("Training finished")
                logger.log("Trained qf %d steps, policy %d steps"%(train_qf_itr, train_policy_itr))
                if pool.size >= self.min_pool_size:
                    self.evaluate(epoch)
                    params = self.get_epoch_snapshot(epoch)
                    logger.save_itr_params(epoch, params)

                logger.dump_tabular(with_prefix=False)
                logger.pop_prefix()
                if self.plot:
                    self.update_plot()
                    if self.pause_for_plot:
                        input("Plotting evaluation run: Press Enter to "
                                  "continue...")
            self.env.terminate()
            self.policy.terminate()

    def get_action_on_policy(self, env_spec, observation, policy, **kwargs):
        action, _ = policy.get_action(observation)
        action_space = env_spec.action_space
        return np.clip(action, action_space.low, action_space.high)

    def init_opt(self):

        # First, create "target" policy and Q functions
        with tf.variable_scope("target_policy"):
            target_policy = Serializable.clone(self.policy)
        with tf.variable_scope("target_qf"):
            target_qf = Serializable.clone(self.qf)

        # y need to be computed first
        obs = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1,
        )

        # The yi values are computed separately as above and then passed to
        # the training functions below
        action = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1,
        )

        yvar = tensor_utils.new_tensor(
            'ys',
            ndim=1,
            dtype=tf.float32,
        )

        obs_offpolicy = self.env.observation_space.new_tensor_variable(
            'obs_offpolicy',
            extra_dims=1,
        )


        action_offpolicy = self.env.action_space.new_tensor_variable(
            'action_offpolicy',
            extra_dims=1,
        )


        yvar = tensor_utils.new_tensor(
            'ys',
            ndim=1,
            dtype=tf.float32,
        )


        yvar_offpolicy = tensor_utils.new_tensor(
            'ys_offpolicy',
            ndim=1,
            dtype=tf.float32,
        )

        qf_weight_decay_term = 0.5 * self.qf_weight_decay * \
                               sum([tf.reduce_sum(tf.square(param)) for param in
                                    self.qf.get_params(regularizable=True)])

        qval = self.qf.get_qval_sym(obs, action)
        qval_off = self.qf.get_qval_sym(obs_offpolicy, action_offpolicy)

        qf_loss = tf.reduce_mean(tf.square(yvar - qval))
        qf_loss_off = tf.reduce_mean(tf.square(yvar_offpolicy - qval_off))

        # TODO: penalize dramatic changes in gating_func
        # if PENALIZE_GATING_DISTRIBUTION_DIVERGENCE:


        policy_weight_decay_term = 0.5 * self.policy_weight_decay * \
                                   sum([tf.reduce_sum(tf.square(param))
                                        for param in self.policy.get_params(regularizable=True)])
        policy_qval = self.qf.get_qval_sym(
            obs, self.policy.get_action_sym(obs),
            deterministic=True
        )

        policy_qval_off = self.qf.get_qval_sym(
            obs_offpolicy, self.policy.get_action_sym(obs_offpolicy),
            deterministic=True
        )

        policy_surr = -tf.reduce_mean(policy_qval)
        policy_surr_off = -tf.reduce_mean(policy_qval_off)


        if self.sigma_type == 'unified-gated' or self.sigma_type == 'unified-gated-decaying':
            print("Using Gated Sigma!")

            input_to_gates= tf.concat([obs, obs_offpolicy], axis=1)

            assert input_to_gates.get_shape().as_list()[-1] == obs.get_shape().as_list()[-1] +  obs_offpolicy.get_shape().as_list()[-1]

            # TODO: right now this is a soft-gate, should make a hard-gate (options vs mixtures)
            gating_func = MLP(name="sigma_gate",
                              output_dim=1,
                              hidden_sizes=(64,64),
                              hidden_nonlinearity=tf.nn.relu,
                              output_nonlinearity=tf.nn.sigmoid,
                              input_var=input_to_gates,
                              input_shape=tuple(input_to_gates.get_shape().as_list()[1:])).output
        elif self.sigma_type == 'unified':
            # sample a bernoulli random variable
            print("Using Bernoulli sigma!")
            gating_func = tf.cast(self.random_dist.sample(qf_loss.get_shape()), tf.float32)
        elif self.sigma_type == 'unified-decaying':
            print("Using decaying sigma!")
            gating_func = tf.train.exponential_decay(1.0, self.train_step, 20, 0.96, staircase=True)
        else:
            raise Exception("sigma type not supported")

        qf_inputs_list = [yvar, obs, action, yvar_offpolicy, obs_offpolicy, action_offpolicy, self.train_step]
        qf_reg_loss = qf_loss*(1.0-gating_func) + qf_loss_off * (gating_func) + qf_weight_decay_term

        policy_input_list = [obs, obs_offpolicy, self.train_step]
        policy_reg_surr = policy_surr*(1.0 - gating_func) + policy_surr_off*(gating_func) + policy_weight_decay_term

        if self.sigma_type == 'unified-gated-decaying':
            print("Adding a decaying factor to gated sigma!")
            decaying_factor = tf.train.exponential_decay(.5, self.train_step, 20, 0.96, staircase=True)
            penalty = decaying_factor*tf.nn.l2_loss(gating_func)
            qf_reg_loss += penalty
            policy_reg_surr += penalty

        self.qf_update_method.update_opt(qf_reg_loss, target=self.qf, inputs=qf_inputs_list)

        self.policy_update_method.update_opt(policy_reg_surr, target=self.policy, inputs=policy_input_list)


        f_train_qf = tensor_utils.compile_function(
            inputs=qf_inputs_list,
            outputs=[qf_loss, qval, self.qf_update_method._train_op],
        )

        f_train_policy = tensor_utils.compile_function(
            inputs=policy_input_list,
            outputs=[policy_surr, self.policy_update_method._train_op],
        )

        self.opt_info = dict(
            f_train_qf=f_train_qf,
            f_train_policy=f_train_policy,
            target_qf=target_qf,
            target_policy=target_policy,
        )

    def do_training(self, itr, batch, offpolicy_batch):

        obs, actions, rewards, next_obs, terminals = ext.extract(
            batch,
            "observations", "actions", "rewards", "next_observations",
            "terminals"
        )

        obs_off, actions_off, rewards_off, next_obs_off, terminals_off = ext.extract(
            offpolicy_batch,
            "observations", "actions", "rewards", "next_observations",
            "terminals"
        )

        # compute the on-policy y values
        target_qf = self.opt_info["target_qf"]
        target_policy = self.opt_info["target_policy"]

        next_actions, _ = target_policy.get_actions(next_obs)
        next_qvals = target_qf.get_qval(next_obs, next_actions)

        ys = rewards + (1. - terminals) * self.discount * next_qvals.reshape(-1)

        next_actions_off, _ = target_policy.get_actions(next_obs_off)
        next_qvals_off = target_qf.get_qval(next_obs_off, next_actions_off)

        ys_off = rewards + (1. - terminals_off) * self.discount * next_qvals_off.reshape(-1)

        f_train_qf = self.opt_info["f_train_qf"]
        f_train_policy = self.opt_info["f_train_policy"]

        qf_loss, qval, _ = f_train_qf(ys, obs, actions, ys_off, obs_off, actions_off, self.global_train_step)

        target_qf.set_param_values(
            target_qf.get_param_values() * (1.0 - self.soft_target_tau) +
            self.qf.get_param_values() * self.soft_target_tau)
        self.qf_loss_averages.append(qf_loss)
        self.q_averages.append(qval)
        self.y_averages.append(ys) #TODO: also add ys_off

        self.train_policy_itr += self.policy_updates_ratio
        train_policy_itr = 0
        while self.train_policy_itr > 0:
            f_train_policy = self.opt_info["f_train_policy"]
            policy_surr, _ = f_train_policy(obs, obs_off, self.global_train_step)
            target_policy.set_param_values(
                target_policy.get_param_values() * (1.0 - self.soft_target_tau) +
                self.policy.get_param_values() * self.soft_target_tau)
            self.policy_surr_averages.append(policy_surr)
            self.train_policy_itr -= 1
            train_policy_itr += 1

        return 1, train_policy_itr # number of itrs qf, policy are trained

    def evaluate(self, epoch):
        paths = parallel_sampler.sample_paths(
            policy_params=self.policy.get_param_values(),
            max_samples=self.eval_samples,
            max_path_length=self.max_path_length,
        )

        average_discounted_return = np.mean(
            [special.discount_return(path["rewards"], self.discount) for path in paths]
        )

        returns = [sum(path["rewards"]) for path in paths]

        all_qs = np.concatenate(self.q_averages)
        all_ys = np.concatenate(self.y_averages)

        average_q_loss = np.mean(self.qf_loss_averages)
        average_policy_surr = np.mean(self.policy_surr_averages)
        average_action = np.mean(np.square(np.concatenate(
            [path["actions"] for path in paths]
        )))

        policy_reg_param_norm = np.linalg.norm(
            self.policy.get_param_values(regularizable=True)
        )
        qfun_reg_param_norm = np.linalg.norm(
            self.qf.get_param_values(regularizable=True)
        )

        logger.record_tabular('Epoch', epoch)
        logger.record_tabular('Iteration', epoch)
        logger.record_tabular('AverageReturn', np.mean(returns))
        logger.record_tabular('StdReturn',
                              np.std(returns))
        logger.record_tabular('MaxReturn',
                              np.max(returns))
        logger.record_tabular('MinReturn',
                              np.min(returns))
        if len(self.es_path_returns) > 0:
            logger.record_tabular('AverageEsReturn',
                                  np.mean(self.es_path_returns))
            logger.record_tabular('StdEsReturn',
                                  np.std(self.es_path_returns))
            logger.record_tabular('MaxEsReturn',
                                  np.max(self.es_path_returns))
            logger.record_tabular('MinEsReturn',
                                  np.min(self.es_path_returns))
        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular('AverageQLoss', average_q_loss)
        logger.record_tabular('AveragePolicySurr', average_policy_surr)
        logger.record_tabular('AverageQ', np.mean(all_qs))
        logger.record_tabular('AverageAbsQ', np.mean(np.abs(all_qs)))
        logger.record_tabular('AverageY', np.mean(all_ys))
        logger.record_tabular('AverageAbsY', np.mean(np.abs(all_ys)))
        logger.record_tabular('AverageAbsQYDiff',
                              np.mean(np.abs(all_qs - all_ys)))
        logger.record_tabular('AverageAction', average_action)

        logger.record_tabular('PolicyRegParamNorm',
                              policy_reg_param_norm)
        logger.record_tabular('QFunRegParamNorm',
                              qfun_reg_param_norm)

        self.env.log_diagnostics(paths)
        self.policy.log_diagnostics(paths)

        self.qf_loss_averages = []
        self.policy_surr_averages = []

        self.q_averages = []
        self.y_averages = []
        self.es_path_returns = []

    def update_plot(self):
        if self.plot:
            plotter.update_plot(self.policy, self.max_path_length)

    def get_epoch_snapshot(self, epoch):
        return dict(
            env=self.env,
            epoch=epoch,
            qf=self.qf,
            policy=self.policy,
            target_qf=self.opt_info["target_qf"],
            target_policy=self.opt_info["target_policy"],
            es=self.es,
        )
