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
import pyprind
import tensorflow as tf
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
#from sandbox.rocky.tf.core.parameterized import suppress_params_loading
from rllab.core.serializable import Serializable
from sampling_utils import SimpleReplayPool
import pickle as pickle
from numpy import linalg as LA
import random
from random import randint


class Persistence_Length_Exploration(RLAlgorithm):
    """
    PolyRL Exploration Strategy
    """

    def __init__(
            self,
            env,
            policy,
            qf,
            L_p=0.08,
            b_step_size=0.0004,
            sigma = 0.1,
            max_exploratory_steps = 20,
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
            pause_for_plot=False):
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
        self.policy = policy
        self.qf = qf
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


        """
        PolyRL Hyperparameters
        """
        self.b_step_size = b_step_size
        self.L_p = L_p
        self.sigma = sigma
        self.max_exploratory_steps = max_exploratory_steps


        self.scale_reward = scale_reward
        self.train_policy_itr = 0

        self.opt_info = None

    def start_worker(self):
        parallel_sampler.populate_task(self.env, self.policy)
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    @overrides
    def lp_exploration(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # This seems like a rather sequential method
            pool = SimpleReplayPool(
                max_pool_size=self.replay_pool_size,
                observation_dim=self.env.observation_space.flat_dim,
                action_dim=self.env.action_space.flat_dim,
                replacement_prob=self.replacement_prob,
            )
            self.start_worker()

            with tf.variable_scope("sample_policy", reuse=True):
                sample_policy = Serializable.clone(self.policy)

            self.init_opt()

            # This initializes the optimizer parameters
            sess.run(tf.global_variables_initializer())

            itr = 0
            path_length = 0
            path_return = 0
            terminal = False
            initial = False
            observation = self.env.reset()

            self.initial_action = self.env.action_space.sample()

            chain_actions = np.array([self.initial_action])
            chain_states = np.array([observation])

            action_trajectory_chain= 0
            state_trajectory_chain = 0

            end_traj_action = 0
            end_traj_state = 0


            H_vector = np.random.uniform(low=self.env.action_space.low, high=self.env.action_space.high, size=(self.env.action_space.shape[0],))
            H = ( self.b_step_size / LA.norm(H_vector) ) * H_vector

            all_H = np.array([H])
            all_theta = np.array([])


            last_action_chosen = self.initial_action

            for epoch in range(self.max_exploratory_steps):

                print ("LP Exploration Episode", epoch)
                print ("Replay Buffer Sample Size", pool.size)

                # logger.push_prefix('epoch #%d | ' % epoch)
                # logger.log("Training started")
                train_qf_itr, train_policy_itr = 0, 0


                if epoch==0:
                    next_action = last_action_chosen + H
                else:
                    next_action = last_action_chosen


                for epoch_itr in pyprind.prog_bar(range(self.epoch_length)):

                    if self.env.action_space.shape[0] == 6:
                        one_vector = self.one_vector_6D()

                    elif self.env.action_space.shape[0] == 21:
                        one_vector = self.one_vector_21D()

                    elif self.env.action_space.shape[0] == 3:
                        one_vector = self.one_vector_3D()

                    elif self.env.action_space.shape[0] == 10:
                        one_vector = self.one_vector_10D()


                    theta_mean = np.arccos( np.exp(   np.true_divide(-self.b_step_size, self.L_p) )  ) * one_vector
                    sigma_iden = self.sigma**2*np.identity(self.env.action_space.shape[0]-1)

                    eta = np.random.multivariate_normal(theta_mean, sigma_iden)
                    eta = np.concatenate((np.array([0]), eta), axis=0)

                    """
                    Map H_t to Spherical coordinate
                    """
                    if self.env.action_space.shape[0] == 3:
                        H_conversion = self.cart2pol_3D(H)
                    elif self.env.action_space.shape[0] ==6:
                        H_conversion = self.cart2pol_6D(H)
                    elif self.env.action_space.shape[0] ==10:
                        H_conversion = self.cart2pol_10D(H)
                    elif self.env.action_space.shape[0] ==21:
                        H_conversion = self.cart2pol_21D(H)

                    H = H_conversion + eta


                    """
                    Map H_t to Cartesian coordinate
                    """
                    if self.env.action_space.shape[0] == 3:
                        H_conversion = self.pol2cart_3D(H)
                    elif self.env.action_space.shape[0] ==6:
                        H_conversion = self.pol2cart_6D(H)
                    elif self.env.action_space.shape[0] ==10:
                        H_conversion = self.pol2cart_10D(H)
                    elif self.env.action_space.shape[0] ==21:
                        H_conversion = self.cart2pol_21D(H)

                    H = H_conversion

                    phi_t = next_action
                    phi_t_1 = phi_t + H

                    chosen_action = np.array([phi_t_1])
                    chain_actions = np.append(chain_actions, chosen_action, axis=0)

                    chosen_action = chosen_action[0,:]



                    # Execute policy
                    if terminal:  # or path_length > self.max_path_length:
                        # Note that if the last time step ends an episode, the very
                        # last state and observation will be ignored and not added
                        # to the replay pool
                        observation = self.env.reset()
                        # self.es.reset()
                        sample_policy.reset()
                        self.es_path_returns.append(path_return)
                        path_length = 0
                        path_return = 0
                        initial = True
                    else:
                        initial = False



                    chosen_state, reward, terminal, _ = self.env.step(chosen_action)

                    chain_states = np.append(chain_states, np.array([chosen_state]), axis=0)

                    action = chosen_action
                    state = chosen_state
                    end_traj_state = chosen_state
                    end_traj_action = chosen_action

                    #updates to be used in next iteration
                    H = phi_t_1 - phi_t
                    all_H = np.append(all_H, np.array([H]), axis=0)
                    next_action = phi_t_1


                    path_length += 1
                    path_return += reward


                    if not terminal and path_length >= self.max_path_length:
                        terminal = True

                        #originally, it was only line above
                        #added these below
                        terminal_state = chosen_state
                        last_action_chosen = self.env.action_space.sample()
                        H_vector = np.random.uniform(low=self.env.action_space.low, high=self.env.action_space.high, size=(self.env.action_space.shape[0],))
                        H = ( self.b_step_size / LA.norm(H_vector) ) * H_vector
                        next_action = last_action_chosen + H

                        # path_length = 0
                        # path_return = 0
                        # state = self.env.reset()
                        # sample_policy.reset()

                        # only include the terminal transition in this case if the flag was set
                        if self.include_horizon_terminal_transitions:
                            pool.add_sample(observation, action, reward * self.scale_reward, terminal, initial)
                    else:
                        pool.add_sample(observation, action, reward * self.scale_reward, terminal, initial)

                    observation = state


                    if pool.size >= self.min_pool_size:
                        for update_itr in range(self.n_updates_per_sample):
                            # Train policy
                            batch = pool.random_batch(self.batch_size)
                            itrs = self.do_training(itr, batch)

                            train_qf_itr += itrs[0]
                            train_policy_itr += itrs[1]
                        sample_policy.set_param_values(self.policy.get_param_values())

                    itr += 1

                last_action_chosen = action
                #last_action_chosen = last_action_chosen[0, :]


            action_trajectory_chain = chain_actions
            state_trajectory_chain = chain_states
            end_trajectory_action = end_traj_action
            end_trajectory_state = end_traj_state


            self.env.terminate()
            self.policy.terminate()

            return self.qf, self.policy, action_trajectory_chain, state_trajectory_chain, end_trajectory_action, end_trajectory_state



    def init_opt(self):

        # First, create "target" policy and Q functions
        with tf.variable_scope("target_policy", reuse=True):
            target_policy = Serializable.clone(self.policy)
        with tf.variable_scope("target_qf", reuse=True):
            target_qf = Serializable.clone(self.qf)

        # target_policy = pickle.loads(pickle.dumps(self.policy))
        # target_qf = pickle.loads(pickle.dumps(self.qf))

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

        qf_weight_decay_term = 0.5 * self.qf_weight_decay * \
                               sum([tf.reduce_sum(tf.square(param)) for param in
                                    self.qf.get_params(regularizable=True)])

        qval = self.qf.get_qval_sym(obs, action)

        qf_loss = tf.reduce_mean(tf.square(yvar - qval))
        qf_reg_loss = qf_loss + qf_weight_decay_term

        policy_weight_decay_term = 0.5 * self.policy_weight_decay * \
                                   sum([tf.reduce_sum(tf.square(param))
                                        for param in self.policy.get_params(regularizable=True)])
        policy_qval = self.qf.get_qval_sym(
            obs, self.policy.get_action_sym(obs),
            deterministic=True
        )
        policy_surr = -tf.reduce_mean(policy_qval)

        policy_reg_surr = policy_surr + policy_weight_decay_term

        qf_input_list = [yvar, obs, action]
        policy_input_list = [obs]

        self.qf_update_method.update_opt(
            loss=qf_reg_loss, target=self.qf, inputs=qf_input_list)
        self.policy_update_method.update_opt(
            loss=policy_reg_surr, target=self.policy, inputs=policy_input_list)

        f_train_qf = tensor_utils.compile_function(
            inputs=qf_input_list,
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

    def do_training(self, itr, batch):

        obs, actions, rewards, next_obs, terminals = ext.extract(
            batch,
            "observations", "actions", "rewards", "next_observations",
            "terminals"
        )

        # compute the on-policy y values
        target_qf = self.opt_info["target_qf"]
        target_policy = self.opt_info["target_policy"]

        next_actions, _ = target_policy.get_actions(next_obs)
        next_qvals = target_qf.get_qval(next_obs, next_actions)

        ys = rewards + (1. - terminals) * self.discount * next_qvals.reshape(-1)

        f_train_qf = self.opt_info["f_train_qf"]
        qf_loss, qval, _ = f_train_qf(ys, obs, actions)
        target_qf.set_param_values(
            target_qf.get_param_values() * (1.0 - self.soft_target_tau) +
            self.qf.get_param_values() * self.soft_target_tau)
        self.qf_loss_averages.append(qf_loss)
        self.q_averages.append(qval)
        self.y_averages.append(ys)

        self.train_policy_itr += self.policy_updates_ratio
        train_policy_itr = 0

        while self.train_policy_itr > 0:
            f_train_policy = self.opt_info["f_train_policy"]
            policy_surr, _ = f_train_policy(obs)
            target_policy.set_param_values(
                target_policy.get_param_values() * (1.0 - self.soft_target_tau) +
                self.policy.get_param_values() * self.soft_target_tau)
            self.policy_surr_averages.append(policy_surr)
            self.train_policy_itr -= 1
            train_policy_itr += 1

        return 1, train_policy_itr

    def evaluate(self, epoch, pool):
        logger.log("Collecting samples for evaluation")
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


    def one_vector_3D(self):
        one_vec_x1 = random.randint(-1, 0)
        if one_vec_x1 == 0:
            one_vec_x1 = 1

        one_vec_x2 = random.randint(-1, 0)
        if one_vec_x2 == 0:
            one_vec_x2 = 1

        one_vector = np.array([one_vec_x1, one_vec_x2])

        return one_vector


    def one_vector_6D(self):
        one_vec_x1 = random.randint(-1, 0)
        if one_vec_x1 == 0:
            one_vec_x1 = 1

        one_vec_x2 = random.randint(-1, 0)
        if one_vec_x2 == 0:
            one_vec_x2 = 1

        one_vec_x3 = random.randint(-1, 0)
        if one_vec_x3 == 0:
            one_vec_x3 = 1

        one_vec_x4 = random.randint(-1, 0)
        if one_vec_x4 == 0:
            one_vec_x4 = 1

        one_vec_x5 = random.randint(-1, 0)
        if one_vec_x5 == 0:
            one_vec_x5 = 1

        one_vector = np.array([one_vec_x1, one_vec_x2, one_vec_x3, one_vec_x4, one_vec_x5])

        return one_vector

    def one_vector_10D(self):
        one_vec_x1 = random.randint(-1, 0)
        if one_vec_x1 == 0:
            one_vec_x1 = 1

        one_vec_x2 = random.randint(-1, 0)
        if one_vec_x2 == 0:
            one_vec_x2 = 1

        one_vec_x3 = random.randint(-1, 0)
        if one_vec_x3 == 0:
            one_vec_x3 = 1

        one_vec_x4 = random.randint(-1, 0)
        if one_vec_x4 == 0:
            one_vec_x4 = 1

        one_vec_x5 = random.randint(-1, 0)
        if one_vec_x5 == 0:
            one_vec_x5 = 1

        one_vec_x6 = random.randint(-1, 0)
        if one_vec_x6 == 0:
            one_vec_x6 = 1

        one_vec_x7 = random.randint(-1, 0)
        if one_vec_x7 == 0:
            one_vec_x7 = 1

        one_vec_x8 = random.randint(-1, 0)
        if one_vec_x8 == 0:
            one_vec_x8 = 1

        one_vec_x9 = random.randint(-1, 0)
        if one_vec_x9 == 0:
            one_vec_x9 = 1

        # one_vec_x10 = random.randint(-1, 0)
        # if one_vec_x10 == 0:
        #     one_vec_x10 = 1


        one_vector = np.array([one_vec_x1, one_vec_x2, one_vec_x3, one_vec_x4, one_vec_x5, one_vec_x6, one_vec_x7, one_vec_x8, one_vec_x9])

        return one_vector



    def one_vector_21D(self):
        one_vec_x1 = random.randint(-1, 0)
        if one_vec_x1 == 0:
            one_vec_x1 = 1

        one_vec_x2 = random.randint(-1, 0)
        if one_vec_x2 == 0:
            one_vec_x2 = 1

        one_vec_x3 = random.randint(-1, 0)
        if one_vec_x3 == 0:
            one_vec_x3 = 1

        one_vec_x4 = random.randint(-1, 0)
        if one_vec_x4 == 0:
            one_vec_x4 = 1

        one_vec_x5 = random.randint(-1, 0)
        if one_vec_x5 == 0:
            one_vec_x5 = 1

        one_vec_x6 = random.randint(-1, 0)
        if one_vec_x6 == 0:
            one_vec_x6 = 1

        one_vec_x7 = random.randint(-1, 0)
        if one_vec_x7 == 0:
            one_vec_x7 = 1

        one_vec_x8 = random.randint(-1, 0)
        if one_vec_x8 == 0:
            one_vec_x8 = 1

        one_vec_x9 = random.randint(-1, 0)
        if one_vec_x9 == 0:
            one_vec_x9 = 1

        one_vec_x10 = random.randint(-1, 0)
        if one_vec_x10 == 0:
            one_vec_x10 = 1

        one_vec_x11 = random.randint(-1, 0)
        if one_vec_x11 == 0:
            one_vec_x12 = 1

        one_vec_x12 = random.randint(-1, 0)
        if one_vec_x12 == 0:
            one_vec_x12 = 1

        one_vec_x13 = random.randint(-1, 0)
        if one_vec_x13 == 0:
            one_vec_x13 = 1

        one_vec_x14 = random.randint(-1, 0)
        if one_vec_x14 == 0:
            one_vec_x14 = 1

        one_vec_x15 = random.randint(-1, 0)
        if one_vec_x15 == 0:
            one_vec_x15 = 1

        one_vec_x16 = random.randint(-1, 0)
        if one_vec_x16 == 0:
            one_vec_x16 = 1

        one_vec_x17 = random.randint(-1, 0)
        if one_vec_x17 == 0:
            one_vec_x17 = 1

        one_vec_x18 = random.randint(-1, 0)
        if one_vec_x18 == 0:
            one_vec_x18 = 1

        one_vec_x19 = random.randint(-1, 0)
        if one_vec_x19 == 0:
            one_vec_x19 = 1

        one_vec_x20 = random.randint(-1, 0)
        if one_vec_x20 == 0:
            one_vec_x20 = 1

        one_vector = np.array([one_vec_x1, one_vec_x2, one_vec_x3, one_vec_x4, one_vec_x5, one_vec_x6, one_vec_x7, one_vec_x8, one_vec_x9, one_vec_x10, one_vec_x11, one_vec_x12, one_vec_x13, one_vec_x14, one_vec_x15, one_vec_x16, one_vec_x17, one_vec_x18, one_vec_x19, one_vec_x20 ])

        return one_vector


    def cart2pol_6D(self, cartesian):

        x_1 = cartesian[0]
        x_2 = cartesian[1]
        x_3 = cartesian[2]
        x_4 = cartesian[3]
        x_5 = cartesian[4]
        x_6 = cartesian[5]

        modulus = x_1**2 + x_2**2 + x_3**2 + x_4**2 + x_5**2 + x_6**2

        radius = np.sqrt(modulus)
        phi_1 = np.arccos(x_1 / radius)
        phi_2 = np.arccos(x_2 / radius)
        phi_3 = np.arccos(x_3 / radius)

        phi_4 = np.arccos(x_4 / (np.sqrt( x_4**2 + x_5**2 + x_6**2  )) )

        if x_6 >= 0:
            phi_5 = np.arccos(x_5 / (np.sqrt( x_5**2 + x_6**2 )))
        else:
            phi_5 = (2 * np.pi) - np.arccos(x_5 / (np.sqrt( x_5**2 + x_6**2 )))


        spherical = np.array([radius, phi_1, phi_2, phi_3, phi_4, phi_5])


        return spherical


    def cart2pol_3D(self, cartesian):

        x_1 = cartesian[0]
        x_2 = cartesian[1]
        x_3 = cartesian[2]


        modulus = x_1**2 + x_2**2 + x_3**2

        radius = np.sqrt(modulus)
        phi_1 = np.arccos(x_1 / radius)
        phi_2 = np.arccos(x_2 / radius)

        spherical = np.array([radius, phi_1, phi_2])


        return spherical


    def cart2pol_10D(self, cartesian):

        x_1 = cartesian[0]
        x_2 = cartesian[1]
        x_3 = cartesian[2]
        x_4 = cartesian[3]
        x_5 = cartesian[4]
        x_6 = cartesian[5]
        x_7 = cartesian[6]
        x_8 = cartesian[7]
        x_9 = cartesian[8]
        x_10 = cartesian[9]


        modulus = x_1**2 + x_2**2 + x_3**2  + x_4**2 + x_5**2 + x_6**2 + x_7**2 + x_8**2 + x_9**2 + x_10**2

        radius = np.sqrt(modulus)
        phi_1 = np.arccos(x_1 / radius)
        phi_2 = np.arccos(x_2 / radius)
        phi_3 = np.arccos(x_3 / radius)
        phi_4 = np.arccos(x_4 / radius)
        phi_5 = np.arccos(x_5 / radius)
        phi_6 = np.arccos(x_6 / radius)
        phi_7 = np.arccos(x_7 / radius)


        phi_8 = np.arccos(x_8 / ( np.sqrt(x_10**2 + x_9**2  + x_8**2) ) )

        if x_10 >= 0:
            phi_9 = np.arccos( x_9 / (  np.sqrt(x_10**2 + x_9**2)   ) )
        else:
            phi_9 = (2 * np.pi) - np.arccos( x_9 / (  np.sqrt(x_10**2 + x_9**2)   ) )

        spherical = np.array([radius, phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_7, phi_8, phi_9])

        return spherical




    def cart2pol_21D(self, cartesian):

        x_1 = cartesian[0]
        x_2 = cartesian[1]
        x_3 = cartesian[2]
        x_4 = cartesian[3]
        x_5 = cartesian[4]
        x_6 = cartesian[5]
        x_7 = cartesian[6]
        x_8 = cartesian[7]
        x_9 = cartesian[8]
        x_10 = cartesian[9]
        x_11 = cartesian[10]
        x_12 = cartesian[11]
        x_13 = cartesian[12]
        x_14 = cartesian[13]
        x_15 = cartesian[14]
        x_16 = cartesian[15]
        x_17 = cartesian[16]
        x_18 = cartesian[17]
        x_19 = cartesian[18]
        x_20 = cartesian[19]
        x_21 = cartesian[20]

        modulus = x_1**2 + x_2**2 + x_3**2  + x_4**2 + x_5**2 + x_6**2 + x_7**2 + x_8**2 + x_9**2 + x_10**2 + x_11**2 + x_12**2 + x_13**2 + x_14**2 + x_15**2 + x_16**2 + x_17**2 + x_18**2 + x_19**2 + x_20**2 + x_21**2

        radius = np.sqrt(modulus)
        phi_1 = np.arccos(x_1 / radius)
        phi_2 = np.arccos(x_2 / radius)
        phi_3 = np.arccos(x_3 / radius)
        phi_4 = np.arccos(x_4 / radius)
        phi_5 = np.arccos(x_5 / radius)
        phi_6 = np.arccos(x_6 / radius)
        phi_7 = np.arccos(x_7 / radius)
        phi_8 = np.arccos(x_8 / radius)
        phi_9 = np.arccos(x_9 / radius)
        phi_10 = np.arccos(x_10 / radius)
        phi_11 = np.arccos(x_11 / radius)
        phi_12 = np.arccos(x_12 / radius)
        phi_13 = np.arccos(x_13 / radius)
        phi_14 = np.arccos(x_14 / radius)
        phi_15 = np.arccos(x_15 / radius)
        phi_16 = np.arccos(x_16 / radius)
        phi_17 = np.arccos(x_17 / radius)
        phi_18 = np.arccos(x_18 / radius)


        phi_19 = np.arccos(x_19 / ( np.sqrt(x_21**2 + x_20**2  + x_19**2) ) )

        if x_21 >= 0:
            phi_20 = np.arccos( x_20 / (  np.sqrt(x_21**2 + x_20**2)   ) )
        else:
            phi_20 = (2 * np.pi) - np.arccos( x_20 / (  np.sqrt(x_21**2 + x_20**2)   ) )

        spherical = np.array([radius, phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_7, phi_8, phi_9, phi_10, phi_11, phi_12, phi_13, phi_14, phi_15, phi_16, phi_17, phi_18, phi_19, phi_20])

        return spherical





    def pol2cart_6D(self, polar):


        radius = polar[0]
        phi_1 = polar[1]
        phi_2 = polar[2]
        phi_3 = polar[3]
        phi_4 = polar[4]
        phi_5 = polar[5]

        x_1 = radius * np.cos(phi_1)
        x_2 = radius * np.sin(phi_1) * np.cos(phi_2)
        x_3 = radius * np.sin(phi_1) * np.sin(phi_2) * np.cos(phi_3)
        x_4 = radius * np.sin(phi_1) * np.sin(phi_2) * np.sin(phi_3) * np.cos(phi_4)
        x_5 = radius * np.sin(phi_1) * np.sin(phi_2) * np.sin(phi_3) * np.sin(phi_4) * np.cos(phi_5)
        x_6 = radius * np.sin(phi_1) * np.sin(phi_2) * np.sin(phi_3) * np.sin(phi_4) * np.sin(phi_5)

        cartesian = np.array([x_1, x_2, x_3, x_4, x_5, x_6])

        return cartesian


    def pol2cart_3D(self, polar):


        radius = polar[0]
        phi_1 = polar[1]
        phi_2 = polar[2]

        x_1 = radius * np.cos(phi_1)
        x_2 = radius * np.sin(phi_1) * np.cos(phi_2)
        x_3 = radius * np.sin(phi_1) * np.sin(phi_2)

        cartesian = np.array([x_1, x_2, x_3])


        return cartesian



    def pol2cart_10D(self, polar):


        radius = polar[0]
        phi_1 = polar[1]
        phi_2 = polar[2]
        phi_3 = polar[3]
        phi_4 = polar[4]
        phi_5 = polar[5]
        phi_6 = polar[6]
        phi_7 = polar[7]
        phi_8 = polar[8]
        phi_9 = polar[9]



        x_1 = radius * np.cos(phi_1)
        x_2 = radius * np.sin(phi_1) * np.cos(phi_2)
        x_3 = radius * np.sin(phi_1) * np.sin(phi_2) * np.cos(phi_3)
        x_4 = radius * np.sin(phi_1) * np.sin(phi_2) * np.sin(phi_3) * np.cos(phi_4)
        x_5 = radius * np.sin(phi_1) * np.sin(phi_2) * np.sin(phi_3) * np.sin(phi_4) * np.cos(phi_5)
        x_6 = radius * np.sin(phi_1) * np.sin(phi_2) * np.sin(phi_3) * np.sin(phi_4) * np.sin(phi_5) * np.cos(phi_6)
        x_7 = radius * np.sin(phi_1) * np.sin(phi_2) * np.sin(phi_3) * np.sin(phi_4) * np.sin(phi_5) * np.sin(phi_6) * np.cos(phi_7)
        x_8 = radius * np.sin(phi_1) * np.sin(phi_2) * np.sin(phi_3) * np.sin(phi_4) * np.sin(phi_5) * np.sin(phi_6) * np.sin(phi_7) * np.cos(phi_8)
        x_9 = radius * np.sin(phi_1) * np.sin(phi_2) * np.sin(phi_3) * np.sin(phi_4) * np.sin(phi_5) * np.sin(phi_6) * np.sin(phi_7) * np.sin(phi_8) * np.cos(phi_9)
        x_10 = radius * np.sin(phi_1) * np.sin(phi_2) * np.sin(phi_3) * np.sin(phi_4) * np.sin(phi_5) * np.sin(phi_6) * np.sin(phi_7)* np.sin(phi_8) * np.sin(phi_9)



        cartesian = np.array([x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10])

        return cartesian






    def pol2cart_21D(self, polar):


        radius = polar[0]
        phi_1 = polar[1]
        phi_2 = polar[2]
        phi_3 = polar[3]
        phi_4 = polar[4]
        phi_5 = polar[5]
        phi_6 = polar[6]
        phi_7 = polar[7]
        phi_8 = polar[8]
        phi_9 = polar[9]
        phi_10 = polar[10]
        phi_11 = polar[11]
        phi_12 = polar[12]
        phi_13 = polar[13]
        phi_14 = polar[14]
        phi_15 = polar[15]
        phi_16 = polar[16]
        phi_17 = polar[17]
        phi_18 = polar[18]
        phi_19 = polar[19]
        phi_20 = polar[20]



        x_1 = radius * np.cos(phi_1)
        x_2 = radius * np.sin(phi_1) * np.cos(phi_2)
        x_3 = radius * np.sin(phi_1) * np.sin(phi_2) * np.cos(phi_3)
        x_4 = radius * np.sin(phi_1) * np.sin(phi_2) * np.sin(phi_3) * np.cos(phi_4)
        x_5 = radius * np.sin(phi_1) * np.sin(phi_2) * np.sin(phi_3) * np.sin(phi_4) * np.cos(phi_5)
        x_6 = radius * np.sin(phi_1) * np.sin(phi_2) * np.sin(phi_3) * np.sin(phi_4) * np.sin(phi_5) * np.cos(phi_6)
        x_7 = radius * np.sin(phi_1) * np.sin(phi_2) * np.sin(phi_3) * np.sin(phi_4) * np.sin(phi_5) * np.sin(phi_6) * np.cos(phi_7)
        x_8 = radius * np.sin(phi_1) * np.sin(phi_2) * np.sin(phi_3) * np.sin(phi_4) * np.sin(phi_5) * np.sin(phi_6) * np.sin(phi_7) * np.cos(phi_8)
        x_9 = radius * np.sin(phi_1) * np.sin(phi_2) * np.sin(phi_3) * np.sin(phi_4) * np.sin(phi_5) * np.sin(phi_6) * np.sin(phi_7) * np.sin(phi_8) * np.cos(phi_9)
        x_10 = radius * np.sin(phi_1) * np.sin(phi_2) * np.sin(phi_3) * np.sin(phi_4) * np.sin(phi_5) * np.sin(phi_6) * np.sin(phi_7)* np.sin(phi_8) * np.sin(phi_9) * np.cos(phi_10)
        x_11 = radius * np.sin(phi_1) * np.sin(phi_2) * np.sin(phi_3) * np.sin(phi_4) * np.sin(phi_5) * np.sin(phi_6) * np.sin(phi_7) *  np.sin(phi_8) * np.sin(phi_9) * np.sin(phi_10) * np.cos(phi_11)
        x_12 = radius * np.sin(phi_1) * np.sin(phi_2) * np.sin(phi_3) * np.sin(phi_4) * np.sin(phi_5) * np.sin(phi_6) * np.sin(phi_7) *  np.sin(phi_8) * np.sin(phi_9) * np.sin(phi_10) * np.sin(phi_11) * np.cos(phi_12)
        x_13 = radius * np.sin(phi_1) * np.sin(phi_2) * np.sin(phi_3) * np.sin(phi_4) * np.sin(phi_5) * np.sin(phi_6) * np.sin(phi_7) *  np.sin(phi_8) * np.sin(phi_9) * np.sin(phi_10) * np.sin(phi_11) * np.sin(phi_12) * np.cos(phi_13)
        x_14 = radius * np.sin(phi_1) * np.sin(phi_2) * np.sin(phi_3) * np.sin(phi_4) * np.sin(phi_5) * np.sin(phi_6) * np.sin(phi_7) *  np.sin(phi_8) * np.sin(phi_9) * np.sin(phi_10) * np.sin(phi_11) * np.sin(phi_12) * np.sin(phi_13) * np.cos(phi_14)
        x_15 = radius * np.sin(phi_1) * np.sin(phi_2) * np.sin(phi_3) * np.sin(phi_4) * np.sin(phi_5) * np.sin(phi_6) * np.sin(phi_7) *  np.sin(phi_8) * np.sin(phi_9) * np.sin(phi_10) * np.sin(phi_11) * np.sin(phi_12) * np.sin(phi_13) * np.sin(phi_14) * np.cos(phi_15)
        x_16 = radius * np.sin(phi_1) * np.sin(phi_2) * np.sin(phi_3) * np.sin(phi_4) * np.sin(phi_5) * np.sin(phi_6) * np.sin(phi_7) *  np.sin(phi_8) * np.sin(phi_9) * np.sin(phi_10) * np.sin(phi_11) * np.sin(phi_12) * np.sin(phi_13) * np.sin(phi_14) * np.sin(phi_15) * np.cos(phi_16)
        x_17 = radius * np.sin(phi_1) * np.sin(phi_2) * np.sin(phi_3) * np.sin(phi_4) * np.sin(phi_5) * np.sin(phi_6) * np.sin(phi_7) *  np.sin(phi_8) * np.sin(phi_9) * np.sin(phi_10) * np.sin(phi_11) * np.sin(phi_12) * np.sin(phi_13) * np.sin(phi_14) * np.sin(phi_15) * np.sin(phi_16) * np.cos(phi_17)
        x_18 = radius * np.sin(phi_1) * np.sin(phi_2) * np.sin(phi_3) * np.sin(phi_4) * np.sin(phi_5) * np.sin(phi_6) * np.sin(phi_7) *  np.sin(phi_8) * np.sin(phi_9) * np.sin(phi_10) * np.sin(phi_11) * np.sin(phi_12) * np.sin(phi_13) * np.sin(phi_14) * np.sin(phi_15) * np.sin(phi_16) * np.sin(phi_17) * np.cos(phi_18)
        x_19 = radius * np.sin(phi_1) * np.sin(phi_2) * np.sin(phi_3) * np.sin(phi_4) * np.sin(phi_5) * np.sin(phi_6) * np.sin(phi_7) *  np.sin(phi_8) * np.sin(phi_9) * np.sin(phi_10) * np.sin(phi_11) * np.sin(phi_12) * np.sin(phi_13) * np.sin(phi_14) * np.sin(phi_15) * np.sin(phi_16) * np.sin(phi_17) * np.sin(phi_18) * np.cos(phi_19)
        x_20 = radius * np.sin(phi_1) * np.sin(phi_2) * np.sin(phi_3) * np.sin(phi_4) * np.sin(phi_5) * np.sin(phi_6) * np.sin(phi_7) *  np.sin(phi_8) * np.sin(phi_9) * np.sin(phi_10) * np.sin(phi_11) * np.sin(phi_12) * np.sin(phi_13) * np.sin(phi_14) * np.sin(phi_15) * np.sin(phi_16) * np.sin(phi_17) * np.sin(phi_18) * np.sin(phi_19) * np.cos(phi_20)
        x_21 = radius * np.sin(phi_1) * np.sin(phi_2) * np.sin(phi_3) * np.sin(phi_4) * np.sin(phi_5) * np.sin(phi_6) * np.sin(phi_7) *  np.sin(phi_8) * np.sin(phi_9) * np.sin(phi_10) * np.sin(phi_11) * np.sin(phi_12) * np.sin(phi_13) * np.sin(phi_14) * np.sin(phi_15) * np.sin(phi_16) * np.sin(phi_17) * np.sin(phi_18) * np.sin(phi_19) * np.sin(phi_20)



        cartesian = np.array([x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21 ])

        return cartesian
