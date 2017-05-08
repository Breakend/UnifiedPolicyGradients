from ddpg_tensorflow.ddpg_polyRL import DDPG
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.exploration_strategies.ou_strategy import OUStrategy

from exploration_strategies_tensorflow.persistence_length_higher_dimensions import Persistence_Length_Exploration

from sandbox.rocky.tf.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from sandbox.rocky.tf.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction

from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.gym_env import GymEnv
from rllab.misc import ext
import pickle
import tensorflow as tf

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("type", help="Type of DDPG to run: unified, unified-gated, regular")
parser.add_argument("env", help="The environment name from OpenAIGym environments")
parser.add_argument("--num_epochs", default=100, type=int)
parser.add_argument("--plot", action="store_true")
# parser.add_argument("--data_dir", default="./data/")
args = parser.parse_args()

stub(globals())
ext.set_seed(1)

supported_gym_envs = ["MountainCarContinuous-v0", "Hopper-v1", "Walker2d-v1", "Humanoid-v1", "Reacher-v1", "HalfCheetah-v1", "Swimmer-v1", "HumanoidStandup-v1"]

other_env_class_map  = { "Cartpole" :  CartpoleEnv}

if args.env in supported_gym_envs:
    gymenv = GymEnv(args.env, force_reset=True, record_video=False, record_log=False)
    # gymenv.env.seed(1)
else:
    gymenv = other_env_class_map[args.env]()

#TODO: assert continuous space


env = TfEnv(normalize(gymenv))

policy = DeterministicMLPPolicy(
    env_spec=env.spec,
    name="policy",
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(100, 50, 25),
    hidden_nonlinearity=tf.nn.relu,
)

es = OUStrategy(env_spec=env.spec)

qf = ContinuousMLPQFunction(env_spec=env.spec,
                            hidden_sizes=(100,100),
                            hidden_nonlinearity=tf.nn.relu,)



max_exploratory_steps_iters = 11
batch_size_value = 32
num_episodes = 3000
steps_per_episode = 1000

"""
Persistence Length Exploration
"""
lp = Persistence_Length_Exploration(
    env=env, 
    qf=qf, 
    policy=policy,
    L_p=0.2,
    b_step_size=0.0004, 
    sigma = 0.1,
    max_exploratory_steps = max_exploratory_steps_iters,
    batch_size=batch_size_value,
    n_epochs=args.num_epochs,
    scale_reward=1.0,
    qf_learning_rate=1e-3,
    policy_learning_rate=1e-4,
)


ddpg_type_map = {"regular" : DDPG}


ddpg_class = ddpg_type_map[args.type]

# n_itr = int(np.ceil(float(n_episodes*max_path_length)/flags['batch_size']))

algo = ddpg_class(
    env=env,
    policy=policy,
    qf=qf,
    lp=lp,
    es=es,
    batch_size=32,
    max_path_length=env.horizon,
    epoch_length=1000,
    min_pool_size=10000,
    n_epochs=args.num_epochs,
    discount=0.99,
    scale_reward=1.0,
    qf_learning_rate=1e-3,
    policy_learning_rate=1e-4,
    # Uncomment both lines (this and the plot parameter below) to enable plotting
    plot=args.plot,
)


run_experiment_lite(
    algo.train(),
    # log_dir=args.data_dir,
    # Number of parallel workers for sampling
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    exp_name="Tensorflow_RLLAB_Results/" + "DDPG/" + "Hopper/",
    seed=1,
    plot=args.plot,
)
