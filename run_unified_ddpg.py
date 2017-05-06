from unifying_policy_gradient.ddpg_unified_gated import DDPG as GatedDDPG
from unifying_policy_gradient.ddpg_unified import DDPG
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.exploration_strategies.ou_strategy import OUStrategy
from sandbox.rocky.tf.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from sandbox.rocky.tf.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction

from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.gym_env import GymEnv
import pickle
import tensorflow as tf

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", default=100, type=int)
parser.add_argument("--plot", action="store_true")
parser.add_argument("--use_gated", action="store_true")
parser.add_argument("--data_dir", default="./data/")
args = parser.parse_args()

stub(globals())

env = TfEnv(normalize(CartpoleEnv()))

policy = DeterministicMLPPolicy(
    env_spec=env.spec,
    name="policy",
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(32, 32)
)

es = OUStrategy(env_spec=env.spec)

qf = ContinuousMLPQFunction(env_spec=env.spec)

if args.use_gated:
    ddpg_class = GatedDDPG
else:
    ddpg_class = DDPG



algo = ddpg_class(
    env=env,
    policy=policy,
    es=es,
    qf=qf,
    batch_size=32,
    max_path_length=100,
    epoch_length=1000,
    min_pool_size=10000,
    n_epochs=args.num_epochs,
    discount=0.99,
    scale_reward=0.01,
    qf_learning_rate=1e-3,
    policy_learning_rate=1e-4,
    # Uncomment both lines (this and the plot parameter below) to enable plotting
    plot=args.plot,
)


run_experiment_lite(
    algo.train(),
    log_dir=args.data_dir,
    # Number of parallel workers for sampling
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    exp_name="Unified_DDPG_CartPole",
    seed=1,
    plot=args.plot,
)
