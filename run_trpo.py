from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv

from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.misc import ext
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp

import pickle
import os.path as osp

import tensorflow as tf

import argparse
parser = argparse.ArgumentParser()
# parser.add_argument("type", help="Type of DDPG to run: ['unified-decaying', 'unified-gated-decaying', 'unified', 'unified-gated', 'regular']")
parser.add_argument("env", help="The environment name from OpenAIGym environments")
parser.add_argument("--num_epochs", default=100, type=int)
parser.add_argument("--data_dir", default="./data/")
parser.add_argument("--use_ec2", action="store_true", help="Use your ec2 instances if configured")
parser.add_argument("--dont_terminate_machine", action="store_false", help="Whether to terminate your spot instance or not. Be careful.")
args = parser.parse_args()

stub(globals())
ext.set_seed(1)

supported_gym_envs = ["MountainCarContinuous-v0", "InvertedPendulum-v1", "InvertedDoublePendulum-v1", "Hopper-v1", "Walker2d-v1", "Humanoid-v1", "Reacher-v1", "HalfCheetah-v1", "Swimmer-v1", "HumanoidStandup-v1"]

other_env_class_map  = { "Cartpole" :  CartpoleEnv}

if args.env in supported_gym_envs:
    gymenv = GymEnv(args.env, force_reset=True, record_video=False, record_log=False)
    # gymenv.env.seed(1)
else:
    gymenv = other_env_class_map[args.env]()

#TODO: assert continuous space


env = TfEnv(normalize(gymenv))


policy = GaussianMLPPolicy(
name="policy",
env_spec=env.spec,
# The neural network policy should have two hidden layers, each with 32 hidden units.
hidden_sizes=(100, 50, 25),
hidden_nonlinearity=tf.nn.relu,
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=5000,
    max_path_length=env.horizon,
    n_itr=args.num_epochs,
    discount=0.99,
    step_size=0.01,
    optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
)

run_experiment_lite(
    algo.train(),
    log_dir=None if args.use_ec2 else args.data_dir,
    # Number of parallel workers for sampling
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    exp_prefix="UnifiedDDPG_" + args.env + "_trpo",
    seed=1,
    mode="ec2" if args.use_ec2 else "local",
    plot=False,
    # dry=True,
    terminate_machine=args.dont_terminate_machine,
    added_project_directories=[osp.abspath(osp.join(osp.dirname(__file__), '.'))]
)
