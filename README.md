# UnifiedPolicyGradients

Running

setup rllab following the directions:

https://github.com/openai/rllab

<code>
$ export PYTHONPATH=~/path/to/rllab/:$PYTHONPATH

$ source activate rllab3
</code>

## for unified gated

python run_ddpg_variations.py unified-gated Hopper-v1 --num_epochs 1000 --data_dir ./gated/

## for baseline ddpg

python run_ddpg_variations.py regular Hopper-v1 --num_epochs 1000 --data_dir ./gated/

