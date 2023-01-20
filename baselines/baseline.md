# Baselines

## RLlib (Version 2.0.0)

Usage: run `rllib/run_all_envs.sh` in a running `ray` cluster with enough resources. Support **atari**, **Google Football**, **Gym MuJoCo** and **SMAC** environments. 

Note that all experiments use models and APPO algorithm inherently implemented in RLlib with the same configuraiton (See [rllib/atari/atari-ppo.py](rllib/atari/atari-ppo.py) for a detailed example) of experiments in our paper. 

Credit to https://github.com/oxwhirl/smac.git and https://github.com/google-research/football for example environment wrappers.

## Surreal

We modified `surreal` to let it support multi-GPU training on a single node. DDP support is added in `surreal/learner/ppo.py`. Modified version of `surreal` is provided in this directory. To run `surreal` with multiple GPUs, under the surreal directory, change the number of GPU in `surreal/world_size.py` and run `run.sh`.