from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import ray
from ray.tune import run_experiments, register_env
from ray.rllib.models import ModelCatalog

from env import RLlibStarCraft2Env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iters", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--expr_name", type=str, default="smac-test")
    parser.add_argument("--map_name", type=str, default="27m_vs_30m")
    args = parser.parse_args()

    ray.init()

    register_env("smac", lambda smac_args: RLlibStarCraft2Env(**smac_args))
    # ModelCatalog.register_custom_model("mask_model", MaskedActionsModel)

    run_experiments(
        {
            args.expr_name: {
                "run": "IMPALA",
                "env": "smac",
                "stop": {
                    "training_iteration": args.num_iters,
                },
                "config": {
                    "framework": "tf",
                    "num_workers": args.num_workers,
                    "num_envs_per_worker": 10,
                    "num_cpus_per_worker": 2,
                    "num_gpus": args.num_gpus, 
                    "num_cpus_for_driver": 8,
                    "train_batch_size": 500*args.num_gpus,
                    "rollout_fragment_length": 100,
                    # "use_critic": True,
                    # "use_gae": True, 
                    "num_sgd_iter": 1,
                    # 'entropy_coeff': 0.01,
                    "env_config": {
                        "map_name": "27m_vs_30m",
                    },
                    # "model": {
                    #     "fcnet_hiddens": [32, 32],
                    #     "use_lstm": True,
                    # },
                    "ignore_worker_failures": True,
                    # "recreate_failed_workers": True,
                    # "restart_failed_sub_environments": True
                },
            },
        }
    )