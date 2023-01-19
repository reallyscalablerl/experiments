import gym
import ray
from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iters", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--expr_name", type=str, default="atari-test")
    args = parser.parse_args()

    ray.init()

    tune.run_experiments(
    {
      args.expr_name:{
        "run": "IMPALA",
        # "env": "pongnoframeskip-v4-fixed",
        "env": "PongNoFrameskip-v4",
        "stop": {
          "training_iteration": args.num_iters,
        },
        "config": {
            "framework": "tf",
            "num_workers": args.num_workers,
            "num_cpus_per_worker": 1,
            "num_gpus": args.num_gpus, 
            "num_cpus_for_driver": 8,
            "num_envs_per_worker": 10,
            # "use_critic": True,
            # "use_gae": True, 
            # "model":{
            #     "fcnet_hiddens": [32, 32],
            #     "use_lstm": True,
            #     "dim": 96
            # },
            "train_batch_size": 500*args.num_gpus,
            "rollout_fragment_length": 100,
            "num_sgd_iter": 1,
        },
      }
    }
  )