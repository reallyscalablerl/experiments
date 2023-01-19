import gym
import ray
from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
import argparse

class HumanoidFixedEnv(gym.Env):
    def __init__(self):
        self._env = gym.make('Humanoid-v4')
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
    
    def reset(self):
        return self._env.reset()[0]
    
    def step(self, action):
        step_res = self._env.step(action)
        return step_res[0], step_res[1], step_res[2], step_res[-1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iters", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--expr_name", type=str, default="smac-test")
    args = parser.parse_args()
    
    ray.init()
    register_env('humanoid-v4-fixed', lambda _: HumanoidFixedEnv())

    tune.run_experiments(
    {
      args.expr_name:{
        "run": "IMPALA",
        "env": "humanoid-v4-fixed",
        "stop": {
          "training_iteration": args.num_iters,
        },
        "config": {
            "framework": "tf",
            "num_workers": args.num_workers,
            "num_cpus_per_worker": 2,
            "num_gpus": args.num_gpus, 
            "num_cpus_for_driver": 8,
            "num_envs_per_worker": 10,
            # "use_critic": True,
            # "use_gae": True, 
            # "model":{
            #     "fcnet_hiddens": [32, 32],
            #     "use_lstm": True
            # },
            "train_batch_size": 500*args.num_gpus,
            "rollout_fragment_length": 100,
            "num_sgd_iter": 1,
        },
      }
    }
  )