# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A simple example of setting up a multi-agent version of GFootball with rllib.
"""

import argparse
from gfootball.env import create_environment
import gym
import ray
from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env


class RllibGFootball(MultiAgentEnv):
  """An example of a wrapper for GFootball to make it compatible with rllib."""

  def __init__(self):
    self.env = create_environment(
                                env_name="11_vs_11_stochastic",
                                number_of_left_players_agent_controls=11,
                                number_of_right_players_agent_controls=11,
                                representation="simple115v2",
                            )
    self.action_space = gym.spaces.Discrete(self.env.action_space.nvec[1])
    self.observation_space = gym.spaces.Box(
        low=self.env.observation_space.low[0],
        high=self.env.observation_space.high[0],
        dtype=self.env.observation_space.dtype)
    self.num_agents = 22

  def reset(self):
    original_obs = self.env.reset()
    obs = {}
    for x in range(self.num_agents):
      if self.num_agents > 1:
        obs['agent_%d' % x] = original_obs[x]
      else:
        obs['agent_%d' % x] = original_obs
    return obs

  def step(self, action_dict):
    actions = []
    for key, value in sorted(action_dict.items()):
      actions.append(value)
    o, r, d, i = self.env.step(actions)
    rewards = {}
    obs = {}
    infos = {}
    for pos, key in enumerate(sorted(action_dict.keys())):
      infos[key] = i
      if self.num_agents > 1:
        rewards[key] = r[pos]
        obs[key] = o[pos]
      else:
        rewards[key] = r
        obs[key] = o
    dones = {'__all__': d}
    return obs, rewards, dones, infos


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--num_iters', type=int, default=50)
  parser.add_argument('--num_gpus', type=int, default=1)
  parser.add_argument('--num_workers', type=int, default=200)
  parser.add_argument('--expr_name', type=str, default="default-run")
  parser.add_argument('--simple', action='store_true')

  args = parser.parse_args()
  ray.init()

  # Simple environment with `num_agents` independent players
  register_env('gfootball', lambda _: RllibGFootball())
  single_env = RllibGFootball()
  obs_space = single_env.observation_space
  act_space = single_env.action_space

  def gen_policy(_):
    return (None, obs_space, act_space, {})

  # Setup PPO with an ensemble of `num_policies` different policies
  policies = {
      'policy_{}'.format(i): gen_policy(i) for i in range(1)
  }
  policy_ids = list(policies.keys())

  tune.run_experiments(
    {
      args.expr_name:{
        "run": "IMPALA",
        "env": "gfootball",
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
            "train_batch_size": 500*args.num_gpus,
            "rollout_fragment_length": 100,
            "num_sgd_iter": 1,
            # "model": {
            #     "fcnet_hiddens": [32, 32],
            #     "use_lstm": True,
            # },
            # "ignore_worker_failures": True,
            # "recreate_failed_workers": True,
        },
      }
    }
  )
