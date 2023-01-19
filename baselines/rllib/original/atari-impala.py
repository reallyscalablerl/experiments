import gym
import ray
from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
import argparse

class PongNoFrameskipFixedEnv(gym.Env):
    def __init__(self):
        self._env = gym.make('PongNoFrameskip-v4')
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
    
    def reset(self):
        return self._env.reset()
    
    def step(self, action):
        step_res = self._env.step(action)
        # print(type(step_res), len(step_res), step_res)
        return step_res[0], step_res[1], step_res[2], step_res[-1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iters", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--expr_name", type=str, default="atari-test")
    args = parser.parse_args()

    ray.init()
    register_env('pongnoframeskip-v4-fixed', lambda _: PongNoFrameskipFixedEnv())

    tune.run_experiments(
    {
      args.expr_name:{
        "run": "IMPALA",
        "env": 'pongnoframeskip-v4-fixed',
        "stop": {
          "training_iteration": args.num_iters,
        },
        "config": {
            # V-trace params (see vtrace_tf/torch.py).
            "vtrace": True,
            "vtrace_clip_rho_threshold": 1.0,
            "vtrace_clip_pg_rho_threshold": 1.0,
            # If True, drop the last timestep for the vtrace calculations, such that
            # all data goes into the calculations as [B x T-1] (+ the bootstrap value).
            # This is the default and legacy RLlib behavior, however, could potentially
            # have a destabilizing effect on learning, especially in sparse reward
            # or reward-at-goal environments.
            # False for not dropping the last timestep.
            "vtrace_drop_last_ts": True,
            # System params.
            #
            # == Overview of data flow in IMPALA ==
            # 1. Policy evaluation in parallel across `num_workers` actors produces
            #    batches of size `rollout_fragment_length * num_envs_per_worker`.
            # 2. If enabled, the replay buffer stores and produces batches of size
            #    `rollout_fragment_length * num_envs_per_worker`.
            # 3. If enabled, the minibatch ring buffer stores and replays batches of
            #    size `train_batch_size` up to `num_sgd_iter` times per batch.
            # 4. The learner thread executes data parallel SGD across `num_gpus` GPUs
            #    on batches of size `train_batch_size`.
            #
            "rollout_fragment_length": 50,
            "train_batch_size": 500,
            "min_time_s_per_reporting": 10,
            "num_workers": 128,
            # Number of GPUs the learner should use.
            "num_gpus": 2,
            # For each stack of multi-GPU towers, how many slots should we reserve for
            # parallel data loading? Set this to >1 to load data into GPUs in
            # parallel. This will increase GPU memory usage proportionally with the
            # number of stacks.
            # Example:
            # 2 GPUs and `num_multi_gpu_tower_stacks=3`:
            # - One tower stack consists of 2 GPUs, each with a copy of the
            #   model/graph.
            # - Each of the stacks will create 3 slots for batch data on each of its
            #   GPUs, increasing memory requirements on each GPU by 3x.
            # - This enables us to preload data into these stacks while another stack
            #   is performing gradient calculations.
            "num_multi_gpu_tower_stacks": 1,
            # How many train batches should be retained for minibatching. This conf
            # only has an effect if `num_sgd_iter > 1`.
            "minibatch_buffer_size": 1,
            # Number of passes to make over each train batch.
            "num_sgd_iter": 1,
            # Set >0 to enable experience replay. Saved samples will be replayed with
            # a p:1 proportion to new data samples.
            "replay_proportion": 0.0,
            # Number of sample batches to store for replay. The number of transitions
            # saved total will be (replay_buffer_num_slots * rollout_fragment_length).
            "replay_buffer_num_slots": 0,
            # Max queue size for train batches feeding into the learner.
            "learner_queue_size": 16,
            # Wait for train batches to be available in minibatch buffer queue
            # this many seconds. This may need to be increased e.g. when training
            # with a slow environment.
            "learner_queue_timeout": 300,
            # Level of queuing for sampling.
            # "max_sample_requests_in_flight_per_worker": 2,
            # Max number of workers to broadcast one set of weights to.
            # "broadcast_interval": 1,
            # Use n (`num_aggregation_workers`) extra Actors for multi-level
            # aggregation of the data produced by the m RolloutWorkers
            # (`num_workers`). Note that n should be much smaller than m.
            # This can make sense if ingesting >2GB/s of samples, or if
            # the data requires decompression.
            # "num_aggregation_workers": 0,

            # Learning params.
            "grad_clip": 40.0,
            # Either "adam" or "rmsprop".
            "opt_type": "adam",
            "lr": 0.0005,
            "lr_schedule": None,
            # `opt_type=rmsprop` settings.
            "decay": 0.99,
            "momentum": 0.0,
            "epsilon": 0.1,
            # Balancing the three losses.
            "vf_loss_coeff": 0.5,
            "entropy_coeff": 0.01,
            "entropy_coeff_schedule": None,
            # Set this to true to have two separate optimizers optimize the policy-
            # and value networks.
            "_separate_vf_optimizer": False,
            # If _separate_vf_optimizer is True, define separate learning rate
            # for the value network.
            "_lr_vf": 0.0005,

            # Callback for APPO to use to update KL, target network periodically.
            # The input to the callback is the learner fetches dict.
            "after_train_step": None,
        },
      }
    }
  )