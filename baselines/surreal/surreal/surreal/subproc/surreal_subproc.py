import os
import sys
import math
import argparse
from copy import copy
from symphony.commandline import SymphonyParser
from symphony.engine import SymphonyConfig, Cluster
from benedict import BeneDict
from surreal.launch import (
    CommandGenerator,
    setup_network,
)
import surreal.utils as U

# Import to avoid duplicate build process
import mujoco_py
# Use spawn to avoid running into fork caused issues
from multiprocessing import set_start_method

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def _merge_setting_dictionaries(customize, base):
    di = copy(base)
    for key in di:
        if isinstance(di[key], dict):
            if key in customize:
                di[key] = _merge_setting_dictionaries(customize[key], di[key])
        else:
            if key in customize and customize[key] is not None:
                di[key] = customize[key]
    return di


class SubprocSurrealParser:
    def __init__(self):
        self.config = BeneDict()
        self.load_config()

    def load_config(self):
        surreal_yml_path = U.get_config_file()
        if not U.f_exists(surreal_yml_path):
            raise ValueError('Cannot find surreal config file at {}'
                             .format(surreal_yml_path))
        self.config = BeneDict.load_yaml_file(surreal_yml_path)
        SymphonyConfig().set_username(self.username)
        SymphonyConfig().set_experiment_folder(self.folder)

    @property
    def folder(self):
        if 'subproc_results_folder' not in self.config:
            raise KeyError('Please specify "subproc_results_folder" in ~/.surreal.yml')
        return U.f_expand(self.config.subproc_results_folder)

    @property
    def username(self):
        if 'username' not in self.config:
            raise KeyError('Please specify "username" in ~/.surreal.yml')
        return self.config.username

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('experiment_name')
        parser.add_argument(
            '-al',
            '--algorithm',
            type=str,
            default='ppo',
            help='ddpg / ppo or the location of algorithm python script'
        )
        parser.add_argument(
            '-na',
            '--num_agents',
            type=int,
            default=2,
            help='number of agent pods to run in parallel.'
        )
        parser.add_argument(
            '-ne',
            '--num_evals',
            type=int,
            default=1,
            help='number of eval pods to run in parallel.'
        )
        parser.add_argument(
            '--env',
            type=str,
            default='gym:HalfCheetah-v2',
            help='What environment to run'
        )
        parser.add_argument(
            '--gpu',
            type=str,
            default="",
            help='Which gpus to use: "auto" (default) uses all gpus '
                 'available through CUDA_VISIBLE_DEVICES, or use a '
                 'comma seperated list to override'
        )
        parser.add_argument(
            '-dr', '--dry-run',
            action='store_true',
            help='print the subprocess commands without actually running.'
        )
        return parser.parse_args()

    def action_create(self, args):
        """
            Spin up a multi-node distributed Surreal experiment.
            Put any command line args that pass to the config script after "--"
        """
        cluster = Cluster.new(
            'subproc',
            stdout_mode='print',
            stderr_mode='print',
            log_dir=None
        )
        experiment_name = args.experiment_name
        exp = cluster.new_experiment(experiment_name)

        algorithm_args = args.remainder
        algorithm_args += [
            "--num-agents",
            str(args.num_agents),
            ]
        experiment_folder = os.path.join(self.folder, experiment_name)
        print('Writing experiment output to {}'.format(experiment_folder))
        algorithm_args += ["--experiment-folder", experiment_folder]
        algorithm_args += ["--env", args.env]
        executable = self._find_executable(args.algorithm)
        cmd_gen = CommandGenerator(
            num_agents=args.num_agents,
            num_evals=args.num_evals,
            executable=executable,
            config_commands=algorithm_args
        )
        learner = exp.new_process(
            'learner',
            cmd=cmd_gen.get_command('learner'))
        # learner.set_env('DISABLE_MUJOCO_RENDERING', "1")

        replay = exp.new_process(
            'replay',
            cmd=cmd_gen.get_command('replay'))

        ps = exp.new_process(
            'ps',
            cmd=cmd_gen.get_command('ps'))

        tensorboard = exp.new_process(
            'tensorboard',
            cmd=cmd_gen.get_command('tensorboard'))

        tensorplex = exp.new_process(
            'tensorplex',
            cmd=cmd_gen.get_command('tensorplex'))

        loggerplex = exp.new_process(
            'loggerplex',
            cmd=cmd_gen.get_command('loggerplex'))

        agents = []
        for i in range(args.num_agents):
            agent_name = 'agent-{}'.format(i)
            agent = exp.new_process(
                agent_name,
                cmd=cmd_gen.get_command(agent_name))
            agents.append(agent)

        evals = []
        for i in range(args.num_evals):
            eval_name = 'eval-{}'.format(i)
            eval_p = exp.new_process(
                eval_name,
                cmd=cmd_gen.get_command(eval_name))
            evals.append(eval_p)

        setup_network(agents=agents,
                      evals=evals,
                      learner=learner,
                      replay=replay,
                      ps=ps,
                      tensorboard=tensorboard,
                      tensorplex=tensorplex,
                      loggerplex=loggerplex)
        self._setup_gpu(agents=agents,
                        evals=evals,
                        learner=learner,
                        gpus=args.gpu)
        cluster.launch(exp, dry_run=args.dry_run)

    def _find_executable(self, name):
        """
            Finds the .py file corresponding to the algorithm specified

        Args:
            name: ddpg / ppo / <path in container to compatible .py file>
        """
        if name == 'ddpg':
            return 'surreal-ddpg'
        elif name == 'ppo':
            return 'python3 -m surreal.main.ppo_configs'
        else:
            return name

    def _setup_gpu(self, agents, evals, learner, gpus):
        """
            Assigns GPU to agents and learners in an optimal way.
            No GPU, do nothing
        """
        actors = agents + evals
        if gpus == "auto":
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                gpus = os.environ['CUDA_VISIBLE_DEVICES']
            else:
                gpus = ''
        gpus_str = gpus
        assert gpus == ""
        try:
            gpus = [x for x in gpus_str.split(',') if len(x) > 0]
        except Exception as e:
            print("Error parsing GPU specification {}\n".format(gpus_str),
                  file=sys.stderr)
            raise e
        if len(gpus) == 0:
            print('Using CPU')
        elif len(gpus) == 1:
            gpu = gpus[0]
            print('Putting agents, evals and learner on GPU {}'.format(gpu))
            for actor in actors + [learner]:
                actor.set_envs({'CUDA_VISIBLE_DEVICES': gpu})
        elif len(gpus) > 1:
            learner_gpu = gpus[0]
            print('Putting learner on GPU {}'.format(learner_gpu))
            learner.set_envs({'CUDA_VISIBLE_DEVICES': learner_gpu})

            actors_per_gpu = float(len(actors)) / (len(gpus) - 1)
            actors_per_gpu = int(math.ceil(actors_per_gpu))
            print('Putting up to {} agents/evals on each of gpus {}'.format(
                    actors_per_gpu, ','.join([x for x in gpus[1:]])))

            for i, actor in enumerate(actors):
                cur_gpu = gpus[1 + i // actors_per_gpu]
                actor.set_envs({'CUDA_VISIBLE_DEVICES': cur_gpu})

    def main(self):
        assert sys.argv.count('--') <= 1, \
            'command line can only have at most one "--"'
        if '--' in sys.argv:
            idx = sys.argv.index('--')
            remainder = sys.argv[idx+1:]
            sys.argv = sys.argv[:idx]
            has_remainder = True  # even if remainder itself is empty
        else:
            remainder = []
            has_remainder = False

        args = self.parse_args()
        args.remainder = remainder
        args.has_remainder = has_remainder
        self.action_create(args)


def main():
    SubprocSurrealParser().main()


if __name__ == '__main__':
    set_start_method('spawn')
    main()
