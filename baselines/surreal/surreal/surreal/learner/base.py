"""
Template class for all learners
"""
import os
import threading
import queue
import time
import numpy as np
from pathlib import Path
from benedict import BeneDict
import surreal.utils as U
from surreal.session import (
    TimeThrottledTensorplex,
    get_loggerplex_client,
    get_tensorplex_client,
    Config
)
from surreal.distributed import ParameterPublisher, LearnerDataPrefetcher
from surreal.world_size import SURREAL_WORLD_SIZE


class Learner(metaclass=U.AutoInitializeMeta):
    """
        Important: When extending this class, make sure to follow the init
            method signature so that orchestrating functions can properly
            initialize the learner.
    """
    def __init__(self,
                 learner_config,
                 env_config,
                 session_config):
        """
        Initializes the learner instance

        Args:
            learner_config, env_config, session_config: configs that define
                an experiment
        """
        self.learner_config = learner_config
        self.env_config = env_config
        self.session_config = session_config
        self.current_iter = 0

        self._setup_logging()
        self._setup_checkpoint()

    def learn(self, batch_exp):
        """
        Abstract method runs one step of learning

        Args:
            batch_exp: batched experience, format is a list of whatever
                experience sender wrapper returns

        Returns:
            td_error or other values for prioritized replay
        """
        raise NotImplementedError

    def module_dict(self):
        """
            Dict of modules to be broadcasted to the parameter server.
            MUST be consistent with the agent's `module_dict()`
        """
        raise NotImplementedError

    def save(self, file_path):
        """
            Saves checkpoint to disk

        Args:
            file_path: locatioin to save
        """
        raise NotImplementedError

    def checkpoint_attributes(self):
        """
            This function defines what attributes should be serialized
            when we are saving a checkpoint.

            See implementations in DDPGLearner and PPOLearner for examples

        Returns:
            list of attributes to be tracked by checkpoint
        """
        return []

    ######
    # Internal, including Communication, etc.
    ######
    def _setup_publish(self):
        min_publish_interval = \
            self.learner_config.parameter_publish.min_publish_interval
        self._ps_publish_tracker = U.TimedTracker(min_publish_interval)

        ps_publish_port = os.environ['SYMPH_PARAMETER_PUBLISH_PORT']
        self._ps_publisher = ParameterPublisher(
            port=ps_publish_port,
            module_dict=self.module_dict()
            # This must happen after subclass __init__
        )

    def _initialize(self):
        """
            For AutoInitializeMeta interface
        """
        if self.session_config.checkpoint.restore:
            self.restore_checkpoint()

        self._setup_publish()
        # Logging should only start here so that all components are
        # properly initialized
        self._tensorplex_thread.start()

    ######
    # Parameter publish
    ######
    def should_publish_parameter(self):
        return self._ps_publish_tracker.track_increment()

    def publish_parameter(self, iteration, message=''):
        """
        Learner publishes latest parameters to the parameter server.

        Args:
            iteration: the current number of learning iterations
            message: optional message, must be pickleable.
        """
        self._ps_publisher.publish(iteration, message=message)

    ######
    # Getting data
    ######

    ######
    # Logging
    ######
    def _setup_logging(self):
        self.learn_timer = U.TimeRecorder()
        self.iter_timer = U.TimeRecorder()
        self.publish_timer = U.TimeRecorder()

        self.init_time = time.time()
        self.current_iter = 0

        self.last_time = self.init_time
        self.last_time_2 = self.init_time
        self.last_iter = 0

        self.log = get_loggerplex_client('learner', self.session_config)
        self.tensorplex = self._get_tensorplex('learner/learner')

        self._tensorplex_thread = U.PeriodicWakeUpWorker(
            target=self.generate_tensorplex_report)

    def _get_tensorplex(self, name):
        """
            Get the periodic tensorplex object
        Args:
            @name: The name of the collection of metrics
        """
        tp = get_tensorplex_client(
            name,
            self.session_config
        )
        update_schedule = self.session_config.tensorplex.update_schedule
        periodic_tp = TimeThrottledTensorplex(
            tensorplex=tp,
            min_update_interval=update_schedule.learner_min_update_interval,
        )
        return periodic_tp

    # def start_tensorplex_thread(self):
    #     if self._tensorplex_thread is not None:
    #         raise RuntimeError('tensorplex thread already running')
    #     self._tensorplex_thread = U.PeriodicWakeUpWorker(target=self.generate_tensorplex_report)
    #     self._tensorplex_thread.start()
    #     return self._tensorplex_thread

    def generate_tensorplex_report(self):
        """
            Adds core and system level tensorplex stats
        """
        cur_time = time.time()
        current_iter = self.current_iter

        iter_elapsed = current_iter - self.last_iter
        self.last_iter = current_iter

        time_elapsed = cur_time - self.last_time
        self.last_time = cur_time

        core_metrics = {}
        system_metrics = {}

        learn_time = self.learn_timer.avg + 1e-6
        iter_time = self.iter_timer.avg + 1e-6
        publish_time = self.publish_timer.avg + 1e-6
        # Time it takes to learn from a batch
        core_metrics['learn_time_s'] = learn_time
        # Time it takes to publish parameters
        core_metrics['publish_time_s'] = publish_time
        # Time it takes to complete one full iteration
        core_metrics['iter_time_s'] = iter_time

        iter_per_s = iter_elapsed / time_elapsed
        # Number of iterations per second
        system_metrics['iter_per_s'] = iter_per_s
        # Number of experience bathces processed per second
        system_metrics['exp_per_s'] = iter_per_s * \
            self.learner_config.replay.batch_size
        # Percent of time spent on learning
        system_metrics['compute_load_percent'] = min(
            learn_time / iter_time * 100, 100)
        # Percent of time spent on publishing
        system_metrics['io_publish_load_percent'] = min(
            publish_time / iter_time * 100, 100)

        all_metrics = {}
        for k in core_metrics:
            all_metrics['.core/' + k] = core_metrics[k]
        for k in system_metrics:
            all_metrics['.system/' + k] = system_metrics[k]

        # These are system metrics,
        # they don't add to counter or trigger updates
        self.tensorplex.add_scalars(all_metrics)

    ######
    # Checkpoint
    ######
    def _setup_checkpoint(self):
        # Setup saving
        tracked_attrs = self.checkpoint_attributes()
        assert U.is_sequence(tracked_attrs), \
            'checkpoint_attributes must return a list of string attr names'
        self._periodic_checkpoint = U.PeriodicCheckpoint(
            U.f_join(self.session_config.folder, 'checkpoint'),
            name='learner',
            period=self.session_config.checkpoint.learner.periodic,
            min_interval=self.session_config.checkpoint.learner.min_interval,
            tracked_obj=self,
            tracked_attrs=tracked_attrs,
            keep_history=self.session_config.checkpoint.learner.keep_history,
            keep_best=self.session_config.checkpoint.learner.keep_best,
            # TODO figure out how to add score to learner
        )

    def periodic_checkpoint(self, global_steps, score=None, **info):
        """
        Will only save at the end of each period

        Args:
            global_steps: the number of iterations
            score: the evaluation score for saving the best parameters.
                Currently Not supported!!!
                None when session_config.checkpoint.keep_best=False
            **info: other metadata to save in checkpoint

        Returns:
            saved(bool): whether save() is actually called or not
        """
        return self._periodic_checkpoint.save(
            score=score,
            global_steps=global_steps,
            reload_metadata=False,
            **info,
        )

    def restore_checkpoint(self):
        SC = self.session_config
        restore_folder = SC.checkpoint.restore_folder
        if (restore_folder and
                U.f_last_part_in_path(restore_folder) != 'checkpoint'):
            # automatically append 'checkpoint' subfolder
            restore_folder = U.f_join(restore_folder, 'checkpoint')
        restored = self._periodic_checkpoint.restore(
            target=SC.checkpoint.learner.restore_target,
            mode=SC.checkpoint.learner.mode,
            reload_metadata=True,
            check_ckpt_exists=True,  # TODO set to False unless debugging
            restore_folder=restore_folder,
        )
        if restored:
            self.log.info('successfully restored from checkpoint', restored)

    ######
    # Main Loop
    # Override to completely change learner behavior
    ######
    def main(self):
        """
            Main function that defines learner process
        """
        self.main_setup()
        self.intervals = intervals = [10, 60, 300]
        # import collections
        # self.time_records = collections.deque(maxlen=max(self.intervals))
        cnt = 0
        self.total_env_frames = 0
        last_env_frames = [0 for _ in range(len(self.intervals))]
        cur = [time.time_ns() for _ in range(len(self.intervals))]
        while True:
            self.main_loop()
            cnt += 1
            for j, interval in enumerate(intervals):
                if time.time_ns() - cur[j] >= interval * 1e9:
                    print(f"FPS ({interval}s): {(self.total_env_frames - last_env_frames[j]) * 1e9 / ((time.time_ns() - cur[j]))}")
                    last_env_frames[j] = self.total_env_frames
                    cur[j] = time.time_ns()

    def main_setup(self):
        """
            Setup before constant looping
        """
        self.save_config()
        self.iter_timer.start()
        self.publish_parameter(0, message='batch '+str(0))

    def main_loop(self):
        """
            One loop of learner, runs one learn operation of learner
        """
        # datas = []
        # for rank, q in enumerate(self._prefetch_queues):
        #     data = q.get()
        #     print(f"get data {rank}")
        #     datas.append(data)
        # sample_cnt = 0
        with self.learn_timer.time():
            sample_cnt = self.learn()
        if self.should_publish_parameter():
            with self.publish_timer.time():
                self.publish_parameter(self.current_iter,
                                       message='batch '+str(self.current_iter))
        if sample_cnt > 0:
            self.iter_timer.lap()
            self.current_iter += 1
        self.total_env_frames += sample_cnt

    def save_config(self):
        """
            Save config into a yaml file in root experiment directory
        """
        folder = Path(self.session_config.folder)
        folder.mkdir(exist_ok=True, parents=True)
        config = Config(
            learner_config=self.learner_config,
            env_config=self.env_config,
            session_config=self.session_config
        )
        config.dump_file(str(folder / 'config.yml'))
