import time
import os

import numpy as np
import torch

from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        algo,
        epoch,
        step_per_epoch,
        rollout_freq,
        logger,
        log_freq,
        eval_episodes=10
    ):
        self.algo = algo

        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._rollout_freq = rollout_freq

        self.logger = logger
        self._log_freq = log_freq
        self._eval_episodes = eval_episodes

    def train_dynamics(self):
        start_time = time.time()
        self.algo.learn_dynamics()
        #self.algo.save_dynamics_model(
            #save_path=os.path.join(self.logger.writer.get_logdir(), "dynamics_model")
        #)
        self.algo.save_dynamics_model("dynamics_model")
        self.logger.print("total time: {:.3f}s".format(time.time() - start_time))

