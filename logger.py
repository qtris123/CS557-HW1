"""
CS 557 RLI Homework Assignment 1
Purdue University
Created by: Joseph Campbell and Guven Gergerli
"""

import numpy as np
import os
import time
import re

from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, env_tag, env_name, action_dim, source_tag, num_params,
                 run_name=None, seed=0, output_root="."):
        """Configure TensorBoard logging."""

        if run_name is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
            raise ValueError("run_name must contain only letters, digits, underscores, dots or hyphens")
        label = run_name if run_name else f"{source_tag}_{time.time_ns()}"
        log_dir = os.path.join(output_root, 'runs', env_tag, f"{label}_seed{seed}")
        # Never silently merge two experiments into the same event directory.
        os.makedirs(log_dir, exist_ok=False)
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Log configuration information
        self.writer.add_text(
            'run/config', 
            f'env={env_name}, action_dim={action_dim}, source={source_tag}, params={num_params}'
        )

    def add_value(self, name, y, x):
        """Add a single scalar value to Tensorboard"""

        self.writer.add_scalar(name, y, x)
        self.writer.flush()

    def add_frames(self, name, frames, x, fps=10):
        """Add rendered frames from a list (should be list of lists) of episodes to Tensorboard"""

        video = np.array([np.transpose(frame, [0, 3, 1, 2]) for frame in frames])

        self.writer.add_video('{}'.format(name), video, x, fps=fps)
