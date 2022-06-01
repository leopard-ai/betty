import atexit
import os
import socket
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from betty.logging.logger_base import LoggerBase


class TensorBoardLogger(LoggerBase):
    def __init__(self, comment=''):
        atexit.register(self.close)
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join(
            'betty_tensorboard', current_time + '_' + socket.gethostname() + comment
        )
        self.writer = SummaryWriter(log_dir=log_dir)

    def close(self):
        self.writer.close()

    def log(self, stats, tag=None, step=None):
        if stats is None:
            return
        for key, values in stats.items():
            prefix = "" if tag is None else tag + '/'
            key_extended = prefix + key
            if isinstance(values, tuple) or isinstance(values, list):
                for value_idx, value in enumerate(values):
                    full_key = key_extended + '_' + str(value_idx)
                    if torch.is_tensor(value):
                        value = value.item()
                    self.writer.add_scalar(full_key, value, step)
