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
        for key, value in stats.items():
            prefix = "" if tag is None else tag + '/'
            full_key = prefix + key
            if torch.is_tensor(value):
                value = value.item()
            self.writer.add_scalar(full_key, value, step)
