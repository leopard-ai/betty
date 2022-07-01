import atexit
import os
import socket
from datetime import datetime

import torch

try:
    from torch.utils.tensorboard import SummaryWriter

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

from betty.logging.logger_base import LoggerBase


class TensorBoardLogger(LoggerBase):
    def __init__(self, comment=""):
        atexit.register(self.close)
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        log_dir = os.path.join(
            "betty_tensorboard", current_time + "_" + socket.gethostname() + comment
        )
        self.writer = SummaryWriter(log_dir=log_dir)

    def close(self):
        """
        Close PyTorch's tensorboard ``SummaryWriter``.
        """
        self.writer.close()

    def log(self, stats, tag=None, step=None):
        """
        Log metrics/stats to PyTorch tensorboard.

        :param stats: Dictoinary of values and their names to be recorded
        :type stats: dict
        :param tag:  Data identifier
        :type tag: str, optional
        :param step: step value associated with ``stats`` to record
        :type step: int, optional
        """
        if not HAS_TENSORBOARD:
            return
        if stats is None:
            return
        for key, values in stats.items():
            prefix = "" if tag is None else tag + "/"
            key_extended = prefix + key
            if isinstance(values, tuple) or isinstance(values, list):
                for value_idx, value in enumerate(values):
                    full_key = key_extended + "_" + str(value_idx)
                    if torch.is_tensor(value):
                        value = value.item()
                    self.writer.add_scalar(full_key, value, step)
            else:
                value = values
                full_key = key_extended
                if torch.is_tensor(value):
                    value = value.item()
                self.writer.add_scalar(full_key, value, step)
