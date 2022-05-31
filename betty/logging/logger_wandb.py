import atexit

import torch
from torch.utils.tensorboard import SummaryWriter

from betty.logging.logger_base import LoggerBase


class WandBLogger(LoggerBase):
    def __init__(self):
        self.writer = SummaryWriter()


    def log(self, stats, tag=None, step=None):
        for key, value in stats.items():
            prefix = "" if tag is None else tag + '/'
            full_key = prefix + key
            if torch.is_tensor(value):
                value = value.item()
            self.writer.add_scalar(full_key, value, step)
