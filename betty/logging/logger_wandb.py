import socket

import torch

from betty.logging.logger_base import LoggerBase


try:
    import wandb
except ImportError:
    wandb = None


class WandBLogger(LoggerBase):
    def __init__(self):
        wandb.init(project="betty", entity=socket.gethostname())

    def log(self, stats, tag=None, step=None):
        """
        Log metrics/stats to Weight & Biases (wandb) logger

        :param stats: Dictoinary of values and their names to be recorded
        :type stats: dict
        :param tag:  Data identifier
        :type tag: str, optional
        :param step: step value associated with ``stats`` to record
        :type step: int, optional
        """
        if stats is None:
            return
        for key, value in stats.items():
            prefix = "" if tag is None else tag + "/"
            full_key = prefix + key
            if torch.is_tensor(value):
                value = value.item()
            wandb.log({full_key: value})
