import socket

import torch

from betty.logging.logger_base import LoggerBase


try:
    import wandb
except ImportError:
    wandb = None


class WandBLogger(LoggerBase):
    def __init__(self):
        wandb.init(project='betty', entity=socket.gethostname())

    def log(self, stats, tag=None, step=None):
        for key, value in stats.items():
            prefix = "" if tag is None else tag + '/'
            full_key = prefix + key
            if torch.is_tensor(value):
                value = value.item()
            wandb.log({full_key: value})
