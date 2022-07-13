import socket

import torch

from betty.logging.logger_base import LoggerBase


try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class WandBLogger(LoggerBase):
    def __init__(self):
        def init(self):
            try:
                wandb.init(project="betty", entity=socket.gethostname())
            except:
                wandb.init(project="betty", reinit=True)

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
        if not HAS_WANDB:
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
                    wandb.log({full_key: value, "global_step": step})
            else:
                value = values
                full_key = key_extended
                if torch.is_tensor(value):
                    value = value.item()
                wandb.log({full_key: value, "global_step": step})
