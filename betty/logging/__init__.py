from .logger_tensorboard import TensorBoardLogger
from .logger_wandb import WandBLogger
from .logger_base import get_logger


logger_mapping = {"tensorboard": TensorBoardLogger, "wandb": WandBLogger}


def type_check(logger_type):
    assert logger_type in logger_mapping

    if logger_type == "wandb":
        try:
            get_logger().warning("[!] WandB is not installed. Tensorboard will be instead used.")
            import wandb
        except ImportError:

            logger_type = "tensorboard"
    return logger_type


def logger(logger_type="tensorboard"):
    logger_type = type_check(logger_type)
    return logger_mapping[logger_type]()
