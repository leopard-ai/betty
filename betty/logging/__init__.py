from .logger_tensorboard import TensorBoardLogger
from .logger_wandb import WandBLogger


logger_mapping = {
    'tensorboard': TensorBoardLogger,
    'wandb': WandBLogger
}

def type_check(logger_type):
    assert logger_type in logger_mapping

    if logger_type == 'wandb':
        try:
            import wandb
        except ImportError:
            logger_type = 'tensorboard'
    return logger_type

def logger(logger_type='tensorboard'):
    logger_type = type_check(logger_type)
    return logger_mapping[logger_type]()
