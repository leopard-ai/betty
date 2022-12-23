from dataclasses import dataclass


@dataclass
class EngineConfig:
    """
    Configuration for ``Engine``.
    """

    train_iters: int = 50000
    valid_step: int = 500

    # logger
    logger_type: str = "none"

    # roll back
    roll_back: bool = False

    # distributed training
    distributed: bool = False
    backend: str = "nccl"
    strategy: str = "default"

    # early stopping
    early_stopping: bool = False
    early_stopping_mode: str = "min"
    early_stopping_tolerance: int = 5
    early_stopping_metric: str = "loss"
