from dataclasses import dataclass


@dataclass
class Config:
    type: str = "darts"
    step: int = 1
    first_order: bool = True
    retain_graph: bool = False
    allow_unused: bool = True

    # gradient accumulation
    gradient_accumulation: int = 1

    # fp16 training
    fp16: bool = False
    dynamic_loss_scale: bool = False
    initial_dynamic_scale: int = 2**32
    static_loss_scale: float = 1.0

    # logging
    log_step: int = -1
    log_local_step: bool = False

    # darts
    darts_alpha: float = 0.01

    # neumann
    neumann_iterations: int = 1
    neumann_alpha: float = 1.0

    # cg
    cg_iterations: int = 1
    cg_alpha: float = 1.0


@dataclass
class EngineConfig:
    train_iters: int = 50000
    valid_step: int = 500

    logger_type: str = "tensorboard"
