from dataclasses import dataclass


@dataclass
class Config:
    """
    Training configuration for ``Problem``.
    """

    type: str = "darts"
    unroll_steps: int = 1
    first_order: bool = True
    retain_graph: bool = False
    allow_unused: bool = True

    # gradient accumulation
    gradient_accumulation: int = 1

    # gradient clipping
    gradient_clipping: float = 0.0

    # fp16 training
    fp16: bool = False
    initial_dynamic_scale: float = 4096.0
    scale_factor: float = 2.0

    # warm-up
    warmup_steps: int = 0

    # logging
    log_step: int = -1
    log_local_step: bool = False

    # darts
    darts_alpha: float = 0.01
    darts_preconditioned: bool = True

    # neumann
    neumann_iterations: int = 1
    neumann_alpha: float = 1.0

    # cg
    cg_iterations: int = 1
    cg_alpha: float = 1.0
