from dataclasses import dataclass


@dataclass
class Config:
    type: str = 'darts'
    step: int = 5
    first_order: bool = True
    retain_graph: bool = False
    allow_unused: bool = True

    # darts
    darts_alpha: float = 0.01

    # neumann
    neumann_iterations: int = 1
    neumann_alpha: float = 1.

    # cg
    cg_iterations: int = 1
    cg_alpha: float = 1.


@dataclass
class EngineConfig:
    train_iters: int = 50000
    valid_step: int = 500
