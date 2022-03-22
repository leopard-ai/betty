from dataclasses import dataclass


@dataclass
class Config:
    type: str = 'darts'
    step: int = 5
    first_order: bool = False
    retain_graph: bool = False
    allow_unused: bool = True

    # darts
    darts_alpha = 0.01

    # neumann
    neumann_iterations: int = 3
    neumann_alpha: float = 1.

    # cg
    cg_iterations: int = 3
    cg_alpha: float = 1.
