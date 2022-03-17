from dataclasses import dataclass


@dataclass
class Config:
    type: str = 'maml'
    step: int = 2
    first_order: bool = False
    retain_graph: bool = False
    allow_unused: bool = True
