# Copyright Sang Keun Choe
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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

    # fp16 training
    fp16: bool = False
    initial_dynamic_scale: float = 4096.0
    scale_factor: float = 2.0

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
