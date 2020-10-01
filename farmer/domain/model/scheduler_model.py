from dataclasses import dataclass
from typing import List, Dict


# @dataclass
# class LRScheduler:
#     step_lr: StepLR = None
#     multi_step_lr: MultiStepLR = None
#     exponential_lr: ExponentialLR = None
#     cyclical_lr: CyclicalLR = None
#     cosine_decay: CosineDecay = None

#     def __post_init__(self, params):


@dataclass
class StepLR:
    step_size: int = None
    step_gamma: float = 0.1

@dataclass
class MultiStepLR:
    step_gamma: float = 0.5
    milestones: dict = None

@dataclass
class ExponentialLR:
    exp_gamma: float = 0.90

@dataclass
class CyclicalLR:
    step_size: int = None
    cyc_lr_max: float = None
    cyc_lr_min: float = None

@dataclass
class CosineDecay:
    cosine_lr_max: float = 0.001
    cosine_lr_min: float = 0.0001
