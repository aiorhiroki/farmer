from dataclasses import dataclass
from typing import List, Dict


@dataclass
class LRScheduler:
    scheduler_name: str = None
    cosine_lr_max: float = 0.01
    cosine_lr_min: float = 0.001
    step_size: int = 20
    step_gamma: float = 0.5
    milestones: dict = None
    exp_gamma: float = 0.95
    cyc_lr_max: float = 0.01
    cyc_lr_min: float = 0.001

class StepLR:
    step_size: int = None
    step_gamma: float = None

class MultiStepLR:
    step_gamma: float = 0.5
    milestones: dict = None

class ExponentialLR:
    exp_gamma: float = 0.90

class CyclicalLR:
    step_size: int = None
    cyc_lr_max: float = None
    cyc_lr_min: float = None

class CosineDecay:
    cosine_lr_max: float = 0.001
    cosine_lr_min: float = 0.0001
