from dataclasses import dataclass
from dataclasses import field
from typing import List, Dict


@dataclass
class LRScheduler:
    scheduler_name: str = None
    cosine_decay: bool = False
    cosine_lr_max: float = 0.01
    cosine_lr_min: float = 0.001
    scheduler_base_lr: float = 0.001
    step_lr: bool = False
    step_size: int = 20
    step_gamma: float = 0.5
    multi_step_lr: bool = False
    milestones: List[int] = field(default_factory=list)
    exponential_lr: bool = False
    exp_gamma: float = 0.95
    cyclical_lr: bool = False
    cyc_lr_max: float = 0.01
    cyc_lr_min: float = 0.001
