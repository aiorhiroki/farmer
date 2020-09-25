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
