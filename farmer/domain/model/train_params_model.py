from dataclasses import dataclass, field
from typing import Dict
# from .scheduler_model import LRScheduler
from . import scheduler_model


@dataclass
class TrainParams:
    # 最適化のために探索するパラメータ
    model_name: str = None
    backbone: str = None
    activation: str = "softmax"
    loss: Dict[str, dict] = field(default_factory=dict)
    classification_class_weight: Dict[str, float] = field(default_factory=dict)
    batch_size: int = None
    weights_info: Dict[str, str] = field(default_factory=dict)
    learning_rate: float = None
    optimizer: str = None
    augmentation: Dict[str, int] = field(default_factory=dict)
    opt_decay: float = 0.001
    scheduler: dict = None

    def __post_init__(self):
        if self.scheduler and isinstance(self.scheduler, dict):
            scheduler = self.scheduler['functions']
            scheduler_name = [k for k in scheduler.keys()][0]
            params = scheduler[scheduler_name]
            self.scheduler = getattr(
                scheduler_model, scheduler_name)(**params)
