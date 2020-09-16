from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class TrainParams:
    # 最適化のために探索するパラメータ
    model_name: str = None
    backbone: str = None
    activation: str = "softmax"
    loss: str = None
    loss_params: Dict[str, float] = field(default_factory=dict)
    batch_size: int = None
    cosine_decay: bool = False
    cosine_lr_max: int = 0.01
    cosine_lr_min: int = 0.001
    weights_info: Dict[str, str] = field(default_factory=dict)
    class_weights: Dict[int, float] = field(default_factory=dict)
    learning_rate: float = None
    optimizer: str = None
    augmentation: List[str] = field(default_factory=list)
    opt_weight_decay: float = 0.0
