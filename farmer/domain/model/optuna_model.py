from dataclasses import dataclass


@dataclass
class Optuna:
    op_batch_size: bool = False
    op_learning_rate: bool = False
    op_optimizer: bool = False
    op_backbone: bool = False
    n_trials: int = 3
    timeout: int = None
