from dataclasses import dataclass


@dataclass
class Optuna:
    op_batch_size: bool = False
    op_learning_rate: bool = False
    n_trials: int = 3
    timeout: int = 3*60*60
