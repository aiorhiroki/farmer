from dataclasses import dataclass


@dataclass
class Optuna:
    op_batch_size: bool = False
    op_learning_rate: bool = False