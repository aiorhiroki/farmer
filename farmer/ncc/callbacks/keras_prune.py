from typing import Dict
from typing import Optional

import optuna

with optuna._imports.try_import() as _imports:
    from tensorflow.keras.callbacks import Callback

if not _imports.is_successful():
    Callback = object  # NOQA


class KerasPruningCallback(Callback):
    def __init__(
        self,
        trial: optuna.trial.Trial,
        monitor: str,
        interval: int = 1
    ) -> None:
        super(KerasPruningCallback, self).__init__()

        _imports.check()

        self._trial = trial
        self._monitor = monitor
        self._interval = interval

    def on_epoch_end(
        self, epoch: int,
        logs: Optional[Dict[str, float]] = None
    ) -> None:

        if (epoch + 1) % self._interval != 0:
            return

        logs = logs or {}
        current_score = logs.get(self._monitor)
        if current_score is None:
            return
        self._trial.report(float(current_score), step=epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)
