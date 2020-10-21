from optuna.pruners import (
    MedianPruner, PercentilePruner, SuccessiveHalvingPruner, HyperbandPruner, ThresholdPruner
)

def median_pruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1):
    pruner = MedianPruner(
        n_startup_trials=n_startup_trials,
        n_warmup_steps=n_warmup_steps,
        interval_steps=interval_steps
    )
    return pruner


def percentile_pruner(percentile=25.0, n_startup_trials=5, n_warmup_steps=0, interval_steps=1):
    pruner = PercentilePruner(
        percentile=percentile,
        n_startup_trials=n_startup_trials,
        n_warmup_steps=n_warmup_steps,
        interval_steps=interval_steps
    )
    return pruner


def successive_halving_pruner(min_resource='auto', reduction_factor=4, min_early_stopping_rate=0):
    pruner = SuccessiveHalvingPruner(
        min_resource=min_resource,
        reduction_factor=reduction_factor,
        min_early_stopping_rate=min_early_stopping_rate
    )
    return pruner


def hyperband_pruner(min_resource=1, max_resource='auto', reduction_factor=3):
    pruner = HyperbandPruner(
        min_resource=min_resource,
        max_resource=max_resource,
        reduction_factor=reduction_factor
    )
    return pruner


def threshold_pruner(lower=0.0, upper=1.0, n_warmup_steps=0, interval_steps=1):
    pruner = ThresholdPruner(
        lower=lower,
        upper=upper,
        n_warmup_steps=n_warmup_steps,
        interval_steps=interval_steps
    )
    return pruner
