import yaml
from collections import OrderedDict
import os
from glob import glob
import optuna
import numpy as np
from tensorflow.keras.backend import clear_session
import logging
from copy import deepcopy

from farmer.ncc.utils import cross_val_split, limit_train_dirs
from farmer.domain.model.task_model import Task
from farmer.domain.model import Trainer
from farmer.domain.workflows.train_workflow import TrainWorkflow


def fit():
    yaml.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        lambda loader,
        node: OrderedDict(loader.construct_pairs(node))
    )

    with open("run.yaml") as yamlfile:
        run_config = yaml.safe_load(yamlfile)
    config_paths = run_config.get("config_paths")
    if os.path.exists("secret.yaml"):
        with open("secret.yaml") as yamlfile:
            secret_config = yaml.safe_load(yamlfile)
    else:
        secret_config = None
    for config_path in config_paths:
        print("config path running: ", config_path)
        with open(config_path) as yamlfile:
            config = yaml.safe_load(yamlfile)
        config.update(
            {k: v for (k, v) in run_config.items() if k != "config_paths"}
        )
        config.update(dict(config_path=config_path))
        if secret_config:
            config.update(secret_config)
        trainer = Trainer(**config)
        val_dirs = trainer.val_dirs
        if trainer.training and (val_dirs is None or len(val_dirs) == 0):
            # cross validation
            if trainer.task == Task.SEMANTIC_SEGMENTATION:
                image_dir = trainer.label_dir
            else:
                image_dir = "*"
            train_counts = list()
            for train_dir in trainer.train_dirs:
                image_dir_path = os.path.join(
                    trainer.target_dir,
                    train_dir,
                    image_dir
                )
                train_counts.append(
                    len(glob(f"{image_dir_path}/*.png")) +
                    len(glob(f"{image_dir_path}/*.jpg"))
                )
            n_splits = trainer.n_splits
            cross_val_dirs = cross_val_split(
                trainer.train_dirs, train_counts, k=n_splits
            )
            print("cross validation dirs: ", cross_val_dirs)
            result_path = trainer.result_path
            for k in range(n_splits):
                trainer = Trainer(**deepcopy(config))
                if type(trainer.cross_val) is int and k != trainer.cross_val:
                    continue
                print(f"cross validation step: {k}")
                trainer.train_dirs = list()
                trainer.val_dirs = list()
                if trainer.cross_val == "all":
                    trainer.val_dirs = cross_val_dirs[k]
                    for val_i in range(n_splits):
                        if val_i == k:
                            continue
                        trainer.train_dirs += cross_val_dirs[val_i]
                elif trainer.cross_val == "step":
                    if k >= (n_splits - 1):
                        continue
                    for val_i in range(n_splits):
                        if val_i <= k:
                            trainer.val_dirs += cross_val_dirs[val_i]
                        else:
                            trainer.train_dirs += cross_val_dirs[val_i]
                elif type(trainer.cross_val) is int:
                    for val_i in range(n_splits):
                        if val_i == k:
                            trainer.val_dirs += cross_val_dirs[k]
                        else:
                            trainer.train_dirs += cross_val_dirs[val_i]
                # cross validation folder path
                k_result = result_path + f"/cv_{k}"
                trainer.result_path = k_result
                trainer.info_path = f"{k_result}/{trainer.info_dir}"
                trainer.model_path = f"{k_result}/{trainer.model_dir}"
                trainer.learning_path = f"{k_result}/{trainer.learning_dir}"
                trainer.image_path = f"{k_result}/{trainer.image_dir}"

                if trainer.optuna:
                    optuna_command(trainer)

                else:
                    train_workflow = TrainWorkflow(trainer)
                    train_workflow.command()
        
        elif trainer.train_count:
            # limit size of train data
            for item in trainer.train_count:
                if type(item) is int: train_count_len = 1
                elif type(item) is list:
                    train_count_len = len(item)
                    break
            if train_count_len == 1:
                count_list = trainer.train_count
                trainer.train_dirs = limit_train_dirs(trainer, count_list)
                # start training
                train_workflow = TrainWorkflow(trainer)
                train_workflow.command()
            else:
                for i, count in enumerate(trainer.train_count):
                    if type(count) is int:
                        trainer.train_count[i] = [count] * train_count_len
                for k, count_list in enumerate(zip(*trainer.train_count)):
                    trainer = Trainer(**deepcopy(config))
                    print(f"train count step: {k}")
                    trainer.train_dirs = limit_train_dirs(trainer, count_list)
                    # separate folder_path
                    k_result = trainer.result_path + f"/count_{k}"
                    trainer.result_path = k_result
                    trainer.info_path = f"{k_result}/{trainer.info_dir}"
                    trainer.model_path = f"{k_result}/{trainer.model_dir}"
                    trainer.learning_path = f"{k_result}/{trainer.learning_dir}"
                    trainer.image_path = f"{k_result}/{trainer.image_dir}"
                    # start training
                    train_workflow = TrainWorkflow(trainer)
                    train_workflow.command()                   

        else:
            if trainer.optuna:
                optuna_command(trainer)

            else:
                train_workflow = TrainWorkflow(trainer)
                train_workflow.command()


class Objective(object):
    def __init__(self, trainer):
        self.trainer = trainer

    def __call__(self, trial):
        clear_session()
        train_workflow = TrainWorkflow(self.trainer, trial)
        result = train_workflow.command(trial)

        if self.trainer.task == Task.CLASSIFICATION:
            return result["accuracy"]
        elif self.trainer.task == Task.SEMANTIC_SEGMENTATION:
            return np.mean(result["dice"][1:])
        else:
            raise NotImplementedError


def optuna_report(study):
    pruned = optuna.structs.TrialState.PRUNED
    complete = optuna.structs.TrialState.COMPLETE
    pruned_trials = [
        t for t in study.trials if t.state == pruned]
    complete_trials = [
        t for t in study.trials if t.state == complete]

    print("Study statistics: ")
    print(" Number of finished trials: ", len(study.trials))
    print(" Number of pruned trials: ", len(pruned_trials))
    print(" Number of complete trials: ", len(complete_trials))

    print('Best trial:')
    trial = study.best_trial
    print('  Value: {}'.format(trial.value))
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))


def optuna_command(trainer):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Setup the root logger.
    # TODO: 任意のパスを設定したら保存されない問題．infoフォルダにいれたい．
    logger.addHandler(logging.FileHandler(
        "optuna_trials.log", mode="w")
    )
    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    optuna.logging.enable_default_handler()  # Stop showing logs in sys.stderr.

    pruner_params = trainer.pruner_params
    if pruner_params is None:
        pruner = getattr(optuna.pruners, trainer.pruner)(
                     n_startup_trials=3, n_warmup_steps=10, interval_steps=1)
    else:
        pruner = getattr(optuna.pruners, trainer.pruner)(**pruner_params)
    study = optuna.create_study(
        storage=f"sqlite:///optuna_study.db",
        load_if_exists=True,
        study_name=trainer.result_path,
        direction='maximize',
        pruner=pruner
    )

    study.optimize(
        Objective(trainer),
        n_trials=trainer.n_trials,
        timeout=trainer.timeout
    )

    optuna_report(study)
