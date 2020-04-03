import yaml
import os
from glob import glob

from farmer.ncc.utils import cross_val_split
from farmer.domain.model.task_model import Task
from farmer.domain.model.trainer_model import Trainer
from farmer.domain.workflows.train_workflow import TrainWorkflow


def fit():
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
                print(f"cross validation step: {k}")
                trainer.val_dirs = cross_val_dirs[k]
                trainer.train_dirs = list()
                for val_i in range(n_splits):
                    if val_i == k:
                        continue
                    trainer.train_dirs += cross_val_dirs[val_i]
                # cross validation folder path
                k_result = result_path + f"/cv_{k}"
                trainer.result_path = k_result
                trainer.info_path = f"{k_result}/{trainer.info_dir}"
                trainer.model_path = f"{k_result}/{trainer.model_dir}"
                trainer.learning_path = f"{k_result}/{trainer.learning_dir}"
                trainer.image_path = f"{k_result}/{trainer.image_dir}"

                train_workflow = TrainWorkflow(trainer)
                if trainer.optuna:
                    train_workflow.optuna_command()
                else:
                    train_workflow.command()
        else:
            train_workflow = TrainWorkflow(trainer)
            if trainer.optuna:
                train_workflow.optuna_command()
            else:
                train_workflow.command()
