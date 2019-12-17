from configparser import ConfigParser

from apps.domain.model.trainer_model import Trainer
from apps.domain.workflows.train_workflow import TrainWorkflow
from apps.domain.model.task_model import Task


def fit():
    run_file = ConfigParser()
    run_file.read("run.ini")
    run_config = run_file.defaults()
    gpu = run_config.get("gpu")
    config_paths = run_config.get("config_paths")

    secret_parser = ConfigParser()
    secret_parser.read("secret.ini")
    if len(secret_parser.defaults()) > 0:
        secret_config = secret_parser.defaults()

    parser = ConfigParser()
    for config_path in config_paths.split():
        parser.read(config_path)
        config = parser.defaults()
        config["task"] = task_id.value
        config["gpu"] = gpu
        config.update(secret_config)

        trainer = Trainer(**config)
        TrainWorkflow(trainer).command()
