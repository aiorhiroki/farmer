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
    secret_config = None
    if len(secret_parser.defaults()) > 0:
        secret_config = secret_parser.defaults()

    parser = ConfigParser()
    for config_path in config_paths.split(","):
        print("config path running: ", config_path)
        parser.read(config_path)
        config = parser.defaults()
        if config_path.startswith('segmentation'):
            config["task"] = Task.SEMANTIC_SEGMENTATION.value
        elif config_path.startswith('classification'):
            config["task"] = Task.CLASSIFICATION.value
        else:
            continue
        config["gpu"] = gpu
        if secret_config:
            config.update(secret_config)
        trainer = Trainer(**config)
        TrainWorkflow(trainer).command()
