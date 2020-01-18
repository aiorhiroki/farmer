import yaml
import os

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
        TrainWorkflow(trainer).command()
