from configparser import ConfigParser

from apps.domain.model.trainer_model import Trainer
from apps.domain.workflows.train_workflow import TrainWorkflow
from apps.domain.model.task_model import Task


def classification():
    trainer = _read_config(Task.CLASSIFICATION)
    TrainWorkflow(trainer).command()


def segmentation():
    trainer = _read_config(Task.SEMANTIC_SEGMENTATION)
    TrainWorkflow(trainer).command()


def _read_config(task_id):
    parser = ConfigParser()

    if task_id == Task.CLASSIFICATION:
        parser.read("classification-config.ini")
    elif task_id == Task.SEMANTIC_SEGMENTATION:
        parser.read("segmentation-config.ini")
    config = parser.defaults()
    config["task"] = task_id.value

    secret_parser = ConfigParser()
    secret_parser.read("secret.ini")
    if len(secret_parser.defaults()) > 0:
        secret_config = secret_parser.defaults()
        config.update(secret_config)

    trainer = Trainer(**config)

    return trainer
