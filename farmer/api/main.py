from configparser import ConfigParser

from farmer.domain.model.trainer_model import Trainer
from farmer.domain.workflows.train_workflow import TrainWorkflow
from farmer.domain.model.task_model import Task


def classification():
    trainer = _read_config(Task.CLASSIFICATION)
    TrainWorkflow(trainer).command()


def segmentation():
    trainer = _read_config(Task.SEMANTIC_SEGMENTATION)
    TrainWorkflow(trainer).command()


def _read_config(task_id):
    parser = ConfigParser()
    parser.read('config.ini')
    config = parser.defaults()
    config['task'] = task_id.value

    secret_parser = ConfigParser()
    secret_parser.read('secret.ini')
    if len(secret_parser.sections()) > 0:
        secret_config = secret_parser.defaults()
        config.update(secret_config)

    trainer = Trainer(**config)

    return trainer
