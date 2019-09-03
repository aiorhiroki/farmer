import json
from configparser import ConfigParser

from farmer.ImageAnalyzer import fit
from farmer.ImageAnalyzer.task import Task
from farmer.ImageAnalyzer.model import Trainer


def classification():
    config = _read_config(Task.CLASSIFICATION)
    fit.train(config)


def segmentation():
    config = _read_config(Task.SEMANTIC_SEGMENTATION)
    fit.train(config)


def _read_config(task_id):
    parser = ConfigParser()
    parser.read('config.ini')
    config = json(parser['project_settings'])
    config['task_id'] = task_id.value

    secret_parser = ConfigParser()
    secret_parser.read('secret.ini')
    if len(secret_parser.sections()) > 0:
        secret_config = json(secret_parser['default'])
        config.update(secret_config)

    trainer = Trainer(**config)

    return trainer
