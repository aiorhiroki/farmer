from configparser import ConfigParser
from farmer.ImageAnalyzer import fit
from .task import Task


def classification():
    config = _read_config(Task.CLASSIFICATION)
    fit.train(config)


def segmentation():
    config = _read_config(Task.SEMANTIC_SEGMENTATION)
    fit.train(config)


def _read_config(task_id):
    parser = ConfigParser()
    parser.read('config.ini')
    parser['project_settings']['task_id'] = task_id

    return parser
