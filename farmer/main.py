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
    if task_id == Task.CLASSIFICATION:
        task = 'classificartion'
    elif task_id == Task.SEMANTIC_SEGMENTATION:
        task = 'segmentation'
    else:
        raise NotImplementedError
    parser['training params'] = parser[f'{task}_default']
    parser['default']['task_id'] = task_id

    return parser
