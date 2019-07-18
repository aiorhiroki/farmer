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
    config = parser['project_settings']

    default_config = parser['default']
    for content in default_config:
        config[content] = default_config[content]

    if task_id == Task.CLASSIFICATION:
        task = 'classificartion'
    elif task_id == Task.SEMANTIC_SEGMENTATION:
        task = 'segmentation'
    else:
        raise NotImplementedError
    task_config = parser[f'{task}_default']
    for content in task_config:
        config[content] = task_config[content]

    config['task_id'] = task_id

    return config
