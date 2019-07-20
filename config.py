import os


class Config(object):
    MILK_API_URL = 'http://127.0.0.1:5000/'
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULT_DIR = os.path.join(ROOT_DIR, 'result')
