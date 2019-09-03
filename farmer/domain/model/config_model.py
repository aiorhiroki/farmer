import os
from datetime import datetime
from dataclasses import dataclass


@dataclass
class Config:
    slack_token: str = None
    slack_channel: str = None
    milk_api_url: str = 'http://127.0.0.1:5000/'
    root_dir: str = 'result'
    result_dir: str = None
    result_path: str = None
    info_dir: str = 'info'
    return_result: bool = False

    def __post_init__(self):
        if self.result_dir is None:
            self.result_dir = datetime.today().strftime("%Y%m%d_%H%M")
        self.result_path = os.path.join(self.root_dir, self.result_dir)
