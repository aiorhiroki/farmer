from dataclasses import dataclass


@dataclass
class Config:
    slack_token: str = None
    slack_channel: str = None
    milk_api_url: str = "http://127.0.0.1:5000/"
    root_dir: str = "result"
    result_dir: str = None
    result_path: str = None
    info_dir: str = "info"
    model_dir: str = "model"
    learning_dir: str = "learning"
    image_dir: str = "image"
    info_path: str = None
    model_path: str = None
    learning_path: str = None
    image_path: str = None
    return_result: bool = False
