from flask import Flask
from farmer.domain.model.config_model import Config

app = Flask(__name__)
app.config.from_object(Config)

from farmer.api import view

