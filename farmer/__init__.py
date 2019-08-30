from flask import Flask
from farmer.ImageAnalyzer.config import Config

app = Flask(__name__)
app.config.from_object(Config)

from farmer import view

