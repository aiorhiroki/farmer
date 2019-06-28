from flask import Flask

app = Flask(__name__)

from farmer import view

