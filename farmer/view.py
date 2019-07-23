from farmer import app
from farmer.ImageAnalyzer import fit
from flask import request, make_response
import shutil


@app.route('/train', methods=["POST"])
def train():
    form = request.json
    fit.train(form)
    return make_response('', 202)


@app.route('/delete_model', methods=["POST"])
def delete_model():
    form = request.json
    shutil.rmtree(form.get('result_dir'))
    return make_response('', 202)
