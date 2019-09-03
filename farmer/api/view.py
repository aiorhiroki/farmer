from farmer import app
from flask import request, make_response, jsonify
from keras.models import load_model
import cv2
import os
import numpy as np
import shutil

from farmer.domain.model.trainer_model import Trainer
from farmer.domain.workflows.train_workflow import TrainWorkflow
from farmer.domain.workflows.test_workflow import TestWorkflow


@app.route('/train', methods=["POST"])
def train():
    form = request.json
    form = {k: v for (k, v) in form.items() if v}
    trainer = Trainer(**form)
    TrainWorkflow(trainer).command()
    return make_response(jsonify(dict()), 200)


@app.route('/predict', methods=["POST"])
def predict():
    img_file = request.files['image']
    file_data = img_file.stream.read()
    nparr = np.fromstring(file_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    result_dir = request.files['result_dir'].stream.read()
    model_path = os.path.join(
        result_dir.decode('utf-8'),
        'model',
        'best_model.h5'
    )
    model = load_model(model_path)
    input_img = np.expand_dims(img, axis=0)/255
    predictions = model.predict(input_img)[0]
    predictions = [float(prediction) for prediction in predictions]
    return make_response(jsonify(dict(prediction=predictions)))


@app.route('/test', methods=["POST"])
def test():
    form = request.json
    form = {k: v for (k, v) in form.items() if v}
    model_path = os.path.join(
        form["result_dir"],
        'model',
        'best_model.h5'
    )
    form['model_path'] = model_path
    report = TestWorkflow(form).command()
    return make_response(jsonify(report))


@app.route('/delete_model', methods=["POST"])
def delete_model():
    form = request.json
    shutil.rmtree(form.get('result_dir'))
    return make_response('', 202)
