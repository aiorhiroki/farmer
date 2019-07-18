from farmer import app
from farmer.ImageAnalyzer import fit
from flask import request, make_response


@app.route('/train', methods=["POST"])
def train():
    form = request.json
    fit.train(form)
    return make_response('', 202)
