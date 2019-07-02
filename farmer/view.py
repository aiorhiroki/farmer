from farmer import app
from farmer.ImageAnalyzer import fit
import request


@app.route('/train', methods=["POST"])
def train():
    form = request.json
    fit.train(form)
