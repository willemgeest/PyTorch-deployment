import io
from PIL import Image
from flask import Flask, jsonify, request
import beer_classification as bc
import object_detection as od
import numpy as np

#app = FastAPI()
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return {"message": "Welkom at FRITS!"}

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['file']
    img_bytes = io.BytesIO(file.read())
    img = Image.open(img_bytes)
    obj_det_model = od.get_obj_det_model()
    boxes, n_beers, _, preds = od.find_bottles(image=img,
                                               model=obj_det_model,
                                               detection_threshold=0.8,
                                               GPU=False)
    return {'boxes': boxes.tolist(),
            'n_beers': n_beers,
            'preds': preds.tolist()}


@app.route('/classify', methods=['POST'])
def classify():
    file = request.files['file']
    img_bytes = io.BytesIO(file.read())
    img = Image.open(img_bytes)
    probabilities = bc.beer_classification(img=img)
    return {'probabilities': probabilities.tolist()[0],
            'beerbrand': bc.get_classes()[np.argmax(probabilities.tolist()[0])]}


if __name__ == '__main__':
    app.run()


#file = open('test_hertogjan.jpg','rb')
#img_bytes = file.read()
#imageStream = io.BytesIO(img_bytes)
#img = Image.open(imageStream)
#probabilities = beer_classification.beer_classification(img=img)
#print(probabilities)

"""
set FLASK_ENV=development
set FLASK_APP=app.py
flask run
"""