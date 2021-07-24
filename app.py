import io
import json

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
import beer_classification


app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello World!'

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = io.BytesIO(file.read())
        img = Image.open(img_bytes)
        probabilities = beer_classification.beer_classification(img=img)
        return jsonify(probabilities.tolist()[0])


if __name__ == '__main__':
    app.run()


#file = open('ragdoll.jpg','rb')
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