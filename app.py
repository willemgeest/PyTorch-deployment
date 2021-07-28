import io
from PIL import Image
from flask import Flask, request
import beer_classification as bc
import object_detection as od
import numpy as np
from datetime import datetime
import logging

app = Flask(__name__)
logging.basicConfig(filename='record.log',
                    level=logging.DEBUG,
                    format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')


@app.route('/', methods=['GET'])
def home():
    app.logger.info('Visitor on home page')
    return {"message": "Welcome to the API of FRITS!",
            "datetime": f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"}


@app.route('/start', methods=['GET'])
def start():
    app.logger.info('Start of beer analysis')
    return {"message": "New beer analysis started",
            "datetime": f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"}



@app.route('/detect', methods=['POST'])
def detect():
    app.logger.info('Start of beer detection')
    file = request.files['file']
    img_bytes = io.BytesIO(file.read())
    img = Image.open(img_bytes)
    img.save(f"images/latest_image.jpg")
    obj_det_model = od.get_obj_det_model()
    boxes, n_beers, _, preds = od.find_bottles(image=img,
                                               model=obj_det_model,
                                               detection_threshold=0.8,
                                               GPU=False)
    # save image with boxes
    image_boxes = od.draw_boxes(image=img, boxes=boxes)
    image_boxes.save('images/latest_image_boxes.jpg')
    image_cropped, _ = od.crop_beers(image=img, boxes=boxes)
    image_cropped.save('images/latest_image_cropped.jpg')

    app.logger.info('End of beer detection')

    return {'boxes': boxes.tolist(),
            'n_beers': n_beers,
            'preds': preds.tolist()
            }


@app.route('/classify', methods=['POST'])
def classify():
    app.logger.info('Start of beer classification')
    file = request.files['file']
    img_bytes = io.BytesIO(file.read())
    img = Image.open(img_bytes)
    beerbrand, probabilities, heatmap = bc.beer_classification(img=img)
    heatmap.save(f"images/latest_image_heatmap.jpg")
    app.logger.info('End of beer classification')
    return {'beerbrand': beerbrand,
            'probabilities': dict(zip(bc.get_classes(),
                                      probabilities.tolist()))
            }





if __name__ == '__main__':
    app.run(Debug=True)


#file = open('test_hertogjan.jpg','rb')
#img_bytes = file.read()
#imageStream = io.BytesIO(img_bytes)
#img = Image.open(imageStream)


"""
set FLASK_ENV=development
set FLASK_APP=app.py
flask run --host 0.0.0.0
"""

