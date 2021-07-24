# https://www.python-engineer.com/posts/pytorch-model-deployment-with-flask/
# https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html
import requests
import numpy as np
from beer_classification import get_classes

#resp = requests.post("http://localhost:5000/predict",
#                     files={"file": open('test_heineken.jfif', 'rb')})

resp = requests.post("https://frits.herokuapp.com/predict",
                     files={"file": open('test_heineken.jfif', 'rb')})


print(resp.json())
print(f"Brand: {get_classes()[np.argmax(resp.json())]}")


