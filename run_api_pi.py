# https://www.python-engineer.com/posts/pytorch-model-deployment-with-flask/
# https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html
import requests
import object_detection as od
from PIL import Image

ip = '10.57.10.10:5000'
filename = 'test_hertogjan.jpg'
filename_cropped = f"{filename.split('.')[0]}_cropped.jpg"

# take picture

# send to detection API
resp_detect = requests.post(f"http://{ip}/detect",
                            files={"file": open(filename, 'rb')})

print(f'Detection API finished with status code {resp_detect.status_code}')

# crop image
if resp_detect.status_code==200:
    if resp_detect.json()['n_beers'] >= 1:
        cropped_image, _ = od.crop_beers(image=Image.open(filename),
                                      boxes=resp_detect.json()['boxes'])
        cropped_image.save(filename_cropped)


# classify image
resp_classify = requests.post(f"http://{ip}/classify",
                              files={"file": open(filename_cropped, 'rb')})

print(f'Classification API finished with status code {resp_detect.status_code}')

# play sound when too few beers
if resp_detect.json()['n_beers'] <= 3:
    pass
    # python play_sound.py"

# control LED
if resp_detect.json()['n_beers'] >= 1:
    pass
    # sudo python3.6 control_led.py {beerbrand.lower().replace(' ', '')}

# semd telegram message
#telegram.send_telegram_message(to=receiver,
#                                           message=f'Dear {receiver}, FRITS contains only {n_beers} beers'
 #                                                  f'at this moment. Please provide new beers immediately!')


