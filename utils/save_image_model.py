"""
This function saves the image model.
The image model id VGG-19 where the image features are extracted from the last pooling layer.
"""


from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.models import model_from_json
import numpy as np

def save_image_model():
    base_model = VGG19(weights='imagenet')
    image_model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
    image_model_json = image_model.to_json()
    with open("models/image_model_json", "w") as json_file:
        json_file.write(image_model_json)
    image_model.save_weights("models/image_model.h5")
