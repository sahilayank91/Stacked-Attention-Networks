"""
This function returns the image features for a given image.
The image is first resized to 224*224
The dimension of the extracted feature is (1, 14, 14, 512)

@Parameters
image : path to image
"""

from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
from get_image_model import get_image_model
import numpy as np


def get_image_features(input):
    input_image = image.load_img(input, target_size=(224, 224))
    image_array = image.img_to_array(input_image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    image_model = get_image_model()
    image_features = image_model.predict(image_array)
    return image_features
