"""
This function saves the image model.
The image model id VGG-19 where the image features are extracted from the last pooling layer.
"""

from keras.models import model_from_json
image_model_weights = "/home/iiitk/SAN/models/image_model.h5"
image_model_json = "/home/iiitk/SAN/models/image_model_json"


def get_image_model():
    json_file = open(image_model_json, 'r')
    image_model = json_file.read()
    json_file.close()
    image_model = model_from_json(image_model)
    image_model.load_weights(image_model_weights)
    return image_model
