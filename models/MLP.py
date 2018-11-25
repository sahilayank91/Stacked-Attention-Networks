from keras.models import Sequential
from keras.layers.core import Reshape, Activation, Dropout
from keras.layers import LSTM, Merge, Dense, Embedding

number_of_hidden_units_LSTM = 512
max_length_questions = 54
word_feature_size = 300
nb_classes = 430


def model(args):

    # Image model
    model_image = Sequential()
    model_image.add(Flatten())
    model_image.add(Dense(4096, activation = "relu"))

    # Language Model
    model_language = Sequential()
    model_language.add(LSTM(number_of_hidden_units_LSTM, return_sequences=True, input_shape=(max_length_questions, word_feature_size)))
    model_language.add(LSTM(number_of_hidden_units_LSTM, return_sequences=True))
    model_language.add(LSTM(number_of_hidden_units_LSTM, return_sequences=False))

    # combined model
    model = Sequential()
    model.add(Merge([model_language, model_image], mode='concat', concat_axis=1))


    for i in range(3):
        model.add(Dense(1024))
        model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))

    return model
