import keras
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.models import Sequential
from keras.layers.core import Reshape, Activation, Dropout
from keras.layers import LSTM, Merge, Dense, Embedding
from keras.utils import plot_model


number_of_hidden_units_LSTM = 512
max_length_questions = 54
word_feature_size = 300

def modify_question_feature(V_i):
    v1 = np.zeros(512)
    for row in range(196):
        v1 += V_i[0][:][row]*p1
    v1 = tf.convert_to_tensor(v1)
    return v1


def model(args):

    # Image model
    v_i = keras.layers.Dense(512, activation='tanh')(image_features)
    v_i = keras.layers.Reshape((512,196),input_shape=(1,14,14,512))(v_i)
    V_i = keras.layers.Reshape((196,512))(v_i)
    v_i = keras.layers.Dense(196, activation='tanh')(v_i)


    # Language Model
    v_q = tf.convert_to_tensor(qusetion_features, np.float32)
    v_q = keras.layers.LSTM(number_of_hidden_units_LSTM, return_sequences=True, input_shape=(max_length_questions, word_feature_size))(v_q)
    v_q = keras.layers.LSTM(number_of_hidden_units_LSTM, return_sequences=True)(v_q)
    v_q = keras.layers.LSTM(number_of_hidden_units_LSTM, return_sequences=True)(v_q)
    v_q = keras.layers.LSTM(number_of_hidden_units_LSTM, return_sequences=False)(v_q)
    V_q = keras.layers.Dropout(0.5)(v_q)
    v_q = keras.layers.RepeatVector(196)(V_q)
    v_q = keras.layers.Dense(196, activation='tanh')(v_q)


    # Combined Model
    # First Attention Layer
    h1 = keras.layers.Add()([v_i, v_q])
    f1 = keras.layers.Flatten()(h1)
    p1 = keras.layers.Dense(512, activation = 'softmax')(f1)
    v1 = keras.layers.Lambda(modify_question_feature)([V_i, p1])
    v_q = keras.layers.Add()([v1,V_q])


    v_i = keras.layers.Dense(196, activation = 'tanh')(input1)
    v_q = keras.layers.Dense(196, activation = 'tanh')(v_q)

    # Second Attention Layer
    h2 = keras.layers.Add()([v_i, v_q])
    f2 = keras.layers.Flatten()(h2)
    p2 = keras.layers.Dense(512, activation = 'softmax')(f2)
    v2 = keras.layers.Lambda(modify_question_feature)([V_i, p2])
    v_q = keras.layers.Add()([v2,V_q])

    p_ans = keras.layers.Dense(430, activation = "softmax")(v_q)

    model = Model(inputs = [image_features,qusetion_features], outputs = p_ans)
    return model
