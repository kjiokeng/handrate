import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from tensorflow.keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pickle
import os.path


def f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def peak_location_diff(y_true, y_pred):
	return K.binary_crossentropy(y_true, y_pred)

# model_filename = "models/handratenn/handrate-model-3s-5ones-x-32units.h5"
# model_filename = "handrate.rnn.conv.1s.10ones.256units.h5"
model_filename = "handrate.rnn.conv.3s.128units.h5"
model = load_model(model_filename,
					custom_objects={'f1': f1,
									'peak_location_diff': peak_location_diff})

# model = handrate_model((75, 300))
opt = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
model.summary()

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.experimental_new_converter = True
# converter.allow_custom_ops = True # does the trick
tflite_model = converter.convert()

# Save the TF Lite model.
with tf.io.gfile.GFile('handrate-model-test.tflite', 'wb') as f:
  f.write(tflite_model)
