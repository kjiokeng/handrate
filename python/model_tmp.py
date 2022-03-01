import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from tensorflow.keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.metrics import BinaryAccuracy
import tensorflow.keras.backend as K
import tensorflow.keras

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import scipy.io as sio
from scipy.signal import find_peaks
import pickle
import argparse
import os.path
import datetime
from SoftDTWTF import SoftDTWTF

n_units = 128
dropout_rate = 0.2

if __name__ == '__main__':
#################### Encoder network ####################
    encoder_inputs = Input(shape = (300, 59), name="input_scalogram")
    X = encoder_inputs
    X = Conv1D(n_units, kernel_size=5, strides=1, padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Dropout(rate=dropout_rate)(X)

    encoder = LSTM(n_units, return_state=True, name="encoder_lstm_2")
    encoder_outputs, state_h, state_c = encoder(X)
    encoder_states = [state_h, state_c] # Keep only the states


    #################### Decoder network ####################
    decoder_inputs = Input(shape=(None, 1), name="target_output")
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(1, activation='sigmoid', name="output_dense")
    decoder_outputs = decoder_dense(decoder_outputs)


    #################### Define the full model ####################
    full_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    full_model.summary()