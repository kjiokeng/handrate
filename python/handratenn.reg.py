import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from tensorflow.keras.layers import GRU, Bidirectional, BatchNormalization, Reshape, MaxPooling2D, AveragePooling2D, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling1D, Concatenate, AveragePooling1D
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

print("Using Tensorflow", tf.__version__)



# datafile = '../matlab/datasets/handrate-dataset-3s-5ones-pca.mat'
datafile = '../matlab/datasets/handrate-reg-dataset-1.5s-1ones-pca.mat'
n_units = 128
n_ones = 1
output_sample_freq = 250

def load_data(filename):
    print("--- Loading data file: ", filename)
    data = sio.loadmat(filename)
    X_train, Y_train = np.array(data['X_train']), np.array(data['Y_train'])
    X_dev, Y_dev = np.array(data['X_dev']), np.array(data['Y_dev'])
    X_test, Y_test = np.array(data['X_test']), np.array(data['Y_test'])

    return (X_train, Y_train, X_dev, Y_dev, X_test, Y_test)

def get_trained_models(X_train, Y_train, X_dev, Y_dev, dropout_rate=0.2):
    # Useful variables
    input_shape = X_train.shape[1:]
    output_shape = Y_train.shape[1:]
    n_output_steps = output_shape[0]

    # Prepare for the model
    decoder_input_data_train = np.zeros(Y_train.shape)
    decoder_input_data_dev = np.zeros(Y_dev.shape)
    decoder_input_data_train[:, 1:] = Y_train[:, :-1]
    decoder_input_data_dev[:, 1:] = Y_dev[:, :-1]

    
    #################### Encoder network ####################
    encoder_inputs = Input(shape = input_shape, name="input_scalogram")
    X = encoder_inputs
    # X = Conv1D(n_units, kernel_size=5, strides=1, padding="same")(X)
    # X = BatchNormalization()(X)
    # X = Activation('relu')(X)

    # X = Bidirectional(GRU(units = n_units, return_sequences = True), name='encoder_gru')(X)
    X = Conv1D(128, 3, padding='same', activation='relu')(X)
    X = Conv1D(32, 3, padding='same', activation='relu')(X)
    X = MaxPooling1D(3, strides=1, padding='same')(X)
    X = Conv1D(32, 3, padding='same', activation='relu')(X)
    X = Conv1D(64, 3, padding='same', activation='relu')(X)
    X = MaxPooling1D(3, strides=1, padding='same')(X)

    for k in range(10):
        # Inception
        tower_1 = Conv1D(32, 1, padding='same', activation='relu')(X)
        tower_1 = Conv1D(32, 3, padding='same', activation='relu')(tower_1)
        tower_2 = Conv1D(32, 1, padding='same', activation='relu')(X)
        tower_2 = Conv1D(32, 5, padding='same', activation='relu')(tower_2)
        tower_3 = MaxPooling1D(3, strides=1, padding='same')(X)
        tower_3 = Conv1D(32, 1, padding='same', activation='relu')(tower_3)
        X = Concatenate(axis = 2)([tower_1, tower_2, tower_3])

    
    # X = AveragePooling1D(3, strides=1, padding='same')(X)
    # X = AveragePooling1D(3, strides=1, padding='same')(X)
    # X = Dropout(rate=dropout_rate)(X)
    # X = Flatten()(X)
    # X = LSTM(64, return_sequences=True)(X)
    X = Dense(n_units, activation="relu")(X)
    X = Dense(1)(X)

    full_model = Model(encoder_inputs, X)
    full_model.summary()


    #################### Model training ####################
    # Load the model if exists
    # model_filename = 'handrate.rnn.conv.{:d}.h5'.format(n_units)
    model_filename = 'handrate.reg.15s.{:d}units.h5'.format(n_units)
    if os.path.isfile(model_filename) or os.path.isdir(model_filename):
        print("------- Loading model from {:s}".format(model_filename))
        saved_model = load_model(model_filename)
        full_model.set_weights(saved_model.get_weights())
    
    # Actual training
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
    earlystopping = EarlyStopping(monitor='val_mae', mode="min", verbose=1, patience=100)
    checkpointer = ModelCheckpoint(filepath=model_filename, monitor="val_mae", mode='min', verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_mae', mode='min', factor=0.1, patience=10, min_lr=0.000001, verbose=1)

    opt = Adam(learning_rate=0.001)
    full_model.compile(loss='mse', optimizer=opt, metrics=["mae"])
    full_model.fit(X_train, Y_train,
          batch_size=32,
          epochs=1000,
          validation_data=(X_dev, Y_dev),
          callbacks=[checkpointer, reduce_lr, earlystopping, tensorboard])


    return full_model

def predict_from_scalogram(input_scal, full_model):
        return full_model.predict(input_scal)


if __name__ == '__main__':
    #################### Data loading and preparation ####################
    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = load_data(datafile)
    print("X_train.shape = ", X_train.shape)
    print("Y_train.shape = ", Y_train.shape)
    print("X_dev.shape = ", X_dev.shape)
    print("Y_dev.shape = ", Y_dev.shape)
    print("X_test.shape = ", X_test.shape)
    print("Y_test.shape = ", Y_test.shape)

    # Expand the output arrays to fit with the model
    Y_train = Y_train.T * 1000
    Y_dev = Y_dev.T * 1000
    Y_test = Y_test.T * 1000

    Y_train = np.expand_dims(Y_train, axis=2).astype(float)
    Y_dev = np.expand_dims(Y_dev, axis=2).astype(float)
    Y_test = np.expand_dims(Y_test, axis=2).astype(float)

    
    #################### Train or load trained models ####################
    input_shape = X_train.shape[1:]
    output_shape = Y_train.shape[1:]
    n_output_steps = output_shape[0]
    dropout_rate = 0.2
    full_model = get_trained_models(X_train, Y_train,
        X_dev, Y_dev, dropout_rate=dropout_rate)


    #################### Make few predictions ####################
    # eval_set_in, eval_set_out = (X_train, Y_train)
    eval_set_in, eval_set_out = (X_test, Y_test)
    for seq_index in range(100):
        # Take one scalogram and predict from it
        input_scal = eval_set_in[seq_index:seq_index + 1]
        print("input_scal.shape = ", input_scal.shape)

        decoded_signal = predict_from_scalogram(input_scal, full_model)
        # decoded_signal = full_model.predict([input_scal, np.zeros((1, n_output_steps, 1))])
        # decoded_signal = full_model.predict([input_scal, eval_set_out[seq_index:seq_index+1]])

        prediction = decoded_signal[0, :, 0]
        ground_truth = eval_set_out[seq_index, :, 0]
        t = range(len(prediction))
        plt.figure(figsize=(20, 5))
        gt_plot, = plt.plot(t, ground_truth, 'b-', label="Ground truth")
        pred_plot, = plt.plot(t, prediction, 'r-', label="Prediction")
        plt.show()