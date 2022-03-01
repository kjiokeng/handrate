import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from tensorflow.keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Adam
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

print("Using Tensorflow", tf.__version__)


# Command line arguments default values
multi = False
n_seconds = 3
n_ones = 5
signal_axis = '123456'
n_units = 128
lr = 0.1
sample_freq = 200

def channelPool(x):
    return K.max(x,axis=-1)

def create_weighted_binary_crossentropy(zero_weight, one_weight):

    def weighted_binary_crossentropy(y_true, y_pred):
        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy

def train_multiple_configs():
    # Train the typical config
    n_seconds_list = [3]
    n_ones_list = [5]
    signal_axis_list = ['pca']
    n_units_list = [128]
    denoising_methods = ['']

    configs = []
    for n_ones in n_ones_list:
        for n_seconds in n_seconds_list:
            for signal_axis in signal_axis_list:
                for n_units in n_units_list:
                    configs.append([n_seconds, n_ones, signal_axis, n_units])

    n_configs = len(configs)
    for config_id in range(n_configs):
        config = configs[config_id]
        n_seconds = config[0]
        n_ones = config[1]
        signal_axis = config[2]
        n_units = config[3]
        
        print("\n\n")
        print("--------------------------------------------------------------------")
        print("--------------------------------------------------------------------")
        print("Training for config {:d}/{:d}: {:.1f}%: n_seconds={:d}, n_ones={:d}, signal_axis={:s}, n_units={:d}" \
        .format(config_id+1, n_configs, (config_id+1)*100/n_configs, n_seconds, n_ones, signal_axis, n_units))
        train_config(n_seconds, n_ones, signal_axis, n_units)

def train_config(n_seconds, n_ones, signal_axis, n_units, lr=0.1):
    # Useful variables
    datafile = '../matlab/datasets/handrate-md-dataset-{:d}s-{:d}ones-{:s}.mat'.format(n_seconds, n_ones, signal_axis)
    raw_datafile = '../matlab/datasets/handrate-md-dataset-raw-{:d}s-{:d}ones-{:s}.mat'.format(n_seconds, n_ones, signal_axis)
    model_filename = 'models/handratenn/md/handrate-md-model-{:d}s-{:d}ones-{:s}-{:d}units.h5'.format(n_seconds, n_ones, signal_axis, n_units)
    model_checkpoint_filename = model_filename

    # Load the data
    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = load_data(datafile)
    print("ssss", X_train.shape)
    
    # Expand the output arrays to fit with the model
    Y_train = np.expand_dims(Y_train, axis=2)
    Y_dev = np.expand_dims(Y_dev, axis=2)
    Y_test = np.expand_dims(Y_test, axis=2)

    # Custom loss
    w0 = (1.2 * n_ones)/sample_freq
    w1 = 1 - w0
    custom_loss = create_weighted_binary_crossentropy(w0, w1)
    keras.losses.weighted_binary_crossentropy = custom_loss

    # Build or load the model
    if os.path.isfile(model_filename):
        print("-------- Loading model file {:s}".format(model_filename))
        model = load_model(model_filename)
    else:
        model = handrate_md_model(X_train.shape[1:], n_units=n_units)

    # Model visualization
    model.summary()


    # Train the model
    opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss=custom_loss, optimizer=opt, metrics=["accuracy"])
    checkpointer = ModelCheckpoint(filepath=model_checkpoint_filename, monitor="val_loss", verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', verbose=1, patience=50)
    hist = model.fit(X_train, Y_train, validation_data=(X_dev, Y_dev), epochs=1000, batch_size=16, callbacks=[checkpointer, earlystopping])
    
    # Free used memory
    K.clear_session()

    return model

def handrate_md_model(input_shape, n_units=128):
    """
    Function creating the model's graph in Keras.
    
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """
    dropout_rate = 0.2
    n_time_steps = input_shape[0]
    height = input_shape[1]
    n_axes = input_shape[2]

    X_input = Input(shape = input_shape)
    X = X_input
    
    # Step 1: CONV layer
    for i in range(1):
        X = Conv2D(n_units, kernel_size=7, strides=1, padding="same")(X)
        # X = Conv1D(n_units, kernel_size=7, strides=1, padding="same")(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Dropout(rate=dropout_rate)(X)

    # X = Reshape((n_time_steps, n_units * height))(X)
    X = Conv2D(1, kernel_size=7, strides=1, padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Lambda(channelPool)(X)

    # Step 2: First GRU Layer
    X = Bidirectional(GRU(units = n_units, return_sequences = True))(X)
    X = Dropout(rate=dropout_rate)(X)
    X = BatchNormalization()(X)

    # X = Bidirectional(GRU(units = n_units, return_sequences = True))(X)
    # X = Dropout(rate=dropout_rate)(X)
    # X = BatchNormalization()(X)
    # X = Dropout(rate=dropout_rate)(X)
    
    # Step 4: Time-distributed dense layer
    X = TimeDistributed(Dense(1, activation = "sigmoid"))(X)

    model = Model(inputs = X_input, outputs = X)
    
    return model

def load_data(filename):
    print("--- Loading data file: ", filename)
    data = sio.loadmat(filename)
    X_train, Y_train = np.array(data['X_train']), np.array(data['Y_train'])
    X_dev, Y_dev = np.array(data['X_dev']), np.array(data['Y_dev'])
    X_test, Y_test = np.array(data['X_test']), np.array(data['Y_test'])

    return (X_train, Y_train, X_dev, Y_dev, X_test, Y_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train HandRate Multi-Dimensional NN')
    parser.add_argument('-m', '--multi', default=multi, action="store_true",
        help='Train for the multiple configs defined in this file')
    parser.add_argument('-s', '--seconds', default=n_seconds, type=float,
        help='Length of input slices')
    parser.add_argument('-o', '--ones', default=n_ones, type=int,
        help='Number of ones for each peak')
    parser.add_argument('-a', '--axes', default=signal_axis,
        help='A concatenation of all signal axes to use. 1, 2, 3 for acc x, y, z;\n4, 5, 6 for gyr x, y, z;\n7 and 8 for acc_pca and gyr_pca')
    parser.add_argument('-u', '--units', default=n_units, type=int,
        help='The number of units (size) to use for the model')
    parser.add_argument('-l', '--lr', default=lr, type=float,
        help='The learning rate to use when training the model')
    args = parser.parse_args()

    multi = args.multi    

    if multi:
        train_multiple_configs()
    else:
        n_seconds = args.seconds
        n_ones = args.ones
        signal_axis = args.axes
        n_units = args.units
        lr = args.lr

        train_config(n_seconds, n_ones, signal_axis, n_units, lr=lr)

