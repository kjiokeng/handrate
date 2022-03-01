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


n_ones = 3
n_seconds = 3
signal_axis = 'pca'
n_units = 128
sample_freq = 100

def main():
    train_multiple_configs()

def train_multiple_configs():
    # Train the typical config
    n_seconds_list = [3]
    n_ones_list = [5]
    signal_axis_list = ['pca']
    n_units_list = [128]
    denoising_methods = ['']

    # Varying the size of the network
    # n_seconds_list = [1, 3, 5, 10]
    # n_ones_list = [5]
    # signal_axis_list = ['pca']
    # n_units_list = [16, 32, 64, 128, 256]
    # denoising_methods = ['']

    # Varying the number of ones per peak
    # n_seconds_list = [3]
    # n_ones_list = [1, 3, 5, 11, 21]
    # signal_axis_list = ['pca']
    # n_units_list = [32, 64, 128]
    # denoising_methods = ['wavelet2', 'bandpass']

    # Varying the signal axis
    # n_seconds_list = [3]
    # n_ones_list = [3, 5]
    # signal_axis_list = ['x', 'y', 'z', 'pca']
    # n_units_list = [32]
    # denoising_methods = ['wavelet2', 'bandpass']

    configs = []
    for n_ones in n_ones_list:
        for n_seconds in n_seconds_list:
            for signal_axis in signal_axis_list:
                for n_units in n_units_list:
                    for denoising_method in denoising_methods:
                        configs.append([n_seconds, n_ones, signal_axis, n_units, denoising_method])

    n_configs = len(configs)
    for config_id in range(n_configs):
        config = configs[config_id]
        n_seconds = config[0]
        n_ones = config[1]
        signal_axis = config[2]
        n_units = config[3]
        denoising_method = config[4]
        
        print("\n\n")
        print("--------------------------------------------------------------------")
        print("--------------------------------------------------------------------")
        print("Training for config {:d}/{:d}: {:.1f}%: n_seconds={:d}, n_ones={:d}, signal_axis={:s}, n_units={:d}, denoising_method={:s}" \
        .format(config_id+1, n_configs, (config_id+1)*100/n_configs, n_seconds, n_ones, signal_axis, n_units, denoising_method))
        train_config(n_seconds, n_ones, signal_axis, n_units, denoising_method)

def f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def one_over_f1(y_true, y_pred): #taken from old keras source code
    return 1/f1(y_true, y_pred)

def create_weighted_binary_crossentropy(zero_weight, one_weight):

    def weighted_binary_crossentropy(y_true, y_pred):

        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy

def train_config(n_seconds, n_ones, signal_axis, n_units, denoising_method=""):
    # Useful variables
    datafile = '../matlab/datasets/handrate-dataset-{:d}s-{:d}ones-{:s}.mat'.format(n_seconds, n_ones, signal_axis)
    raw_datafile = '../matlab/datasets/handrate-dataset-raw-{:d}s-{:d}ones-{:s}.mat'.format(n_seconds, n_ones, signal_axis)
    model_filename = 'models/handratenn/handrate-model-{:d}s-{:d}ones-{:s}-{:d}units.h5'.format(n_seconds, n_ones, signal_axis, n_units)
    model_checkpoint_filename = model_filename

    # snrtype = "lowsnrdenoised"
    # datafile = '../matlab/datasets/handrate-simu-dataset-{:d}s-{:d}ones-{:s}{:s}.mat'.format(n_seconds, n_ones, signal_axis, snrtype)
    # raw_datafile = '../matlab/datasets/handrate-simu-dataset-raw-{:d}s-{:d}ones-{:s}{:s}.mat'.format(n_seconds, n_ones, signal_axis, snrtype)
    # model_filename = 'models/handratenn/test/handrate-simu-model-test-{:d}s-{:d}ones-{:s}-{:d}units{:s}.h5'.format(n_seconds, n_ones, signal_axis, n_units, snrtype)
    # history_filename = 'models/handratenn/test/handrate-simu-model-test-{:d}s-{:d}ones-{:s}-{:d}units-history.pkl'.format(n_seconds, n_ones, signal_axis, n_units)
    # model_checkpoint_filename = model_filename

    # datafile = '../matlab/datasets/handrate-dataset-{:d}s-{:d}ones-{:s}-{:s}.mat'.format(n_seconds, n_ones, signal_axis, denoising_method)
    # raw_datafile = '../matlab/datasets/handrate-dataset-raw-{:d}s-{:d}ones-{:s}-{:s}.mat'.format(n_seconds, n_ones, signal_axis, denoising_method)
    # model_filename = 'models/handratenn/handrate-model-{:d}s-{:d}ones-{:s}-{:d}units-{:s}.h5'.format(n_seconds, n_ones, signal_axis, n_units, denoising_method)
    # model_checkpoint_filename = model_filename

    # Load the data
    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = load_data(datafile)
    
    # Expand the output arrays to fit with the model
    Y_train = np.expand_dims(Y_train, axis=2).astype(float)
    Y_dev = np.expand_dims(Y_dev, axis=2).astype(float)
    Y_test = np.expand_dims(Y_test, axis=2).astype(float)

    # Custom loss
    w0 = (1.2 * n_ones)/sample_freq
    w1 = 1 - w0
    weighted_binary_crossentropy = create_weighted_binary_crossentropy(w0, w1)


    # Build or load the model
    if os.path.isfile(model_filename) and False:
        print("-------- Loading model file {:s}".format(model_filename))
        model = load_model(model_filename, custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy})
    else:
        print("aaaaaaaaaaaa\naaaaaaaaaaa\naaaaaaaaaaa\naaaaaaaaaaa")
        model = handrate_model(X_train.shape[1:], n_units=n_units)

    # Model visualization
    model.summary()


    # Train the model
    # opt = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, decay=0.01)
    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
    # opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.01)
    # opt = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, decay=0.01)
    # opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
    # model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
    model.compile(loss=weighted_binary_crossentropy, optimizer=opt, metrics=["accuracy", f1])
    # checkpointer = ModelCheckpoint(filepath=model_checkpoint_filename, monitor="val_loss", verbose=1, save_best_only=True)
    # earlystopping = EarlyStopping(monitor='val_loss', verbose=1, patience=5)
    # hist = model.fit(X_train, Y_train, validation_data=(X_dev, Y_dev), epochs=50, batch_size=8, callbacks=[checkpointer,earlystopping])

    # checkpointer = ModelCheckpoint(filepath=model_checkpoint_filename, monitor="loss", verbose=1, save_best_only=True)
    # earlystopping = EarlyStopping(monitor='val_loss', verbose=1, patience=500)
    # hist = model.fit(X_train, Y_train, validation_data=(X_dev, Y_dev), epochs=1000, batch_size=8, callbacks=[checkpointer,earlystopping])

    # checkpointer = ModelCheckpoint(filepath=model_checkpoint_filename, monitor="val_loss", verbose=1, save_best_only=True)
    checkpointer = ModelCheckpoint(filepath=model_checkpoint_filename, monitor="val_f1", verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', verbose=1, patience=50)
    hist = model.fit(X_train, Y_train, validation_data=(X_dev, Y_dev), epochs=1000, batch_size=16, callbacks=[checkpointer, earlystopping])

    # Saving the objects:
    # with open(variables_filename, 'wb') as f:
    #     pickle.dump([hist], f)

    # Evaluate the model
    # loss, acc = model.evaluate(X_test, Y_test)
    # print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
    
    # Free used memory
    K.clear_session()

    # Visualize some predictions
    # predictions = model.predict(X_test)
    # for i in range(len(predictions)):
    #     p = predictions[i,:,0]
    #     gt = Y_test[i,:,0]
    #     t = range(300)
    #     plt.plot(t, gt, 'b-', t, p, 'r--')
    #     plt.show()

    return model

def handrate_model(input_shape, n_units=128):
    """
    Function creating the model's graph in Keras.
    
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """
    dropout_rate = 0.2

    X_input = Input(shape = input_shape)
    X = X_input
    
    # Step 1: CONV layer
    for i in range(1):
        X = Conv1D(n_units, kernel_size=3, strides=1, padding="same")(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Dropout(rate=dropout_rate)(X)

    # Step 2: First GRU Layer (encoder)
    X = Bidirectional(GRU(units = n_units, return_sequences = True), name='encoder_gru')(X)
    X = Dropout(rate=dropout_rate)(X)
    X = BatchNormalization()(X)
    
    # Step 3: Second GRU Layer (decoder)
    X = Bidirectional(GRU(units = n_units, return_sequences = True), name='decoder_gru')(X)
    X = Dropout(rate=dropout_rate)(X)
    X = BatchNormalization()(X)
    # X = Dropout(rate=dropout_rate)(X)
    
    # Step 4: Time-distributed dense layer
    X = TimeDistributed(Dense(1, activation = "sigmoid"))(X)

    model = Model(inputs = X_input, outputs = X)
    
    return model

# def handrate_model(input_shape, n_units=128):
#     """
#     Function creating the model's graph in Keras.
    
#     Argument:
#     input_shape -- shape of the model's input data (using Keras conventions)

#     Returns:
#     model -- Keras model instance
#     """
#     X_input = Input(shape = input_shape)
    
#     # Step 1: CONV layer
#     X = Conv1D(n_units, kernel_size=15, strides=1, padding="same")(X_input)
#     X = BatchNormalization()(X)
#     X = Activation('relu')(X)
#     X = Dropout(rate=0.2)(X)

#     # Step 2: First GRU Layer
#     # X = GRU(units = 16, return_sequences = True)(X)
#     X = Bidirectional(GRU(units = n_units, return_sequences = True))(X)
#     X = Dropout(rate=0.2)(X)
#     X = BatchNormalization()(X)
    
#     # Step 3: Second GRU Layer
#     # X = GRU(units = 16, return_sequences = True)(X)
#     X = Bidirectional(GRU(units = n_units, return_sequences = True))(X)
#     X = Dropout(rate=0.2)(X)
#     X = BatchNormalization()(X)
#     X = Dropout(rate=0.2)(X)
    
#     # Step 4: Time-distributed dense layer
#     X = TimeDistributed(Dense(1, activation = "sigmoid"))(X)

#     model = Model(inputs = X_input, outputs = X)
    
#     return model 

def load_data(filename):
    print("--- Loading data file: ", filename)
    data = sio.loadmat(filename)
    X_train, Y_train = np.array(data['X_train']), np.array(data['Y_train'])
    X_dev, Y_dev = np.array(data['X_dev']), np.array(data['Y_dev'])
    X_test, Y_test = np.array(data['X_test']), np.array(data['Y_test'])

    return (X_train, Y_train, X_dev, Y_dev, X_test, Y_test)


if __name__ == '__main__':
    main()
