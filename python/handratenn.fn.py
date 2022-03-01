import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from tensorflow.keras.layers import GRU, Bidirectional, BatchNormalization, Reshape, Concatenate
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

from layers.attention import AttentionLayer

print("Using Tensorflow", tf.__version__)



datafile = '../matlab/datasets/handrate-dataset-3s-5ones-pca.mat'
n_units = 128

def load_data(filename):
    print("--- Loading data file: ", filename)
    data = sio.loadmat(filename)
    X_train, Y_train = np.array(data['X_train']), np.array(data['Y_train'])
    X_dev, Y_dev = np.array(data['X_dev']), np.array(data['Y_dev'])
    X_test, Y_test = np.array(data['X_test']), np.array(data['Y_test'])

    return (X_train, Y_train, X_dev, Y_dev, X_test, Y_test)


def define_models(input_shape, output_shape, hidden_size):
    """ Defining a NMT model """

    input_timesteps = input_shape[0]
    input_dim2 = input_shape[1]
    output_timesteps = output_shape[0]
    output_dim2 = output_shape[1]

    # Define an input sequence and process it.
    encoder_inputs = Input(shape=input_shape, name='encoder_inputs')

    # Conv1D layer
    X = encoder_inputs
    for k in range(2):
        X = Conv1D(hidden_size, kernel_size=3, strides=1, padding="same")(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

    # Encoder GRU
    encoder_gru = GRU(hidden_size, return_sequences=True, return_state=True, name='encoder_gru')
    encoder_out, encoder_state = encoder_gru(X)

    # Set up the decoder GRU, using `encoder_states` as initial state.
    decoder_gru = GRU(hidden_size, return_sequences=True, return_state=True, name='decoder_gru')
    decoder_out, decoder_state = decoder_gru(encoder_out, initial_state=encoder_state)

    # Attention layer
    attn_layer = AttentionLayer(name='attention_layer')
    attn_out, attn_states = attn_layer([encoder_out, decoder_out])

    # Concat attention input and decoder GRU output
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_out, attn_out])

    # Dense layer
    dense = Dense(output_dim2, activation='sigmoid', name='sigmoid_layer')
    dense_time = TimeDistributed(dense, name='time_distributed_layer')
    decoder_pred = dense_time(decoder_concat_input)

    # Full model
    full_model = Model(inputs=encoder_inputs, outputs=decoder_pred)
    full_model.compile(optimizer='adam', loss='binary_crossentropy')

    full_model.summary()

    return full_model


if __name__ == '__main__':
    """
    Function creating the model's graph in Keras.
    
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """

    # Useful variables
    hidden_size = 128

    # Load data
    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = load_data(datafile)
    print("X_train.shape = ", X_train.shape)
    print("Y_train.shape = ", Y_train.shape)
    print("X_dev.shape = ", X_dev.shape)
    print("Y_dev.shape = ", Y_dev.shape)
    print("X_test.shape = ", X_test.shape)
    print("Y_test.shape = ", Y_test.shape)
    
    # Expand the output arrays to fit with the model
    Y_train = np.expand_dims(Y_train, axis=2).astype(float)
    Y_dev = np.expand_dims(Y_dev, axis=2).astype(float)
    Y_test = np.expand_dims(Y_test, axis=2).astype(float)

    # Prepare for the model
    encoder_input_data = np.concatenate((X_train, X_dev))
    decoder_target_data = np.concatenate((Y_train, Y_dev))
    decoder_input_data = np.zeros(decoder_target_data.shape)
    decoder_input_data[:, 1:] = decoder_target_data[:, :-1]


    # Useful variables
    input_shape = X_train.shape[1:]
    output_shape = Y_train.shape[1:]
    input_timesteps = input_shape[0]
    input_dim2 = input_shape[1]
    output_timesteps = output_shape[0]
    output_dim2 = output_shape[1]

    # Defining the full model
    full_model = define_models(
        hidden_size=hidden_size, input_shape=input_shape, output_shape=output_shape)
    
    # Train the model
    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
    full_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

    model_filename = 'attn-direct-{:d}.h5'.format(hidden_size)
    if os.path.isfile(model_filename):
        saved_model = load_model(model_filename, custom_objects={'AttentionLayer': AttentionLayer})
        full_model.set_weights(saved_model.get_weights())
    else:
        full_model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=16,
              epochs=2,
              validation_split=0.2)
        full_model.save(model_filename)




    for seq_index in range(100):
        # Take one sequence (part of the training set)
        # for trying out decoding.
        input_seq = encoder_input_data[seq_index:seq_index + 1]
        
        decoded_signal = full_model.predict([input_seq, np.zeros((1, output_timesteps, 1))])
        print("aaaaaaaaaaaa", decoded_signal.shape)
        prediction = decoded_signal
        
        should_plot = True
        if should_plot:
            ground_truth = decoder_target_data[seq_index, :, 0]
            prediction = decoded_signal[0, :, 0]

            # print("1111111111111111111", prediction.shape)
            # print("2222222222222222222", ground_truth.shape)

            t = range(len(prediction))
            plt.figure(figsize=(20, 5))
            gt_plot, = plt.plot(t, ground_truth, 'b-', label="Ground truth")
            pred_plot, = plt.plot(t, prediction, 'r-', label="Prediction")
            # gt_peaks_plot, = plt.plot(gt_peaks, ground_truth[gt_peaks], "bx", label="Ground truth peaks")
            # pred_peaks_plot, = plt.plot(pred_peaks, prediction[pred_peaks], "rx", label="Prediction peaks")
            # plt.legend(handles=[gt_plot, gt_peaks_plot, pred_plot, pred_peaks_plot])
            plt.show()