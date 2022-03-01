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
    decoder_inputs = Input(shape=(output_timesteps, 1), name='decoder_inputs')

    # Conv1D layer
    # X = encoder_inputs
    # X = Conv1D(hidden_size, kernel_size=7, strides=1, padding="same")(X)
    # X = BatchNormalization()(X)
    # X = Activation('relu')(X)

    # Encoder GRU
    encoder_gru = GRU(hidden_size, return_sequences=True, return_state=True, name='encoder_gru')
    encoder_out, encoder_state = encoder_gru(encoder_inputs)

    # Set up the decoder GRU, using `encoder_states` as initial state.
    decoder_gru = GRU(hidden_size, return_sequences=True, return_state=True, name='decoder_gru')
    decoder_out, decoder_state = decoder_gru(decoder_inputs, initial_state=encoder_state)

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
    full_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)
    full_model.compile(optimizer='adam', loss='binary_crossentropy')

    full_model.summary()

    """ Inference model """
    batch_size = 1

    """ Encoder (Inference) model """
    encoder_inf_inputs = Input(batch_shape=(batch_size, input_timesteps, input_dim2), name='encoder_inf_inputs')
    encoder_inf_out, encoder_inf_state = encoder_gru(encoder_inf_inputs)
    encoder_model = Model(inputs=encoder_inf_inputs, outputs=[encoder_inf_out, encoder_inf_state])

    """ Decoder (Inference) model """
    decoder_inf_inputs = Input(batch_shape=(batch_size, 1, output_dim2), name='decoder_word_inputs')
    encoder_inf_states = Input(batch_shape=(batch_size, input_timesteps, hidden_size), name='encoder_inf_states')
    decoder_init_state = Input(batch_shape=(batch_size, hidden_size), name='decoder_init')

    decoder_inf_out, decoder_inf_state = decoder_gru(decoder_inf_inputs, initial_state=decoder_init_state)
    attn_inf_out, attn_inf_states = attn_layer([encoder_inf_states, decoder_inf_out])
    decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_inf_out, attn_inf_out])
    decoder_inf_pred = TimeDistributed(dense)(decoder_inf_concat)
    decoder_model = Model(inputs=[encoder_inf_states, decoder_init_state, decoder_inf_inputs],
                          outputs=[decoder_inf_pred, attn_inf_states, decoder_inf_state])

    return full_model, encoder_model, decoder_model


def infer_model(encoder_model, decoder_model, input_scal, output_timesteps):
    """
    Infer logic
    :param encoder_model: keras.Model
    :param decoder_model: keras.Model
    :param test_en_seq: sequence of word ids
    :param en_vsize: int
    :param fr_vsize: int
    :return:
    """

    encoder_input_values = np.zeros((input_scal.shape[0], 1, 1))
    encoder_input_values[0, 0, 0] = 0.5
    # encoder_input_values = np.random.rand(input_scal.shape[0], 1, 1)

    enc_outs, enc_last_state = encoder_model.predict(input_scal)
    dec_state = enc_last_state
    attention_weights = []
    output_signal = []
    for i in range(output_timesteps):
        dec_out, attention, dec_state = decoder_model.predict([enc_outs, dec_state, encoder_input_values])
        sampled_val = dec_out[0, 0, 0]

        encoder_input_values[0, 0] = sampled_val

        attention_weights.append((sampled_val, attention))
        output_signal.append(sampled_val)

    print("End of function")
    return np.array(output_signal), attention_weights



if __name__ == '__main__':
    """
    Function creating the model's graph in Keras.
    
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """

    # Useful variables
    hidden_size = 256

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
    dropout_rate = 0.2

    # Defining the full model
    full_model, infer_enc_model, infer_dec_model = define_models(
        hidden_size=hidden_size, input_shape=input_shape, output_shape=output_shape)
    
    # Train the model
    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
    full_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

    model_filename = 'attn-256.tf'
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
        decoded_signal, attn_weights = infer_model(
            encoder_model=infer_enc_model, decoder_model=infer_dec_model,
            input_scal=input_seq, output_timesteps=output_timesteps)
        
        # decoded_signal = model.predict([input_seq, np.zeros((1, output_timesteps, 1))])
        print("aaaaaaaaaaaa", decoded_signal.shape)
        prediction = decoded_signal
        
        should_plot = True
        if should_plot:
            ground_truth = decoder_target_data[seq_index, :, 0]
            # prediction = decoded_signal[0, :, 0]

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