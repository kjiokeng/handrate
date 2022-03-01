import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from tensorflow.keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Adam, RMSprop
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



datafile = '../matlab/datasets/handrate-dataset-3s-5ones-pca.mat'
# datafile = '../matlab/datasets/handrate-dataset-5s-5ones-pca.mat'
# datafile = '../matlab/datasets/handrate-fs-dataset-1.0s-10ones-pca.mat'
n_units = 128
n_seconds = 3

def load_data(filename):
    print("--- Loading data file: ", filename)
    data = sio.loadmat(filename)
    X_train, Y_train = np.array(data['X_train']), np.array(data['Y_train'])
    X_dev, Y_dev = np.array(data['X_dev']), np.array(data['Y_dev'])
    X_test, Y_test = np.array(data['X_test']), np.array(data['Y_test'])

    return (X_train, Y_train, X_dev, Y_dev, X_test, Y_test)

def soft_argmax(x):
    beta = 10
    return tf.reduce_sum(tf.cumsum(tf.ones_like(x)) * tf.exp(beta * x) / tf.reduce_sum(tf.exp(beta * x))) - 1

def peak_location_diff(y_true, y_pred):
    global weighted_binary_crossentropy
    res = K.binary_crossentropy(y_true, y_pred)
    # res = weighted_binary_crossentropy(y_true, y_pred)
    # res = res + 1e-2 * K.abs(soft_argmax(y_pred) - soft_argmax(y_true))
    return res


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

def f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

if __name__ == '__main__':
    """
    Function creating the model's graph in Keras.
    
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """

    # Load data
    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = load_data(datafile)
    print("X_train.shape = ", X_train.shape)
    print("Y_train.shape = ", Y_train.shape)
    print("X_dev.shape = ", X_dev.shape)
    print("Y_dev.shape = ", Y_dev.shape)
    print("X_test.shape = ", X_test.shape)
    print("Y_test.shape = ", Y_test.shape)
    
    print("X_train.isnan", np.count_nonzero(np.isnan(X_train)))
    print("Y_train.isnan", np.count_nonzero(np.isnan(Y_train)))
    print("X_dev.isnan", np.count_nonzero(np.isnan(X_dev)))
    print("Y_dev.isnan", np.count_nonzero(np.isnan(Y_dev)))
    print("X_test.isnan", np.count_nonzero(np.isnan(X_test)))
    print("Y_test.isnan", np.count_nonzero(np.isnan(Y_test)))

    # Expand the output arrays to fit with the model
    Y_train = np.expand_dims(Y_train, axis=2).astype(float)
    Y_dev = np.expand_dims(Y_dev, axis=2).astype(float)
    Y_test = np.expand_dims(Y_test, axis=2).astype(float)

    # Prepare fore the model
    encoder_input_data = np.concatenate((X_train, X_dev))
    decoder_target_data = np.concatenate((Y_train, Y_dev))
    decoder_input_data = np.zeros(decoder_target_data.shape)
    decoder_input_data[:, 1:] = decoder_target_data[:, :-1]


    # Useful variables
    input_shape = X_train.shape[1:]
    output_shape = Y_train.shape[1:]
    n_output_steps = output_shape[0]
    dropout_rate = 0

    # Encoder network
    encoder_inputs = Input(shape = input_shape)
    X = encoder_inputs
    for k in range(1):
        X = Conv1D(n_units, kernel_size=5, strides=1, padding="same")(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Dropout(rate=dropout_rate)(X)

    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(X)
    encoder_states = [state_h, state_c] # Keep only the states


    # Decoder network
    decoder_inputs = Input(shape=(None, 1))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True, dropout=dropout_rate)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(1, activation='sigmoid')
    decoder_outputs = decoder_dense(decoder_outputs)


    # Define the model itself
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.summary()


    n_ones = 10
    sample_freq = 100
    sample_freq = 250
    w0 = (1.2 * n_ones)/sample_freq
    w0 = w0 / 2
    w1 = 1 - w0
    weighted_binary_crossentropy = create_weighted_binary_crossentropy(w0, w1)

    
    model_filename = 'seq2seq.v3.{:d}s.{:d}units.h5'.format(n_seconds, n_units)
    if os.path.isfile(model_filename) or os.path.isdir(model_filename):
        print("------- Loading model from {:s}".format(model_filename))
        saved_model = load_model(model_filename,
            custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy,
                            'f1': f1,
                            'peak_location_diff': peak_location_diff})
        model.set_weights(saved_model.get_weights())
    

    # Train the model
    lr = 0.01
    for k in range(2):
        earlystopping = EarlyStopping(monitor='val_f1', mode='max', verbose=1, patience=15)
        checkpointer = ModelCheckpoint(filepath=model_filename, monitor="val_f1", mode='max', verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_f1', mode='max', factor=0.1, patience=5, min_lr=0.00001, verbose=1)

        print("------ Trainning with learning rate: ", lr)
        opt = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, decay=0.01)
        # opt = RMSprop(learning_rate=lr)
        model.compile(loss=peak_location_diff, optimizer=opt, metrics=["accuracy", f1])
        # model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
        #       batch_size=32,
        #       epochs=1000,
        #       validation_split=0.4,
        #       callbacks=[checkpointer, reduce_lr])
        lr = lr/10
        
        model.save(model_filename)


    # Inference mode (sampling)
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)


    def decode_scalogram(input_seq):
        # Encode the input as state vectors.
        model.reset_states()
        encoder_model.reset_states()
        decoder_model.reset_states()
        states_value = encoder_model.predict(input_seq)

        # Generate empty target sequence
        target_seq = np.zeros((1, 1, 1))
        # target_seq = np.random.rand(1, 1, 1)

        # Sampling loop for a batch of sequences
        decoded_signal = np.zeros((1, n_output_steps, 1))
        for k in range(n_output_steps):
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + states_value)

            # Sample a value
            sampled_val = output_tokens[0, -1, 0]
            decoded_signal[0, k, 0] = sampled_val

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, 1))
            target_seq[0, :, 0] = sampled_val

            # Update states
            states_value = [h, c]

        return decoded_signal


    for seq_index in range(100):
        # Take one scalogram and predict from it
        input_scal = eval_set_in[seq_index:seq_index + 1]
        print("input_scal.shape = ", input_scal.shape)

        decoded_signal = predict_from_scalogram(input_scal, n_output_steps, encoder_model, decoder_model)
        # decoded_signal = full_model.predict([input_scal, np.zeros((1, n_output_steps, 1))])
        # decoded_signal = full_model.predict([input_scal, eval_set_out[seq_index:seq_index+1]])

        fig = plt.figure(figsize=(20, 5))
        axes = fig.add_subplot(2, 1, 1)
        prediction = decoded_signal[0, :, 0]
        ground_truth = eval_set_out[seq_index, :, 0]
        t = [k/output_sample_freq for k in range(len(prediction))]
        gt_plot, = axes.plot(t, ground_truth, 'b-', label="Ground truth")
        pred_plot, = axes.plot(t, prediction, 'r-', label="Prediction")
        axes.margins(x=0, y=None)

        axes = fig.add_subplot(2, 1, 2)
        axes.imshow(np.squeeze(input_scal).T, aspect="auto")
        t = [k for k in range(0, input_scal.shape[1], 50)]
        t_labels = [str(k/100) for k in t]
        axes.set_xticks(t)
        axes.set_xticklabels(t_labels)
        
        plt.show()