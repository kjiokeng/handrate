import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from tensorflow.keras.layers import GRU, Bidirectional, BatchNormalization, Reshape, Lambda
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
# datafile = '../matlab/datasets/handrate-fs-dataset-1.5s-10ones-pca.mat'
datafile = '../matlab/datasets/handrate-fs-dataset-1.0s-10ones-pca.mat'
# datafile = '../matlab/datasets/handrate-fs-dataset-3.0s-10ones-pca.mat'
# datafile = '../matlab/datasets/handrate-fs-dataset-3.0s-50ones-pca.mat'
# datafile = '../matlab/datasets/handrate-fs-dataset-3.0s-25ones-pca.mat'
n_units = 128
n_ones = 10
output_sample_freq = 250
weighted_binary_crossentropy = None

def load_data(filename):
    print("--- Loading data file: ", filename)
    data = sio.loadmat(filename)
    X_train, Y_train = np.array(data['X_train']), np.array(data['Y_train'])
    X_dev, Y_dev = np.array(data['X_dev']), np.array(data['Y_dev'])
    X_test, Y_test = np.array(data['X_test']), np.array(data['Y_test'])

    return (X_train, Y_train, X_dev, Y_dev, X_test, Y_test)

def f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def create_weighted_binary_crossentropy(zero_weight=0.1, one_weight=0.9):
    def weighted_binary_crossentropy(y_true, y_pred):
        # Compute the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy

def soft_argmax(x):
    beta = 10
    return tf.reduce_sum(tf.cumsum(tf.ones_like(x)) * tf.exp(beta * x) / tf.reduce_sum(tf.exp(beta * x))) - 1

def peak_location_diff(y_true, y_pred):
    global weighted_binary_crossentropy
    # res = K.binary_crossentropy(y_true, y_pred)
    res = weighted_binary_crossentropy(y_true, y_pred)
    # res = res + 1e-2 * K.abs(soft_argmax(y_pred) - soft_argmax(y_true))
    return res

def get_trained_models(X_train, Y_train, X_dev, Y_dev, dropout_rate=0.2):
    # Useful variables
    input_shape = X_train.shape[1:]
    output_shape = Y_train.shape[1:]
    n_output_steps = output_shape[0]

    # Prepare for the model
    decoder_input_data_train = np.zeros((Y_train.shape[0], 1, 1))
    decoder_input_data_dev = np.zeros((Y_dev.shape[0], 1, 1))

    
    #################### Encoder network ####################
    encoder_inputs = Input(shape = input_shape, name="input_scalogram")
    # X = encoder_inputs
    # X = Conv1D(n_units, kernel_size=5, strides=1, padding="same")(X)
    # X = BatchNormalization()(X)
    # X = Activation('relu')(X)
    # X = Dropout(rate=dropout_rate)(X)

    # encoder = LSTM(n_units, return_state=True, name="encoder_lstm_2")
    # encoder_outputs, state_h, state_c = encoder(X)
    # encoder_states = [state_h, state_c] # Keep only the states


    #################### Decoder network ####################
    # decoder_inputs = Input(shape=(1, 1), name="target_output")
    # decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True, name="decoder_lstm")
    # decoder_dense = Dense(1, activation='sigmoid', name="output_dense")
    # decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    # decoder_outputs = decoder_dense(decoder_outputs)

    # all_outputs = []
    # inputs = decoder_inputs
    # states = encoder_states
    # for _ in range(n_output_steps):
    #     # Run the decoder on one timestep
    #     outputs, state_h, state_c = decoder_lstm(inputs,
    #                                             initial_state=states)
    #    outputs = decoder_dense(outputs)
    #    # Store the current prediction (we will concatenate all predictions later)
    #    all_outputs.append(outputs)
    #    # Reinject the outputs as inputs for the next loop iteration
    #    # as well as update the states
    #    inputs = outputs
    #    states = [state_h, state_c]

    # Concatenate all predictions
    # decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)


    #################### Define the full model ####################
    # full_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # full_model.summary()


    #################### Model training ####################
    # Custom loss function
    w0 = (1.2 * n_ones)/output_sample_freq
    w1 = 1 - w0
    global weighted_binary_crossentropy
    weighted_binary_crossentropy = create_weighted_binary_crossentropy(w0, w1)

    # Load the model if exists
    # model_id = 'handrate.rnn.conv.15s.{:d}units'.format(n_units)
    # model_id = 'handrate.rnn.conv.3s.{:d}units'.format(n_units)
    # model_id = 'handrate.rnn.conv.3s.128units'
    # model_id = 'handrate.rnn.3s.{:d}ones.{:d}units'.format(n_ones, n_units)
    model_id = 'handrate.rnn.nof.conv.1s.{:d}ones.{:d}units'.format(n_ones, n_units)
    model_filename = model_id + '.h5'
    if os.path.isfile(model_filename) or os.path.isdir(model_filename):
        print("------- Loading model from {:s}".format(model_filename))
        # saved_model = load_model(model_filename, 
        full_model = load_model(model_filename, 
            custom_objects={"weighted_binary_crossentropy": weighted_binary_crossentropy,
                            "f1": f1,
                            "peak_location_diff": peak_location_diff})
        print("Saved model loaded")
        # full_model.set_weights(saved_model.get_weights())
        # del saved_model
    
    full_model.summary()

    # Actual training
    log_dir = "logs/fit/" + model_id + "-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
    earlystopping = EarlyStopping(monitor='val_f1', mode='max', verbose=1, patience=100)
    # checkpointer = ModelCheckpoint(filepath=model_filename, monitor="val_f1", mode='max', verbose=1, save_best_only=True)
    checkpointer = ModelCheckpoint(filepath=model_filename, monitor="val_f1", mode='max', verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_f1', mode='max', factor=0.2, patience=30, min_lr=0.00001, verbose=1)

    lr = 0.01
    opt = Adam(learning_rate=lr)
    # opt = RMSprop(learning_rate=lr)
    # full_model.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=["accuracy", f1])
    full_model.compile(loss=peak_location_diff, optimizer=opt, metrics=["accuracy", f1])
    full_model.fit([X_train, decoder_input_data_train], Y_train,
          batch_size=32,
          epochs=10000,
          # epochs=50,
          validation_data=([X_dev, decoder_input_data_dev], Y_dev),
          # callbacks=[checkpointer, reduce_lr, earlystopping, tensorboard])
          callbacks=[checkpointer, reduce_lr, tensorboard])
    # full_model.save(model_filename)


    #################### Inference mode (sampling) ####################
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


    return full_model, encoder_model, decoder_model

def predict_from_scalogram(input_scal, n_output_steps, encoder_model, decoder_model, full_model=None):
        batch_size = input_scal.shape[0]

        # Encode the input as state vectors
        states_value = encoder_model.predict(input_scal)

        # Sampling loop
        target_seq = np.zeros((batch_size, 1, 1))
        decoded_signal = np.zeros((batch_size, n_output_steps, 1))
        for k in range(n_output_steps):
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + states_value)

            # Sample a value
            # print('output_tokens.shape', output_tokens.shape)
            sampled_vals = output_tokens[:, -1, 0]
            decoded_signal[:, k, 0] = sampled_vals

            # Update the target sequence (of length 1).
            target_seq = np.zeros((batch_size, 1, 1))
            target_seq[:, 0, 0] = sampled_vals

            # Update states
            states_value = [h, c]

        return decoded_signal


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
    Y_train = np.expand_dims(Y_train, axis=2).astype(float)
    Y_dev = np.expand_dims(Y_dev, axis=2).astype(float)
    Y_test = np.expand_dims(Y_test, axis=2).astype(float)

    
    #################### Train or load trained models ####################
    input_shape = X_train.shape[1:]
    output_shape = Y_train.shape[1:]
    n_output_steps = output_shape[0]
    dropout_rate = 0.2
    full_model, encoder_model, decoder_model = get_trained_models(X_train, Y_train,
        X_dev, Y_dev, dropout_rate=dropout_rate)


    #################### Make few predictions ####################
    # eval_set_in, eval_set_out = (X_train, Y_train)
    eval_set_in, eval_set_out = (X_test, Y_test)
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
