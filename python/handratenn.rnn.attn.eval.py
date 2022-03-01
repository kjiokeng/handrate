import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from tensorflow.keras.layers import GRU, Bidirectional, BatchNormalization, Reshape, Dot, Concatenate
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

# Command line arguments default values
should_plot = False
dataset = 'dev'
distance = 65
height = 0
prominence = 0
sample_freq = 100
out_sample_freq = 100

weighted_binary_crossentropy = None


def evaluate_config(model_filename, datafile):
    # Useful variables
    global dataset

    # Load the model
    with_conv = 'conv' in model_filename
    full_model, encoder_model, decoder_model = get_trained_models(model_filename)

    # Load the data
    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = load_data(datafile)
    dataset = dataset.lower()
    if 'train' in dataset:
        print("Using training dataset")
        X = X_train
        Y = Y_train
    elif ('dev' in dataset) or ('val' in dataset):
        print("Using validation dataset")
        X = X_dev
        Y = Y_dev
    elif 'test' in dataset:
        print("Using test dataset")
        X = X_test
        Y = Y_test
    else:
        print("Unknown value for dataset argument. Using validation dataset")
        X = X_dev
        Y = Y_dev

    # Actual evaluation of each of the items
    n_items = X.shape[1]
    ibi_errs = []
    hr_errs =  []
    for item_id in range(n_items):
        x = X[0, item_id]
        y = Y[0, item_id]

        print("Evaluating file {:} / {:d} : {:.2f}%".format(item_id+1, n_items, (item_id + 1) * 100 / n_items))
        res = evaluate(full_model, encoder_model, decoder_model, n_seconds, x, y, item_id=item_id)
        if res['IBI_ERROR'] < 200:
            ibi_errs.append(res['IBI_ERROR'])
            hr_errs.append(res['HR_ERROR'])

    # Global result
    result = {
        'IBI_ERROR': ibi_errs,
        'IBI_ERROR_MEAN': np.mean(ibi_errs),
        'IBI_ERROR_MEDIAN': np.median(ibi_errs),
        'IBI_ERROR_90PRCTILE': np.percentile(ibi_errs, 90),

        'HR_ERROR': hr_errs,
        'HR_ERROR_MEAN': np.mean(hr_errs),
        'HR_ERROR_MEDIAN': np.median(hr_errs),
        'HR_ERROR_90PRCTILE': np.percentile(hr_errs, 90)
    }

    return result

def evaluate(full_model, encoder_model, decoder_model, n_seconds, input_scalogram, ground_truth_heartbeats, item_id=0):
    # Useful global variables
    global distance
    global height
    global prominence
    global sample_freq

    # Make the prediction
    prediction = predict(full_model, encoder_model, decoder_model, n_seconds, input_scalogram.T)
    end_time = prediction.shape[0]
    ground_truth = ground_truth_heartbeats[0:end_time, 0]

    # Find peaks
    gt_peaks, _ = find_peaks(ground_truth, distance=distance*out_sample_freq/100)
    pred_peaks, _ = find_peaks(prediction, distance=distance*out_sample_freq/100, height=height, prominence=prominence)
    pred_instants = np.zeros(prediction.shape)
    pred_instants[pred_peaks] = 1;

    # Compute statistics: InterBeat Interval, Heart Rate, Errors
    gt_ibi = np.mean(np.diff(gt_peaks)) / out_sample_freq
    # gt_hr = 60 * len(gt_peaks) / (end_time / out_sample_freq)
    gt_hr = 60 / gt_ibi

    pred_ibi = np.mean(np.diff(pred_peaks)) / out_sample_freq
    # pred_hr = 60 * len(pred_peaks) / (end_time / out_sample_freq)
    pred_hr = 60 / pred_ibi

    ibi_err = abs(pred_ibi - gt_ibi)
    hr_err = abs(pred_hr - gt_hr)

    # Save the results
    np.savetxt('predictions/gt{:d}.txt'.format(item_id), ground_truth)
    np.savetxt('predictions/pred{:d}.txt'.format(item_id), prediction)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", input_scalogram.shape)
    # Visualization
    global should_plot
    if should_plot:
        fig = plt.figure(figsize=(20, 5))
        axes = fig.add_subplot(2, 1, 1)
        t = [k/out_sample_freq for k in range(len(prediction))]
        gt_plot, = axes.plot(t, ground_truth, 'b-', label="Ground truth")
        pred_plot, = axes.plot(t, prediction, 'r-', label="Prediction")
        gt_peaks_plot, = axes.plot(gt_peaks/out_sample_freq, ground_truth[gt_peaks], "bx", label="Ground truth peaks")
        pred_peaks_plot, = axes.plot(pred_peaks/out_sample_freq, prediction[pred_peaks], "rx", label="Prediction peaks")
        axes.legend(handles=[gt_plot, gt_peaks_plot, pred_plot, pred_peaks_plot])
        axes.margins(x=0, y=None)

        axes = fig.add_subplot(2, 1, 2)
        axes.imshow(np.squeeze(input_scalogram).T, aspect="auto")
        t = [k for k in range(0, input_scalogram.shape[1], 50)]
        t_labels = [str(k/sample_freq) for k in t]
        axes.set_xticks(t)
        axes.set_xticklabels(t_labels)
        
        plt.show()

    # Result (convert from s to ms)
    return {
        'GROUNDTRUTH_HR': gt_hr,
        'GROUNDTRUTH_IBI': gt_ibi * 1000,
        'PREDICTED_HR': pred_hr,
        'PREDICTED_IBI': pred_ibi * 1000,
        'HR_ERROR': hr_err,
        'IBI_ERROR': ibi_err * 1000
    }

def predict(full_model, encoder_model, decoder_model, n_seconds, input_scalogram):
    # Useful variables
    # print("Prediction on input scalogram with shape", input_scalogram.shape)
    input_len, input_height = input_scalogram.shape
    # item_stride_seconds = max(n_seconds-1, 1)
    item_stride_seconds = n_seconds
    item_duration = int(n_seconds * sample_freq)
    item_stride = int(item_stride_seconds * sample_freq)

    item_duration_out = int(n_seconds * out_sample_freq)
    item_stride_out = int(item_stride_seconds * out_sample_freq)


    # Slice the (30s long) data into different items and build the model input
    begin_time = 0
    begin_times = []
    begin_time_out = 0
    begin_times_out = []
    while begin_time + item_duration <= input_len:
        begin_times.append(begin_time)
        begin_time += item_stride

        begin_times_out.append(begin_time_out)
        begin_time_out += item_stride_out

    # Build an input matrix for single-time prediction
    n_slices = len(begin_times)
    inputs = np.zeros((n_slices, item_duration, input_height))
    for k in range(n_slices):
        begin_time = begin_times[k]
        end_time = begin_time + item_duration
        input_scal = input_scalogram[begin_time:end_time, :]
        inputs[k, :, :] = input_scal / input_scal.max()

    # Actual prediction
    predictions = predict_from_scalogram(inputs,
            item_duration_out, encoder_model, decoder_model, full_model=full_model)
    end_time_out = begin_times_out[-1] + item_duration_out
    out = np.zeros(end_time_out) + 2

    # Reconstruct the output
    for k in range(n_slices):
        begin_time_out = begin_times_out[k]
        end_time_out = begin_time_out + item_duration_out
        # out[begin_time_out:end_time_out] = np.maximum(out[begin_time_out:end_time_out], np.squeeze(predictions[k, :]))
        # out[begin_time_out:end_time_out] = np.minimum(out[begin_time_out:end_time_out], np.squeeze(predictions[k, :]))
        out[begin_time_out:end_time_out] = np.squeeze(predictions[k, :])

    # Build an input matrix for single-time prediction
    # end_time_out = begin_times_out[-1] + item_duration_out
    # out = np.zeros(end_time_out)
    # for k in range(len(begin_times)):
    #     begin_time = begin_times[k]
    #     end_time = begin_time + item_duration
    #     input_scal = input_scalogram[begin_time:end_time, :]
    #     input_scal = input_scal / input_scal.max()
        
    #     input_scal = np.expand_dims(input_scal, axis=0)
    #     output_signal = predict_from_scalogram(input_scal,
    #         item_duration_out, encoder_model, decoder_model, full_model=full_model)
    #     output_signal = output_signal[0, :, 0]

    #     begin_time_out = begin_times_out[k]
    #     end_time_out = begin_time_out + item_duration_out
    #     out[begin_time_out:end_time_out] = np.maximum(out[begin_time_out:end_time_out], np.squeeze(output_signal))

    return out


def load_data(filename):
    print("Loading data from file", filename)
    data = sio.loadmat(filename)
    X_train, Y_train = np.array(data['scalograms_matrix_train']), np.array(data['heartbeats_matrix_train'])
    X_dev, Y_dev = np.array(data['scalograms_matrix_dev']), np.array(data['heartbeats_matrix_dev'])
    X_test, Y_test = np.array(data['scalograms_matrix_test']), np.array(data['heartbeats_matrix_test'])

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

def get_trained_models(model_filename, dropout_rate=0.2):
    # Useful variables
    # input_shape = X_train.shape[1:]
    # output_shape = Y_train.shape[1:]
    # n_output_steps = output_shape[0]

    # # Prepare for the model
    # decoder_input_data_train = np.zeros(Y_train.shape)
    # decoder_input_data_dev = np.zeros(Y_dev.shape)
    # decoder_input_data_train[:, 1:] = Y_train[:, :-1]
    # decoder_input_data_dev[:, 1:] = Y_dev[:, :-1]

    input_shape = (int(n_seconds*sample_freq), 75)

    
    #################### Encoder network ####################
    encoder_inputs = Input(shape = input_shape, name="input_scalogram")
    X = encoder_inputs
    X = Conv1D(n_units, kernel_size=5, strides=1, padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Dropout(rate=dropout_rate)(X)

    encoder = LSTM(n_units, return_sequences=True, return_state=True, name="encoder_lstm_2")
    encoder_outputs, state_h, state_c = encoder(X)
    encoder_states = [state_h, state_c] # Keep only the states


    #################### Decoder network ####################
    decoder_inputs = Input(shape=(250, 1), name="target_output")
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    # decoder_dense = Dense(1, activation='sigmoid', name="output_dense")
    # decoder_outputs = decoder_dense(decoder_outputs)


    #################### Attention mechanism ####################
    attention_dot = Dot(axes=-1)
    attention_out = attention_dot([decoder_outputs, encoder_outputs])
    attention_softmax = Activation('softmax')
    attention_out = attention_softmax(attention_out)

    context_dot = Dot(axes=[2,1])
    context_out = context_dot([attention_out, encoder_outputs])
    decoder_concat = Concatenate()
    decoder_combined_context = decoder_concat([context_out, decoder_outputs])

    # Has another weight + tanh layer as described in equation (5) of the paper
    output_timedist_1 = TimeDistributed(Dense(64, activation="tanh")) # equation (5) of the paper
    output = output_timedist_1(decoder_combined_context) # equation (5) of the paper
    output_timedist_2 = TimeDistributed(Dense(1, activation="sigmoid")) # equation (6) of the paper
    output = output_timedist_2(output) # equation (6) of the paper



    #################### Define the full model ####################
    full_model = Model([encoder_inputs, decoder_inputs], output)
    full_model.summary()


    #################### Model training ####################
    # Custom loss function
    w0 = (1.2 * n_ones)/out_sample_freq
    w1 = 1 - w0
    global weighted_binary_crossentropy
    weighted_binary_crossentropy = create_weighted_binary_crossentropy(w0, w1)

    # Load the model if exists
    saved_model = load_model(model_filename, 
        custom_objects={"weighted_binary_crossentropy": weighted_binary_crossentropy,
                        "f1": f1,
                        "peak_location_diff": peak_location_diff})
    full_model.set_weights(saved_model.get_weights())
    
    # Actual training
    # log_dir = "logs/fit/" + model_id + "-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
    # earlystopping = EarlyStopping(monitor='val_f1', mode='max', verbose=1, patience=40)
    # checkpointer = ModelCheckpoint(filepath=model_filename, monitor="val_f1", mode='max', verbose=1, save_best_only=True)
    # # checkpointer = ModelCheckpoint(filepath=model_filename, monitor="val_f1", mode='max', verbose=1)
    # reduce_lr = ReduceLROnPlateau(monitor='val_f1', mode='max', factor=0.1, patience=10, min_lr=0.00001, verbose=1)

    lr = 0.01
    opt = Adam(learning_rate=lr)
    # opt = RMSprop(learning_rate=lr)
    # full_model.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=["accuracy", f1])
    full_model.compile(loss=peak_location_diff, optimizer=opt, metrics=["accuracy", "Recall", "Precision", f1, "AUC"])
    # full_model.fit([X_train, decoder_input_data_train], Y_train,
    #       batch_size=32,
    #       epochs=10000,
    #       # epochs=50,
    #       validation_data=([X_dev, decoder_input_data_dev], Y_dev),
    #       callbacks=[checkpointer, reduce_lr, earlystopping, tensorboard])
          # callbacks=[checkpointer, reduce_lr, tensorboard])
    # full_model.save(model_filename)


    #################### Inference mode (sampling) ####################
    encoder_model = Model(encoder_inputs, [encoder_outputs] + encoder_states)

    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    # decoder_outputs = decoder_dense(decoder_outputs)
    # decoder_model = Model(
    #     [decoder_inputs] + decoder_states_inputs,
    #     [decoder_outputs] + decoder_states)

    print("########################################", encoder_outputs.shape)
    encoder_outputs = Input(shape=encoder_outputs.shape[1:])

    attention_out = attention_dot([decoder_outputs, encoder_outputs])
    attention_out = attention_softmax(attention_out)

    context_out = context_dot([attention_out, encoder_outputs])
    decoder_combined_context = decoder_concat([context_out, decoder_outputs])

    # Has another weight + tanh layer as described in equation (5) of the paper
    output = output_timedist_1(decoder_combined_context) # equation (5) of the paper
    output = output_timedist_2(output) # equation (6) of the paper


    decoder_model = Model(
        [encoder_outputs, decoder_inputs] + decoder_states_inputs,
        [output] + decoder_states)


    return full_model, encoder_model, decoder_model

def predict_from_scalogram(input_scal, n_output_steps, encoder_model, decoder_model, full_model=None):
        batch_size = input_scal.shape[0]

        # Encode the input as state vectors
        encoder_outputs, states_value_h, states_value_c = encoder_model.predict(input_scal)
        states_value = [states_value_h, states_value_c]

        # Sampling loop
        target_seq = np.zeros((batch_size, 1, 1))
        # target_seq = np.zeros((batch_size, n_output_steps, 1))
        decoded_signal = np.zeros((batch_size, n_output_steps, 1))
        for k in range(n_output_steps):
            output_tokens, h, c = decoder_model.predict(
                [encoder_outputs, target_seq, states_value])

            # output_tokens = full_model.predict([input_scal, target_seq])

            # Sample a value
            print('output_tokens.shape', output_tokens.shape)
            sampled_vals = output_tokens[:, -1, 0]
            # sampled_vals = output_tokens[:, k, 0]
            decoded_signal[:, k, 0] = sampled_vals

            # Update the target sequence (of length 1).
            target_seq = np.zeros((batch_size, 1, 1))
            target_seq[:, 0, 0] = sampled_vals
            # target_seq = np.zeros((batch_size, n_output_steps, 1))
            # target_seq[:, k, 0] = sampled_vals
            # target_seq = np.zeros((batch_size, n_output_steps, 1))
            # target_seq[:, :k+1, 0] = decoded_signal[:, :k+1, 0]

            # Update states
            states_value = [h, c]

        return decoded_signal



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate one or multiple HandRate NN models')
    parser.add_argument('-m', '--model', required=True,
        help='HandRate model to evaluate')
    parser.add_argument('-d', '--data', required=True,
        help='Data file to use during the evaluation')
    parser.add_argument('-s', '--dataset', default=dataset,
        help='Dataset to be used for the evaluation. Valid values here include train, dev (or val), test')
    parser.add_argument('-p', '--plot', default=should_plot, action="store_true",
        help='Plot each prediction made by the model')
    parser.add_argument('-i', '--infreq', default=sample_freq, type=float,
        help='Sampling frequency of the input signal')
    parser.add_argument('-o', '--outfreq', default=out_sample_freq, type=float,
        help='Sampling frequency of the output signal')
    parser.add_argument('-D', '--distance', default=distance, type=float,
        help='Minimum distance (in 100th of second) between two consecutive heartbeat peaks')
    parser.add_argument('-v', '--height', default=height, type=float,
        help='Minimum height of each heartbeat peak')
    parser.add_argument('-P', '--prominence', default=prominence, type=float,
        help='Minimum prominence of each heartbeat peak')
    parser.add_argument('-u', '--units', default=n_units, type=int,
        help='The number of hidden units of the neural net')
    parser.add_argument('-n', '--nseconds', default=n_seconds, type=float,
        help='The number of seconds of a sample')
    args = parser.parse_args()

    should_plot = args.plot
    dataset = args.dataset
    distance = args.distance
    height = args.height
    prominence = args.prominence
    sample_freq = args.infreq
    out_sample_freq = args.outfreq
    model = args.model
    data = args.data
    n_units = args.units
    n_seconds = args.nseconds

    res = evaluate_config(model, data)
    keys = ['IBI_ERROR_MEAN', 'IBI_ERROR_MEDIAN', 'IBI_ERROR_90PRCTILE', 
    'HR_ERROR_MEAN', 'HR_ERROR_MEDIAN', 'HR_ERROR_90PRCTILE',
    'IBI_ERROR', 'HR_ERROR']
    for key in keys:
        print(key, ':', res[key])