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
import os


print("Using Tensorflow", tf.__version__)


users = ['DEMO', 'USER100', 'USER101', 'USER102', 'USER104', 'USER105', 'USER106', 'USER107', 'USER108', 
        'USER1', 'USER2', 'USER3', 'USER5', 'USER6', 'USER7', 'USER8', 'USER9', 'USER']
begin = 4
end = 18


test_datafile = '../matlab/datasets/handrate-dataset-XY-raw-3s-20ones-pca.mat'

n_ones = 20
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
save_dir = 'predictions/'

weighted_binary_crossentropy = None


def evaluate_config(model_filename, X, Y, user):
    # Useful variables
    global dataset
    global save_dir
    save_dir = 'predictions/{:s}'.format(user)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Load the model
    with_conv = 'conv' in model_filename
    with_conv = True
    full_model, encoder_model, decoder_model = get_trained_models(model_filename, with_conv=with_conv)

    # Actual evaluation of each of the items
    n_items = X.shape[0]
    ibi_errs = []
    hr_errs =  []
    gt_hrs = []
    pred_hrs = []
    for item_id in range(n_items):
        x = X[item_id]
        y = Y[item_id]

        print("Evaluating file {:} / {:d} : {:.2f}%".format(item_id+1, n_items, (item_id + 1) * 100 / n_items))
        res = evaluate(full_model, encoder_model, decoder_model, n_seconds, x, y, item_id=item_id)
        if res['IBI_ERROR'] < 300:
            ibi_errs.append(res['IBI_ERROR'])
            hr_errs.append(res['HR_ERROR'])
            gt_hrs.append(res['GROUNDTRUTH_HR'])
            pred_hrs.append(res['PREDICTED_HR'])

    # Global result
    result = {
        'IBI_ERROR': ibi_errs,
        'IBI_ERROR_MEAN': np.mean(ibi_errs),
        'IBI_ERROR_MEDIAN': np.median(ibi_errs),
        # 'IBI_ERROR_90PRCTILE': np.percentile(ibi_errs, 90),

        'HR_ERROR': hr_errs,
        'HR_ERROR_MEAN': np.mean(hr_errs),
        'HR_ERROR_MEDIAN': np.median(hr_errs),
        # 'HR_ERROR_90PRCTILE': np.percentile(hr_errs, 90),
        
        'GROUNDTRUTH_HR': gt_hrs,
        'PREDICTED_HR': pred_hrs,
    }

    return result

def evaluate(full_model, encoder_model, decoder_model, n_seconds, input_scalogram, ground_truth_heartbeats, item_id=0):
    # Useful global variables
    global distance
    global height
    global prominence
    global sample_freq
    global save_dir

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
    np.savetxt('{:s}/gt{:d}.txt'.format(save_dir, item_id), ground_truth)
    np.savetxt('{:s}/pred{:d}.txt'.format(save_dir, item_id), prediction)

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
        axes.imshow(np.squeeze(input_scalogram), aspect="auto")
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
    item_stride_seconds = 1
    # item_stride_seconds = n_seconds
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
    out = np.zeros(end_time_out)
    # out = np.zeros(end_time_out) + 2

    # Reconstruct the output
    for k in range(n_slices):
        begin_time_out = begin_times_out[k]
        end_time_out = begin_time_out + item_duration_out
        # out[begin_time_out:end_time_out] = np.maximum(out[begin_time_out:end_time_out], np.squeeze(predictions[k, :]))
        # out[begin_time_out:end_time_out] = np.minimum(out[begin_time_out:end_time_out], np.squeeze(predictions[k, :]))
        out[begin_time_out:end_time_out] = out[begin_time_out:end_time_out] + np.squeeze(predictions[k, :])
        # out[begin_time_out:end_time_out] = np.squeeze(predictions[k, :])
    out = out / 3;
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
    
    X = np.array(data['scalograms_matrix'])
    Y = np.array(data['heartbeats_matrix'])
    filenames = np.squeeze(data['filenames'])
    users = np.squeeze(data['users'])
    
    return (X, Y, filenames, users)


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

def get_trained_models(model_filename, dropout_rate=0.2, with_conv=False):
    # Useful variables
    input_shape = (int(n_seconds*sample_freq), 75)
    # input_shape = (int(n_seconds*sample_freq), 59)

    
    #################### Encoder network ####################
    encoder_inputs = Input(shape = input_shape, name="input_scalogram")
    X = encoder_inputs
    if with_conv:
        # X = Conv1D(n_units*2, kernel_size=7, strides=1, padding="same")(X)
        X = Conv1D(n_units, kernel_size=7, strides=1, padding="same")(X)
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
    full_model.compile(loss=peak_location_diff, optimizer="rmsprop", metrics=["accuracy"])


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
    parser = argparse.ArgumentParser(description='Evaluate one or multiple HandRate NN models')
    # parser.add_argument('-m', '--model', required=True,
    #     help='HandRate model to evaluate')
    # parser.add_argument('-d', '--data', required=True,
    #     help='Data file to use during the evaluation')
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
    n_units = args.units
    n_seconds = args.nseconds
    

    # Load data from file
    X, Y, filenames, users = load_data(test_datafile)
    users = users[begin:end]
    n_users = len(users)

    for (user_id, user) in enumerate(users):
        user = user[0]
        print('\n\n\n\n')
        print('=======================================================================')
        print('============ Evaluating for user {:d}/{:d}:{:.2f}%  -- {:s} =============='.format(user_id, n_users, (user_id+1)*100/n_users, user))
        print('=======================================================================')

        model = 'handrate.rnn.conv.3s.20ones.128units.teston_' + user + '.h5'
        
        X_user = []
        Y_user = []
        for (k, filename) in enumerate(filenames):
            filename = str(filename[0])
            filename = filename.split('/')[-1]
            if filename.startswith(user + '-'):
                X_user.append(X[:, k])
                Y_user.append(Y[:, k])

        X_user = np.concatenate(X_user)
        Y_user = np.concatenate(Y_user)
        print('X_user.shape', X_user.shape)
        print('Y_user.shape', Y_user.shape)
        

        res = evaluate_config(model, X_user, Y_user, user)
        # keys = ['IBI_ERROR_MEAN', 'IBI_ERROR_MEDIAN', 'IBI_ERROR_90PRCTILE', 
        # 'HR_ERROR_MEAN', 'HR_ERROR_MEDIAN', 'HR_ERROR_90PRCTILE',
        # 'IBI_ERROR', 'HR_ERROR', 'GROUNDTRUTH_HR', 'PREDICTED_HR']
        keys = ['IBI_ERROR_MEAN', 'IBI_ERROR_MEDIAN', 
        'HR_ERROR_MEAN', 'HR_ERROR_MEDIAN',
        'IBI_ERROR', 'HR_ERROR', 'GROUNDTRUTH_HR', 'PREDICTED_HR']
        for key in keys:
            print(key, ':', res[key])