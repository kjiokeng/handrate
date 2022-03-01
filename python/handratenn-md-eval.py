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

# Command line arguments default values
should_plot = False
dataset = 'dev'
distance = 65
height = 0
prominence = 0
sample_freq = 200

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

def evaluate_config(model_filename, datafile):
    # Useful variables
    global dataset

    w0 = (1.2 * n_ones)/100
    w1 = 1 - w0
    custom_loss = create_weighted_binary_crossentropy(w0, w1)
    keras.losses.weighted_binary_crossentropy = custom_loss

    # Load the model
    model = load_model(model_filename)
    model.summary()
    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss=custom_loss, optimizer=opt, metrics=["accuracy"])

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
    print("ssssssssss", X.shape)
    for item_id in range(n_items):
        x = X[0, item_id]
        y = Y[0, item_id]

        res = evaluate(model, n_seconds, x, y)
        if res['IBI_ERROR'] < 20000:
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

def evaluate(model, n_seconds, input_scalogram, ground_truth_heartbeats):
    # Useful global variables
    global distance
    global height
    global prominence
    global sample_freq

    # Make the prediction
    prediction = predict(model, n_seconds, input_scalogram)
    end_time = prediction.shape[0]
    ground_truth = ground_truth_heartbeats[0:end_time, 0]

    # Find peaks
    # gt_peaks, _ = find_peaks(ground_truth, distance=distance*sample_freq/100)
    # pred_peaks, _ = find_peaks(prediction, distance=distance*sample_freq/100, height=height, prominence=prominence)

    gt_peaks, _ = find_peaks(ground_truth, distance=distance*sample_freq/100)
    pred_peaks, _ = find_peaks(prediction, distance=distance*sample_freq/100, height=height, prominence=prominence)

    # Compute statistics: InterBeat Interval, Heart Rate, Errors
    gt_ibi = np.mean(np.diff(gt_peaks)) / sample_freq
    gt_hr = 60 / gt_ibi

    pred_ibi = np.mean(np.diff(pred_peaks)) / sample_freq
    pred_hr = 60 / pred_ibi

    ibi_err = abs(pred_ibi - gt_ibi)
    hr_err = abs(pred_hr - gt_hr)

    # Visualization
    global should_plot
    if should_plot:
        t = range(len(prediction))
        plt.figure(figsize=(20, 5))
        gt_plot, = plt.plot(t, ground_truth, 'b-', label="Ground truth")
        pred_plot, = plt.plot(t, prediction, 'r-', label="Prediction")
        gt_peaks_plot, = plt.plot(gt_peaks, ground_truth[gt_peaks], "bx", label="Ground truth peaks")
        pred_peaks_plot, = plt.plot(pred_peaks, prediction[pred_peaks], "rx", label="Prediction peaks")
        plt.legend(handles=[gt_plot, gt_peaks_plot, pred_plot, pred_peaks_plot])
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

def predict(model, n_seconds, input_scalogram):
    # Useful variables
    input_len, input_height, n_axes = input_scalogram.shape
    item_stride_seconds = max(n_seconds-1, 1)
    # item_stride_seconds = n_seconds
    item_duration = n_seconds * sample_freq
    item_stride = item_stride_seconds * sample_freq

    # Slice the (30s long) data into different items and build the model input
    begin_time = 0
    begin_times = []
    while begin_time + item_duration <= input_len:
        begin_times.append(begin_time)
        begin_time += item_stride

    # Build an input matrix for single-time prediction
    n_slices = len(begin_times)
    inputs = np.zeros((n_slices, item_duration, input_height, n_axes))
    for k in range(n_slices):
        begin_time = begin_times[k]
        end_time = begin_time + item_duration
        inputs[k, :, :, :] = input_scalogram[begin_time:end_time, :, :]

    # Actual prediction
    predictions = model.predict(inputs)
    out = np.zeros(end_time)
    # out = np.zeros(end_time) + 1

    # Reconstruct the output
    for k in range(n_slices):
        begin_time = begin_times[k]
        end_time = begin_time + item_duration
        out[begin_time:end_time] = np.maximum(out[begin_time:end_time], np.squeeze(predictions[k, :]))
        # out[begin_time:end_time] = np.minimum(out[begin_time:end_time], np.squeeze(predictions[k, :]))
        # out[begin_time:end_time] = np.squeeze(predictions[k, :])

    return out

def load_data(filename):
    data = sio.loadmat(filename)
    X_train, Y_train = np.array(data['scalograms_matrix_train']), np.array(data['heartbeats_matrix_train'])
    X_dev, Y_dev = np.array(data['scalograms_matrix_dev']), np.array(data['heartbeats_matrix_dev'])
    X_test, Y_test = np.array(data['scalograms_matrix_test']), np.array(data['heartbeats_matrix_test'])

    return (X_train, Y_train, X_dev, Y_dev, X_test, Y_test)


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
    parser.add_argument('-f', '--frequency', default=sample_freq, type=float,
        help='Sampling frequency of the signal')
    parser.add_argument('-D', '--distance', default=distance, type=float,
        help='Minimum distance (in 100th of second) between two consecutive heartbeat peaks')
    parser.add_argument('-v', '--height', default=height, type=float,
        help='Minimum height of each heartbeat peak')
    parser.add_argument('-P', '--prominence', default=prominence, type=float,
        help='Minimum prominence of each heartbeat peak')
    args = parser.parse_args()

    should_plot = args.plot
    dataset = args.dataset
    distance = args.distance
    height = args.height
    prominence = args.prominence
    sample_freq = args.frequency
    model = args.model
    data = args.data

    evaluate_config(model, data)