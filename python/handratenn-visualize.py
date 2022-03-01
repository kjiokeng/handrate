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


datafile = '../matlab/handrate-dataset-1pts.mat'
model_filename = 'models/handratenn/handrate_model-1pts.h5'
model = None

def main():
    # Load the data
    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = load_data(datafile)
    
    # Expand the output arrays to fit with the model
    Y_train = np.expand_dims(Y_train, axis=2)
    Y_dev = np.expand_dims(Y_dev, axis=2)
    Y_test = np.expand_dims(Y_test, axis=2)

    # Build or load the model
    # model = handrate_model(X_train.shape[1:])
    model = load_model(model_filename)

    # Model visualization
    model.summary()

    # Train the model
    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
    # checkpointer = ModelCheckpoint(filepath=model_filename, verbose=1)
    # hist = model.fit(X_train, Y_train, epochs=500, batch_size=100, callbacks=[checkpointer])

    # Evaluate the model
    loss, acc = model.evaluate(X_dev, Y_dev)
    print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
    
    # Visualize some predictions
    predictions = model.predict(X_dev)
    for i in range(len(predictions)):
        p = predictions[i,:,0]
        gt = Y_dev[i,:,0]
        t = range(300)
        plt.plot(t, gt, 'b-', t, p, 'r--')
        plt.show()



def handrate_model(input_shape):
    """
    Function creating the model's graph in Keras.
    
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """
    X_input = Input(shape = input_shape)
    
    # Step 1: CONV layer
    X = Conv1D(128, kernel_size=7, strides=1, padding="same")(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Dropout(rate=0.2)(X)

    # Step 2: First GRU Layer
    # X = GRU(units = 16, return_sequences = True)(X)
    X = Bidirectional(GRU(units = 128, return_sequences = True))(X)
    X = Dropout(rate=0.2)(X)
    X = BatchNormalization()(X)
    
    # Step 3: Second GRU Layer
    # X = GRU(units = 16, return_sequences = True)(X)
    X = Bidirectional(GRU(units = 128, return_sequences = True))(X)
    X = Dropout(rate=0.2)(X)
    X = BatchNormalization()(X)
    X = Dropout(rate=0.2)(X)
    
    # Step 4: Time-distributed dense layer
    X = TimeDistributed(Dense(1, activation = "sigmoid"))(X)

    model = Model(inputs = X_input, outputs = X)
    
    return model 

def load_data(filename):
	data = sio.loadmat(filename);
	X_train, Y_train = np.array(data['X_train']), np.array(data['Y_train'])
	X_dev, Y_dev = np.array(data['X_dev']), np.array(data['Y_dev'])
	X_test, Y_test = np.array(data['X_test']), np.array(data['Y_test'])

	return (X_train, Y_train, X_dev, Y_dev, X_test, Y_test)


if __name__ == '__main__':
    main()