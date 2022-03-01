import numpy as np
import scipy.io as sio
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Conv1D, Activation, MaxPooling1D, Dropout, Input, Masking, TimeDistributed
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.utils import to_categorical
from keras import optimizers
import keras.backend as K
import tensorflow as tf


def main():
    # Load the data
    # X_train, Y_train, X_dev, Y_dev, X_test, Y_test = load_data('../matlab/heartbeatdata.mat')
    # X_train, Y_train, X_dev, Y_dev, X_test, Y_test = load_data('../matlab/heartbeatdata-all.mat')
    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = load_data('../matlab/heartbeatdata-all-spectrograms.mat')
    
    # Infer number of features and number of angles
    input_size = X_train.shape[1]
    input_shape = X_train.shape[1:]

    n_users = np.unique(np.concatenate((Y_train, Y_dev, Y_test))).size

    # Convert the targets to one-hot vectors
    Y_train = to_categorical(Y_train, num_classes=n_users)
    Y_dev = to_categorical(Y_dev, num_classes=n_users)
    Y_test = to_categorical(Y_test, num_classes=n_users)

    # Build and train the model
    # model = heartbeatnn_model_classif(input_size, n_users)
    model = heartbeatnn_model_recurrent(input_shape, n_users)
    # model = heartbeatnn_model_reg(input_size, n_users)
    # model = heartbeatnn_model_conv(X_train)

    # opt = optimizers.Adam(learning_rate=0.1)
    opt = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
    # model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
    
    model.summary()
    model.fit(X_train, Y_train, epochs=250, batch_size=1024)
    
    # loss, acc = model.evaluate(X_train, Y_train)
    loss, acc = model.evaluate(X_dev, Y_dev)
    print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
    # print('loss_and_metrics', loss_and_metrics)
    
    # predictions = model.predict(X_train)
    # for i in range(len(predictions)):
    # 	print('True value=', Y_train[i], 'Prediction=', predictions[i])

    # model.summary()
    # weights = model.get_weights()
    # print(weights)


def heartbeatnn_model_classif(input_size, n_users):
	model = Sequential()
	model.add(Dense(100, input_dim=input_size))
	model.add(Dense(n_users, activation="softmax"))

	return model

def heartbeatnn_model_reg(input_size, n_users):
	model = Sequential()
	model.add(Dense(1000, input_dim=input_size, activation="tanh"))
	model.add(Dense(1000, activation="sigmoid"))
	model.add(Dense(1000, activation="tanh"))
	model.add(Dense(1, activation="relu"))

	return model

def heartbeatnn_model_conv(X_train):
	model = Sequential()
	model.add(Conv1D(32, 3, padding='same',
                 input_shape=X_train.shape[1:]))
	model.add(Activation('relu'))
	model.add(Conv1D(32, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling1D(pool_size=2))
	# model.add(Dropout(0.25))

	model.add(Conv1D(64, 3, padding='same'))
	model.add(Activation('relu'))
	model.add(Conv1D(64, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling1D(pool_size=2))
	# model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	# model.add(Dropout(0.5))
	model.add(Dense(n_users))
	model.add(Activation('softmax'))

	return model

def heartbeatnn_model_recurrent(input_shape, n_users):
    """
    Function creating the model's graph in Keras.
    
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """
    
    X_input = Input(shape = input_shape)
    
    ### START CODE HERE ###
    
    # Step 1: CONV layer (≈4 lines)
    X = Conv1D(196, kernel_size=15, strides=2)(X_input)              # CONV1D
    X = BatchNormalization()(X)                                 # Batch normalization
    X = Activation('relu')(X)                                 # ReLu activation
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)

    # Step 2: First GRU Layer (≈4 lines)
    X = GRU(units = 128, return_sequences = True)(X)                                 # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)
    X = BatchNormalization()(X)                                 # Batch normalization
    
    # Step 3: Second GRU Layer (≈4 lines)
    X = GRU(units = 128, return_sequences = False)(X)                                 # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)
    X = BatchNormalization()(X)                                 # Batch normalization
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)
    
    # Step 4: Time-distributed dense layer (≈1 line)
    # X = TimeDistributed(Dense(n_users, activation = "sigmoid"))(X) # time distributed  (sigmoid)
    X = Dense(n_users, activation = "sigmoid")(X) # dense (sigmoid)

    ### END CODE HERE ###

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