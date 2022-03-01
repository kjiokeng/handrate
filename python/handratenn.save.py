import numpy as np
import scipy.io as sio
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

import matplotlib.pyplot as plt
import pickle
from memory_profiler import profile


n_points = 3
datafile = '../matlab/handrate-dataset-pca-' + str(n_points) + 'pts.mat'
model_filename = 'models/handratenn-pca/handrate_model-pca-' + str(n_points) + 'pts.h5'
# model_filename = 'models/handratenn-pca/handrate_model-pca-' + str(10) + 'pts.h5'
model_checkpoint_filename = model_filename
model_checkpoint_filename = 'models/handratenn-pca/handrate_model-pca-v2-' + str(n_points) + 'pts-epoch={epoch:02d}-loss={loss:.4f}-acc={accuracy:.4f}.h5'
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
    # opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
    # opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.01)
    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
    checkpointer = ModelCheckpoint(filepath=model_checkpoint_filename, verbose=1)
    hist = model.fit(X_train, Y_train, epochs=200, batch_size=100, callbacks=[checkpointer])

    # Evaluate the model
    loss, acc = model.evaluate(X_dev, Y_dev)
    print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
    
    # Visualize some predictions
    predictions = model.predict(X_train)
    for i in range(len(predictions)):
        p = predictions[i,:,0]
        gt = Y_dev[i,:,0]
        t = range(300)
        plt.plot(t, gt, 'b-', t, p, 'r--')
        plt.show()


@profile
def learning_curves():
    # Variable file
    n_points = 10
    datafile = '../matlab/handrate-dataset-pca-' + str(n_points) + 'pts.mat'
    model_filename = 'models/handratenn-pca/handrate_model-pca-' + str(n_points) + 'pts.h5'
    # model_filename = 'models/handratenn-pca/handrate_model-pca-' + str(10) + 'pts.h5'
    model_checkpoint_filename = model_filename
    model_checkpoint_filename = 'models/handratenn-pca/handrate_model-pca-v2-' + str(n_points) + 'pts-epoch={epoch:02d}-loss={loss:.4f}-acc={accuracy:.4f}.h5'
    model = None
    n_epochs = 1000
    variables_filename = 'variables-pca-{:d}pts.pkl'.format(n_points)

    # Load the data
    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = load_data(datafile)
    
    # Expand the output arrays to fit with the model
    Y_train = np.expand_dims(Y_train, axis=2)
    Y_dev = np.expand_dims(Y_dev, axis=2)
    Y_test = np.expand_dims(Y_test, axis=2)

    # Evaluate the different models on train and dev sets
    losses_train = []
    losses_dev = []
    epochs = range(1, n_epochs+1);
    for epoch in epochs:
        # Build or load the model
        model_filename = 'models/handratenn-pca/handrate_model-pca-' + \
            '{:d}pts-epoch={:02d}.h5'.format(n_points, epoch)
        print(model_filename)
        model = load_model(model_filename)
        opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

        # Evaluate the model
        loss_train, acc_train = model.evaluate(X_train, Y_train)
        loss_dev, acc_dev = model.evaluate(X_dev, Y_dev)
        K.clear_session()
        print(('Epoch = {:d}; Train loss = {:.4f}; Dev loss = {:.4f};' + \
            ' Train acc= {:.4f}; Dev acc= {:.4f}').format(epoch, loss_train, loss_dev, acc_train, acc_dev))

        losses_train.append(loss_train)
        losses_dev.append(loss_dev)

    # Saving the objects:
    with open(variables_filename, 'wb') as f:
        pickle.dump([losses_train, losses_dev], f)

    plt.plot(epochs, losses_train, 'b-', epochs, losses_dev, 'r--')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train", "Dev"])
    plt.show()

    # Getting back the objects:
    # with open(variables_filename, 'rb') as f:
    #     losses_train, losses_dev = pickle.load(f)



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
    # main()
    learning_curves()