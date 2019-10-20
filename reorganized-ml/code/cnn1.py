#!/usr/bin/python3

import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, Concatenate, BatchNormalization, Flatten, Conv1D, TimeDistributed, LSTM
from keras.models import Model, load_model
from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
from data_generator import *
from numpy import zeros, ones, append, int64, loadtxt, random
from os import system as sys


def convolutional_block(X, F1):
    # Save the input value
    X_path1 = X
    X_path2 = X
    X_path3 = X
    X_identity = X
    
    # path 1
    X_path1 = Conv1D(F1, 3, strides = 1, padding='same', kernel_initializer = glorot_uniform(seed=0))(X_path1)
    X_path1 = BatchNormalization()(X_path1)

    # path 2 
    X_path2 = Conv1D(F1, 5, strides = 1, padding='same', kernel_initializer = glorot_uniform(seed=0))(X_path2)
    X_path2 = BatchNormalization()(X_path2)
 
    # path 3 
    X_path3 = Conv1D(F1, 7, strides = 1, padding='same', kernel_initializer = glorot_uniform(seed=0))(X_path3)
    X_path3 = BatchNormalization()(X_path3)

    X = Concatenate()([X_identity, X_path1, X_path2, X_path3])
    X = Activation('relu')(X)
	
    return X

def multiPath(input_shape=(400, 2), classes=2):
    X_input = Input(input_shape)
    X = BatchNormalization()(X_input)
    X = convolutional_block(X, 16)
    X = convolutional_block(X, 32)
    X = convolutional_block(X, 64)
    X = convolutional_block(X, 128)
    #X = TimeDistributed(Flatten())(X)
    #X = LSTM(256)(X) 
    X = Flatten()(X) 
    X = Dense(1, activation='sigmoid', kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model


class EarlyStoppingByAccuracy(keras.callbacks.Callback):
    def __init__(self, monitor='val_acc', value=0.91, verbose=0):
        super(keras.callbacks.Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current >= self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


cb = EarlyStoppingByAccuracy(monitor='val_acc', value=0.91, verbose=1) 
model = multiPath(input_shape = (400, 2), classes = 2)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
val = ValidationGenerator(numBeforeRepeated=4) 

gen = DataReader(numBeforeRepeated=25, batch_size=32)
model.fit_generator(gen, epochs = 50)
model.save('junk5.h5')

gen = DataReader(numBeforeRepeated=10, batch_size=32)
model.fit_generator(gen, epochs = 20)
model.save('junk6.h5')

gen = DataReader(numBeforeRepeated=5, batch_size=32)
model.fit_generator(gen, epochs = 25)
model.save('junk7.h5')

gen = DataReader(numBeforeRepeated=2, batch_size=32)
model.fit_generator(gen, validation_data=val, epochs = 100, callbacks=[cb])
model.save('junk8.h5')


