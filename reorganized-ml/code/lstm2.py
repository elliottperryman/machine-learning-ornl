#!/usr/bin/python3
# coding: utf-8

# ## Residual Networks

# In[1]:


import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers import ZeroPadding1D, Conv1D, AveragePooling1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers import LSTM
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
#get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pyplot import *

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)


from data_generator import *
# In[2]:


from numpy import zeros, ones, append, int64, loadtxt, random


# In[3]:


from os import system as sys


# In[4]:


def ResNet50(input_shape=(1000, 3), classes=6):
    X_input = Input(input_shape)
    X = BatchNormalization()(X_input)
    X = LSTM(1000)(X)   
	X = Dense(1, activation='sigmoid', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
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
model = ResNet50(input_shape = (400, 1), classes = 2)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
val = ValidationGenerator(numBeforeRepeated=4) #gen = DataReader(numBeforeRepeated=5, batch_size=32, dataset='geo')

gen = DataReader(numBeforeRepeated=10, batch_size=32)
model.fit_generator(gen, epochs = 20)

gen = DataReader(numBeforeRepeated=10, batch_size=32)
model.fit_generator(gen, validation_data=val, epochs = 100, callbacks=[cb])

model.save('no_trap.h5')


