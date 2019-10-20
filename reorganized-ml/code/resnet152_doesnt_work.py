#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from data_generator import *
from keras.layers import Input, Dense, Flatten, Activation, add, ZeroPadding1D, MaxPooling1D, Conv1D, AveragePooling1D, GlobalMaxPooling1D 
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import initializers
from keras.engine import Layer, InputSpec
from keras import backend as K

import sys
sys.setrecursionlimit(3000)

class Scale(Layer):
    '''Custom Layer for ResNet used for BatchNormalization.
    
    Learns a set of weights and biases used for scaling the input data.
    the output consists simply in an element-wise multiplication of the input
    and a sum of a set of constants:

        out = in * gamma + beta,

    where 'gamma' and 'beta' are the weights and biases larned.

    # Arguments
        axis: integer, axis along which to normalize in mode 0. For instance,
            if your input tensor has shape (samples, channels, rows, cols),
            set axis to 1 to normalize per feature map (channels axis).
        momentum: momentum in the computation of the
            exponential average of the mean and standard deviation
            of the data, for feature-wise normalization.
        weights: Initialization weights.
            List of 2 Numpy arrays, with shapes:
            `[(input_shape,), (input_shape,)]`
        beta_init: name of initialization function for shift parameter
            (see [initializers](../initializers.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        gamma_init: name of initialization function for scale parameter (see
            [initializers](../initializers.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
    '''
    def __init__(self, weights=None, axis=-1, momentum = 0.9, beta_init='zero', gamma_init='one', **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        self.gamma = K.variable(self.gamma_init(shape), name='%s_gamma'%self.name)
        self.beta = K.variable(self.beta_init(shape), name='%s_beta'%self.name)
        self.trainable_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def identity_block(input_tensor, kernel_size, filters, stage, block):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv1D(nb_filter1, 1, name=conv_name_base + '2a', padding='same', use_bias=False)(input_tensor)
#    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
#    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

#    x = ZeroPadding1D(1, name=conv_name_base + '2b_zeropadding')(x)
    x = Conv1D(nb_filter2, kernel_size, name=conv_name_base + '2b', padding='same', use_bias=False)(x)
#    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
#    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv1D(nb_filter3, 1, name=conv_name_base + '2c', padding='same', use_bias=False)(x)
#    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
#    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    x = add([x, input_tensor], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=2):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv1D(nb_filter1, 1, strides=strides, name=conv_name_base + '2a', padding='same', use_bias=False)(input_tensor)
#    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
#    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv1D(nb_filter3, 1, name=conv_name_base + '2c', padding='same', use_bias=False)(x)
#    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
#    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    shortcut = Conv1D(nb_filter3, 1, strides=strides,
                             name=conv_name_base + '1', padding='same', use_bias=False)(input_tensor)
#    shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1')(shortcut)
#    shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)

    x = add([x, shortcut], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

def resnet152_model(weights_path=None):
    '''Instantiate the ResNet152 architecture,
    # Arguments
        weights_path: path to pretrained weight file
    # Returns
        A Keras model instance.
    '''
    eps = 1.1e-5

    # Handle Dimension Ordering for different backends
    global bn_axis
    """ 
	if K.image_dim_ordering() == 'tf':
        bn_axis = 3
        img_input = Input(shape=(224, 224, 3), name='data')
    else:
        bn_axis = 1
        img_input = Input(shape=(3, 224, 224), name='data')
    """
    bn_axis = -1  
    signal_input = Input(shape=(800,1), name='data')
#    x = ZeroPadding1D(3, name='conv1_zeropadding')(signal_input)
    x = Conv1D(64, 7, strides=2, name='conv1', use_bias=False)(signal_input)
    x = Conv1D(64, 7, strides=2, name='conv1_again', use_bias=False)(x)
    #x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
#    x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
#    x = MaxPooling1D(3, strides=2, name='pool1')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=1)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
#    for i in range(1,8):
    for i in range(1,3):
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b'+str(i))

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
#    for i in range(1,36):
    for i in range(1,6):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b'+str(i))

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

#    x_fc = AveragePooling1D(7, name='avg_pool')(x)
    x_fc = Flatten()(x)
    #x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)
    x_fc = Dense(1, activation='sigmoid', name='fc1')(x_fc)

    model = Model(signal_input, x_fc)
    
    # load weights
    if weights_path:
        model.load_weights(weights_path, by_name=True)

    return model

model = resnet152_model()
gen = DataReader(numBeforeRepeated=10, edgeCases=False, batch_size=16)
model.compile(optimizer='adamax', loss='binary_crossentropy', metrics=['accuracy'])
model.fit_generator(gen, epochs=5)


