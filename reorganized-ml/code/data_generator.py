#!/bin/python3

# In[1]:


import numpy as np
import keras
from os import system as sys


# In[3]:


def fft_trapezoid(length, rise_time, flat_top , tau):
    length = length+2*rise_time+flat_top;
    p = np.zeros(length)
    s = np.zeros(length)
    input2 = np.zeros(length)
    p[0] = s[0] = input2[0] = 0.;
    for i in range(1,length):
        input2[i] = s[i] = 0.;
    input2[1]=1.;
    tau = 1/(np.exp(1./tau)-1);
    for i in range(1,length):
        if i>=2*rise_time+flat_top:
            d = input2[i]-input2[i-rise_time]-input2[i-rise_time-flat_top]+input2[i-2*rise_time-flat_top]
        else:
            if i>=rise_time+flat_top:
                d = input2[i]-input2[i-rise_time]-input2[i-rise_time-flat_top]
            else:
                if i>=rise_time:
                    d = input2[i]-input2[i-rise_time]
                else:
                    d = input2[i];
        p[i] = p[i-1]+d;
        s[i] = s[i-1]+p[i]+tau*d;
    for i in range(length):
        s[i] = s[i]/(rise_time*tau);
    
    res = np.fft.rfft(s)
    return res[:-(flat_top+rise_time)]


# In[4]:
class DataReader(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, batch_size=32, dim=(400,1), numBeforeRepeated=5, n_channels=1, n_classes=1):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.epoch = 0
        self.numBeforeRepeated = numBeforeRepeated 
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.X)/self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate data
        X, y = self.__data_generation()
        return X, y
    
    def shuffle_in_unison(self, a, b):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        
    def on_epoch_end(self):
        if (self.epoch % self.numBeforeRepeated == 0):
            num = int(self.epoch/self.numBeforeRepeated)
            'Updates indexes after each epoch'
            my_data = np.loadtxt('../data/'+str(num+1)+'.dat')
            my_labels = np.append(np.zeros(int(len(my_data)/2)),np.ones(int(len(my_data)/2)))
            trap=fft_trapezoid(3500, 5, 0, 1250)
            self.X = np.empty((len(my_data), 400, 1))
            for i in range(len(my_data)):
                my_data[i] -= np.mean(my_data[i][:800]) 
                my_data[i] = my_data[i]/max(my_data[i])
                #self.X[i][:,1] = np.fft.irfft(trap*np.fft.rfft(my_data[i]))[800:1200]
                self.X[i][:,0] = my_data[i][800:1200]
            #self.X = self.X.reshape(len(my_data), 400, 2)
            #y = convert_to_one_hot(my_labels.astype(int64), 2).T
            self.y = my_labels            
        self.shuffle_in_unison(self.X, self.y)
        self.index=0
        self.epoch += 1

    def __data_generation(self):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X_tmp = self.X[self.index*self.batch_size:(self.index+1)*self.batch_size]
        y_tmp = self.y[self.index*self.batch_size:(self.index+1)*self.batch_size]
        self.index += 1
        return X_tmp, y_tmp



class ValidationGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, batch_size=32, dim=(400,1), numBeforeRepeated=1, n_channels=1, n_classes=1):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.epoch = 0
        self.numBeforeRepeated = numBeforeRepeated 
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.X)/self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate data
        X, y = self.__data_generation()
        return X, y
    
    def shuffle_in_unison(self, a, b):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        
    def on_epoch_end(self):
        if (self.epoch % self.numBeforeRepeated == 0):
            num = int(self.epoch/self.numBeforeRepeated)
            'Updates indexes after each epoch'
            my_data = np.loadtxt('../data/val'+str(num+1)+'.dat')
            my_labels = np.append(np.zeros(int(len(my_data)/3)),np.ones(int(2*len(my_data)/3)))
            trap=fft_trapezoid(3500, 5, 0, 1250)
            self.X = np.empty((len(my_data), 400, 1))
            for i in range(len(my_data)):
                my_data[i] -= np.mean(my_data[i][:800]) 
                my_data[i] = my_data[i]/max(my_data[i])
                #self.X[i][:,1] = np.fft.irfft(trap*np.fft.rfft(my_data[i]))[800:1200]
                self.X[i][:,0] = my_data[i][800:1200]
            #self.X = self.X.reshape(len(my_data), self.dim[0], 2)
            #y = convert_to_one_hot(my_labels.astype(int64), 2).T
            self.y = my_labels            
        self.shuffle_in_unison(self.X, self.y)
        self.index=0
        self.epoch += 1
        

    def __data_generation(self):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X_tmp = self.X[self.index*self.batch_size:(self.index+1)*self.batch_size]
        y_tmp = self.y[self.index*self.batch_size:(self.index+1)*self.batch_size]
        self.index += 1
        return X_tmp, y_tmp





