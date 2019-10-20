#!/usr/bin/python3

# coding: utf-8

# In[1]:


from keras import callbacks
from keras.models import Model, load_model
from keras.layers import Input, Conv1D, Flatten, BatchNormalization, Activation, Add, Dense, MaxPooling1D
from keras.optimizers import Adam, SGD
from keras.initializers import glorot_uniform
from data_generator import *
from os import system as sys


# In[2]:


def conv(X, filters, k_size):
    X = Conv1D(filters, k_size, padding='same', kernel_initializer = glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)
    return X


# In[3]:


def occam_cnn(F1, K1, input_shape=(400,1)):
    X_in = Input(input_shape)
    X = BatchNormalization()(X_in)
    X = conv(X_in, F1, K1)
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid')(X)
    m = Model(inputs=X_in, outputs=X, name="Occam's CNN")
    o = Adam()
    m.compile(optimizer=o, loss='binary_crossentropy', metrics=['accuracy'])
    return m


# In[4]:


reduce_lr = callbacks.ReduceLROnPlateau(monitor='acc', factor=0.5, patience=5)


# In[5]:


def eval_model(model):
    val = ValidationGenerator()
    return model.evaluate_generator(val, verbose=False)[1]


# In[6]:


# Gen model
model = occam_cnn(128, 64)


# In[7]:


# Gen model
#model = load_model('occam.h5')


# In[9]:


## Training Montage

# Rough Training, no callbacks, just running over large amounts of data
gen = DataReader(numBeforeRepeated=25, batch_size=16)
model.fit_generator(gen, epochs = 25*100, verbose=False);


# In[ ]:


gen = DataReader(numBeforeRepeated=10, batch_size=32)
model.fit_generator(gen, epochs = 10*100, verbose=False);


# In[ ]:


gen = DataReader(numBeforeRepeated=10, batch_size=32)
model.fit_generator(gen, epochs = 10*100, verbose=False);


# In[ ]:


# Intermediate Training increase batch size, decrease times each wave is seen, decrease learning rate
sys('~/Documents/ML/make_training.sh & sleep 2 ');
gen = DataReader(numBeforeRepeated=5, batch_size=32)
model.fit_generator(gen, epochs = 5*100, verbose=False);

# In[ ]:

sys('~/Documents/ML/make_training.sh & sleep 2 ');
gen = DataReader(numBeforeRepeated=2, batch_size=32)
model.fit_generator(gen, epochs = 2*100, verbose=False);


# In[ ]:

sys('~/Documents/ML/make_training.sh & sleep 2 ');
gen = DataReader(numBeforeRepeated=2, batch_size=64)
model.fit_generator(gen, epochs = 2*100, verbose=False, callbacks=[reduce_lr]);


# In[ ]:


sys('~/Documents/ML/make_training.sh & sleep 2 ');
gen = DataReader(numBeforeRepeated=2, batch_size=128)
model.fit_generator(gen, epochs = 2*100, verbose=False);


# In[ ]:


sys('~/Documents/ML/make_training.sh & sleep 2 ');
gen = DataReader(numBeforeRepeated=2, batch_size=256)
model.fit_generator(gen, epochs = 2*100, verbose=False);


# In[ ]:


sys('~/Documents/ML/make_training.sh & sleep 2 ');
gen = DataReader(numBeforeRepeated=1, batch_size=256)
model.fit_generator(gen, epochs = 100, verbose=False, callbacks=[reduce_lr]);

sys('~/Documents/ML/make_training.sh & sleep 2 ');
gen = DataReader(numBeforeRepeated=1, batch_size=256)
model.fit_generator(gen, epochs = 100, verbose=False, callbacks=[reduce_lr]);


sys('~/Documents/ML/make_training.sh & sleep 2 ');
gen = DataReader(numBeforeRepeated=1, batch_size=256)
model.fit_generator(gen, epochs = 100, verbose=False, callbacks=[reduce_lr]);

sys('~/Documents/ML/make_training.sh & sleep 2 ');
gen = DataReader(numBeforeRepeated=1, batch_size=256)
model.fit_generator(gen, epochs = 100, verbose=False, callbacks=[reduce_lr]);


sys('~/Documents/ML/make_training.sh & sleep 2 ');
gen = DataReader(numBeforeRepeated=1, batch_size=256)
model.fit_generator(gen, epochs = 100, verbose=False, callbacks=[reduce_lr]);



# In[ ]:

# #### Best F1 and F2 values are those that are largest (128, 32) - > 

# In[ ]:


eval_model(model)


# In[ ]:


model.save('occam3.h5')


# In[ ]:




