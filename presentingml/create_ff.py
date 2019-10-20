#!/usr/bin/python3

from os import system as sys
from array import array as fileExtractor
from numpy import arange, empty, ones, zeros, float64, mean, load, save, array, rint
from keras.models import load_model
from matplotlib import pyplot as plt
from time import sleep
import keras.backend as K
from sklearn.metrics import accuracy_score

simWF_id = '1XevoOZFumlfF81_amlZ-L6elPTNteyku'

def fileNameGen(delay,percent):
    return str(delay)+'nsD_'+str(percent)+'P.dat'

def folderNameGen(energy):
	return str(energy)+'adc_gausR_5usF_randNS_varP_10K'

m = load_model('occam3.h5')
records = empty(10)

for i, energy in enumerate(arange(100,1100,100)):
#    for j, percent in enumerate(arange(10,100,10)):
#        for k,delay in enumerate(arange(20,200,20)):
            X = empty((int(1e4),400))
            try: 
                with open('/media/david/backups/37adcSim/'+folderNameGen(energy)+'/noP.dat','rb') as file:
                    data = file.read()
            except:
                records[i] = float('nan') 
                continue
            for l in range(1,int(1e4)+1):
                tmp = fileExtractor('h',data[8+33*l+7000*(l-1):8+7033*l])
                tmp -= mean(tmp[:800])
                X[l-1] = tmp[800:1200]
                X[l-1] = X[l-1]/max(X[l-1])
            X = X.reshape((len(X),400,1)).astype(float64)
            records[i] = m.evaluate(X, zeros((len(X),1)), verbose=True)[1]
save('ff3.npy',records)
#			acc = m.evaluate(X, zeros((len(X),1)), verbose=True)[1:]
#            acc = m.evaluate(X, zeros((len(X),1)), verbose=True)[1:]
#            print(acc)
#            records[i] = acc

