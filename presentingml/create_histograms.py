#!/usr/bin/python3

from os import system as sys
from array import array as fileExtractor
from numpy import arange, empty, ones, zeros, float64, mean, load, save, array
from keras.models import load_model
from matplotlib import pyplot as plt
from time import sleep

simWF_id = '1XevoOZFumlfF81_amlZ-L6elPTNteyku'

"""
folder_ids = {}
for energy in range(50,2500,50):
    folderName = str(energy)+'adc_52nsR_varF_2016NS_varP_10K'
    sys('./gdrive-linux-x64 list --no-header --query "name contains \''+folderName+'\'" > junk')
    with open('junk','r') as file:
        data = file.read().split()
        if len(data) != 5:
            print('error')
        else:
            folder_ids[energy] = data[0]

import csv

w = csv.writer(open("folder_ids.csv", "w"))
for key, val in folder_ids.items():
    w.writerow([key, val])

folder_ids = {}
with open('folder_ids.csv','r') as file:
    for line in file:
        folder_ids[int(line.strip().split('acd')[0].split(',')[0])] = line.strip().split(',')[1]
def getFileIDs(parent):
    while(True):
        result = sys('./gdrive-linux-x64 list --no-header --query "\''+parent+'\' in parents" > junk')
        if result != 0:
            print('error downloading: ',result)
            continue
        with open('junk', 'r') as file:
            data = array(file.read().split())
            if (len(data)<21):
                print('error with output: ',data)
                sleep(2)
            else:
                data = data.reshape((int(len(data)/7),7))
                sys('rm junk')
                return data[:,0]

def downloadFiles(file_ids):
    for name in file_ids:
        while(True):
            res = sys('./gdrive-linux-x64 download --force '+name+' > junk')
            if res != 0:
                print('error downloading: ',res)
                continue
            with open('junk','r') as file:
                data = file.read().split()
                if len(data) != 12:
                    continue
                else:
                    sys('rm junk')
                    break

""";

def fileNameGen(delay,percent):
    return str(delay)+'nsD_'+str(percent)+'P.dat'

def folderNameGen(energy):
	return str(energy)+'adc_gausR_5usF_randNS_varP_10K'

m = load_model('occam3.h5')
records = empty((10,9,7))
for i, energy in enumerate(arange(100,1100,100)):
    for j, percent in enumerate(arange(10,100,10)):
        for k,delay in enumerate(arange(20,160,20)):
            X = empty((int(1e4),400))
            try: 
                with open('/media/david/backups/37adcSim/'+folderNameGen(energy)+'/'+fileNameGen(delay,percent),'rb') as file:
                    data = file.read()
            except:
                records[i][j][k] = float('nan') 
                continue
            for l in range(1,int(1e4)+1):
                tmp = fileExtractor('h',data[8+33*l+7000*(l-1):8+7033*l])
                tmp -= mean(tmp[:800])
                X[l-1] = tmp[800:1200]
                X[l-1] = X[l-1]/max(X[l-1])
            X = X.reshape((len(X),400,1)).astype(float64)
            acc = m.evaluate(X, ones((len(X),1)), verbose=True)[1]
            records[i][j][k] = acc

save('records3.npy',records)

