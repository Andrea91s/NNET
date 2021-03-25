#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 13:39:03 2020

@author: andrea
"""
import numpy as np
import os
import scipy.io as sio
from matplotlib import gridspec
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import Input
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils import get_custom_objects
from tensorflow.python.keras.models import load_model
from scipy import stats


def pure_linear(x):
    return x

get_custom_objects().update({'pure_linear': pure_linear})


EQD_X = np.loadtxt('/home/andrea/DEMO/NNET/DEMO_S1_HNC_06ch_10ms_EQD_X.txt')
EQD_Yn = np.loadtxt('/home/andrea/DEMO/NNET/DEMO_S1_HNC_06ch_10ms_EQD_Yn.txt')
DEMO_total_counts = np.loadtxt('/home/andrea/DEMO/NNET/LIE_counts_06.txt')


a = np.linspace(EQD_X[0], 1, 21)

training = sio.loadmat('DEMO_S1_HNC_06ch_10ms_Training.mat')
validation = sio.loadmat('DEMO_S1_HNC_06ch_10ms_Validation.mat')
test = sio.loadmat('DEMO_S1_HNC_06ch_10ms_Test.mat')


data_Train_Target = training['YNT']    	#targets of the training set
data_Train_Input =  training['LIE']		# inputs of the training set
                      # number of output neurons

data_Validation_Target = validation['YNT']		#% targets of the validation set
data_Validation_Input = validation['LIE'] # inputs of the validation set

data_Test_Target = test['YNT']		# targets of the validation set
data_Test_Input = test['LIE']		# inputs of the validation set


#% ------------------------------------------------------------------
#% Neural Network Layout
#% ------------------------------------------------------------------
#% Define the network layout


data_Train_Input_norm = stats.zscore(data_Train_Input,axis=1)
data_Train_Input_mean = np.mean(data_Train_Input, axis=1)
data_Train_Input_std = np.std(data_Train_Input, axis=1, ddof=1)


data_Train_Target_norm = stats.zscore(data_Train_Target,axis=1)
data_Train_Target_mean = np.mean(data_Train_Target, axis=1)
data_Train_Target_std = np.std(data_Train_Target, axis=1, ddof=1)


data_Validation_Target_norm = stats.zscore(data_Validation_Target,axis=1)
data_Validation_Target_mean = np.mean(data_Validation_Target, axis=1)
data_Validation_Target_std = np.std(data_Validation_Target, axis=1, ddof=1)

data_Validation_Input_norm = stats.zscore(data_Validation_Input,axis=1)
data_Validation_Input_mean = np.mean(data_Validation_Input, axis=1)
data_Validation_Input_std = np.std(data_Validation_Input, axis=1, ddof=1)

data_Test_Target_norm = stats.zscore(data_Test_Target,axis=1)
data_Test_Target_mean = np.mean(data_Test_Target, axis=1)
data_Test_Target_std = np.std(data_Test_Target, axis=1, ddof=1)

data_Test_Input_norm = stats.zscore(data_Test_Input,axis=1)
data_Test_Input_mean = np.mean(data_Test_Input, axis=1)
data_Test_Input_std = np.std(data_Test_Input, axis=1, ddof=1)       
#
# 
model_filename = "Model_DEMO.h5"

if os.path.isfile(model_filename):

    model = load_model(model_filename)

    print("Loaded model")

else:

    print('Create new model')
 



model = Sequential()
model.add(Input(shape=(6,)))
model.add(Dense(20,activation='tanh', name="hidden_layer1"))
model.add(Dense(20,activation='tanh', name="hidden_layer2"))
model.add(Dense(21, activation='pure_linear', name="final_layer"))

model.compile(loss='MeanSquaredError', optimizer= Adam(), metrics=['accuracy'])
model.summary()

callback1 = ModelCheckpoint(model_filename, save_best_only=True, verbose=True)
callback2 = EarlyStopping(monitor='loss', patience=500)

history = model.fit(data_Train_Input_norm, data_Train_Target_norm, 
                   validation_data=(data_Validation_Input_norm, data_Validation_Target_norm), 
                   epochs=2000, batch_size=20, callbacks=[callback1, callback2])


plt.figure(1)
plt.subplot(1,2,1)
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



prediction_test_norm = model.predict(data_Test_Input_norm)
prediction_test = np.zeros((2090,21))
for i in range(0,21):
    prediction_test[:,i] = prediction_test_norm[:,i]* data_Test_Target_std + data_Test_Target_mean


plt.figure(2)
plt.plot(a, 100*(1-prediction_test.T/data_Test_Target.T),'k--')
plt.xlabel('normalized minor radius', fontsize=20)
plt.ylabel('(%)', fontsize=20)
plt.title('Test data')
plt.ylim(-10, 100)

plt.close('all')

error = 100*(1-prediction_test.T/data_Test_Target.T)
error[error == np.inf] = 1e9
runs = np.linspace(1,2090,2090).T

plt.figure(100)
plt.pcolor(a, runs,error.T)
plt.colorbar()


DEMO_targets = np.interp(a,EQD_X, EQD_Yn)

DEMO_target_norm = stats.zscore(DEMO_targets)
DEMO_target_mean= np.mean(DEMO_targets)
DEMO_target_std = np.std(DEMO_targets, ddof=1)

DEMO_total_counts_norm = stats.zscore(DEMO_total_counts)
DEMO_total_counts_mean= np.mean(DEMO_total_counts)
DEMO_total_counts_std = np.std(DEMO_total_counts, ddof=1)



NET_target_norm = model.predict(DEMO_total_counts_norm.reshape(-1, 6))

NET_target = NET_target_norm*DEMO_target_std + DEMO_target_mean

plt.figure(3,figsize = (17, 10))
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1]) 
ax0 = plt.subplot(gs[0])
plt.subplot(gs[0])
plt.plot(a,DEMO_target_norm.T,'r')
plt.plot(a,NET_target_norm.T,'b')
plt.ylabel('Normalized emissivity', fontsize=14)
plt.legend(['DEMO','NNET'],fontsize=14,frameon=False)

plt.subplot(gs[1], sharex = ax0)
plt.plot(a,DEMO_targets.T,'r')
plt.plot(a,NET_target.T,'b')

plt.ylabel('emissivity (1/m3/s)', fontsize=14)
plt.legend(['DEMO','NNET'],fontsize=14,frameon=False)


plt.subplot(gs[2], sharex = ax0)
plt.plot(a,100*(1-NET_target.T/DEMO_targets.reshape(21, 1)),'k--')
plt.ylabel('(%)', fontsize=14)
plt.ylim(-20, 20)
plt.xlabel('normalized minor radius', fontsize=14)
#



