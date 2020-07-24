#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 11:29:01 2019

@author: chuyangye
"""
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers.advanced_activations import ThresholdedReLU
from keras.layers import Dense, Dropout, Input, Activation, add, multiply, Conv3D, AveragePooling3D, Flatten, Reshape, Conv2D, concatenate
from keras import regularizers
#from keras.backend import concatenate
import numpy as np
import os
import pickle
#%%
def espcn_srep(input_shape_dwi, input_shape_T1, featurenumbers, upsample, nDict, nChannels1, nChannels2):

    #nDict = 301 # 578,363 weights
    
    #alpha=1
    nLayers1 = 8
    
    input_dwi = Input(shape=input_shape_dwi) #Tensor("input_1:0", shape=(?, 5, 5, 5, 36), dtype=float32)
    input_T1 = Input(shape=input_shape_T1)  #Tensor("input_2:0", shape=(?, 5, 5, 5, 8), dtype=float32) 

    W = Conv3D(filters=nDict, kernel_size=(1, 1, 1), activation='relu')(input_dwi)
    
    TS_input_shape = list(input_shape_dwi)
    TS_input_shape[0] = input_shape_dwi[0] 
    TS_input_shape[1] = input_shape_dwi[1]
    TS_input_shape[2] = input_shape_dwi[2]
    TS_input_shape[3] = nDict
    TS_input_shape = tuple(TS_input_shape)
    ReLUThres = 0.01
    TS_input = Input(shape=TS_input_shape)
    TS_thresReLu = ThresholdedReLU(theta = ReLUThres)(TS_input)
    TS_output = Conv3D(filters=nDict, kernel_size=(1, 1, 1), activation='linear')(TS_thresReLu)
    
    TS = Model(outputs = TS_output , inputs = TS_input)
    
    b = TS(W)
    
    for l in range(nLayers1-1):
        c = add([W,b])
        b = TS(c)
    y = add([W,b])
    x = ThresholdedReLU(theta = ReLUThres)(y)
    print "component one output:", x.shape
    
    xT1=concatenate([x, input_T1], axis = 4)
    print "component two input:", xT1.shape
    #nChannels1 = 50
    #nChannels2 = 100
    H_input_shape = list(input_shape_dwi)
    H_input_shape[0] = input_shape_dwi[0] 
    H_input_shape[1] = input_shape_dwi[1]
    H_input_shape[2] = input_shape_dwi[2]
    H_input_shape[3] = nDict+np.power(upsample, 3)
    H_input_shape = tuple(H_input_shape)
    H_input = Input(shape =  H_input_shape) 
    H_1 = Conv3D(filters=nChannels1, kernel_size=(3, 3, 3), activation='relu')(H_input)
    H_2 = Dropout(0.1)(H_1)
    H_3 = Conv3D(filters=nChannels2, kernel_size=(1, 1, 1), activation='relu')(H_2)
    H_4 = Dropout(0.1)(H_3)
    H_output = Conv3D(filters=featurenumbers*np.power(upsample, 3), kernel_size=(3, 3, 3), activation='relu')(H_4)
    
    H = Model(outputs = H_output , inputs = H_input)
        
    outputs = H(xT1)
    
    ### fitting the model ###                    
    print "Fitting"    
    
    regressor = Model(inputs=[input_dwi,input_T1],outputs=outputs)
    regressor.compile(optimizer=Adam(lr=0.0001), loss='mse')
    print regressor.summary()
    return regressor
#%%
def espcn(input_shape, featurenumbers, upsample):
    nChannels1 = 50
    nChannels2 = 100 # 118,574 weights

    inputs = Input(shape=input_shape)
    C = Sequential() 
    
    C.add(Conv3D(filters=nChannels1, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape))
    C.add(Dropout(0.1))
    C.add(Conv3D(filters=nChannels2, kernel_size=(1, 1, 1), activation='relu'))
    C.add(Dropout(0.1))
    C.add(Conv3D(filters=featurenumbers*np.power(upsample, 3), kernel_size=(3, 3, 3), activation='relu'))
    
    outputs = C(inputs)
    
    ### fitting the model ###                    
    print "Fitting"    
    
    regressor = Model(inputs=inputs,outputs=outputs)
    regressor.compile(optimizer=Adam(lr=0.0001), loss='mse')
    print regressor.summary()
    return regressor
#%%
def espcn_wide(input_shape, featurenumbers, upsample, nChannels1, nChannels2):
    #nChannels1 = 215
    #nChannels2 = 430 # 580,739 weights

    inputs = Input(shape=input_shape)
    C = Sequential() 
    
    C.add(Conv3D(filters=nChannels1, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape))
    C.add(Dropout(0.1))
    C.add(Conv3D(filters=nChannels2, kernel_size=(1, 1, 1), activation='relu'))
    C.add(Dropout(0.1))
    C.add(Conv3D(filters=featurenumbers*np.power(upsample, 3), kernel_size=(3, 3, 3), activation='relu'))
    
    outputs = C(inputs)
    
    ### fitting the model ###                    
    print "Fitting"    
    
    regressor = Model(inputs=inputs,outputs=outputs)
    regressor.compile(optimizer=Adam(lr=0.0001), loss='mse')
    print regressor.summary()
    return regressor
