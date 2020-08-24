#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 11:29:01 2019

@author: chuyangye
"""
from keras import backend as K
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers.advanced_activations import ThresholdedReLU
from keras.layers import Dense, Dropout, Input, Activation, concatenate, add, multiply, Conv3D, AveragePooling3D, Flatten, Reshape, Conv2D
from keras import regularizers
import numpy as np
import os
import pickle
####
def scoring(y_true, y_pred):
    tau = 1e-6
    mean = y_pred[:,:,:,:,:24]
    var = y_pred[:,:,:,:,24:]
    gt = y_true[:,:,:,:,:24]
    return K.log(var+tau)/2.0 + (mean - gt)*(mean - gt)/(2*var+tau)
    
def prob_srqdl(input_shape, featurenumbers, upsample):

    nDict = 201 
    nLayers1 = 8
    
    inputs = Input(shape=input_shape)
    W = Sequential() 
    W.add(Conv3D(filters=nDict, kernel_size=(1, 1, 1), activation='relu', input_shape=input_shape))
    
    a = W(inputs)
    
    TS = Sequential()
    TS_input_shape = list(input_shape)
    TS_input_shape[0] = input_shape[0] 
    TS_input_shape[1] = input_shape[1]
    TS_input_shape[2] = input_shape[2]
    TS_input_shape[3] = nDict
    TS_input_shape = tuple(TS_input_shape)
    ReLUThres = 0.01
    TS.add(ThresholdedReLU(theta = ReLUThres, input_shape=TS_input_shape))
    TS.add(Conv3D(filters=nDict, kernel_size=(1, 1, 1), activation='linear'))
    
    b = TS(a)
    
    for l in range(nLayers1-1):
        c = add([a,b])
        b = TS(c)
    y = add([a,b])
    x = ThresholdedReLU(theta = ReLUThres)(y)
    
    
    nChannels1 = 200
    nChannels2 = 400
    H = Sequential() 
    H.add(Conv3D(filters=nChannels1, kernel_size=(3, 3, 3), activation='relu', input_shape=TS_input_shape))
    H.add(Dropout(0.1))
    H.add(Conv3D(filters=nChannels2, kernel_size=(1, 1, 1), activation='relu'))
    H.add(Dropout(0.1))
    H.add(Conv3D(filters=featurenumbers*np.power(upsample, 3), kernel_size=(3, 3, 3), activation='relu'))
    
    S = Sequential() 
    S.add(Conv3D(filters=nChannels1, kernel_size=(3, 3, 3), activation='relu', input_shape=TS_input_shape))
    S.add(Dropout(0.1))
    S.add(Conv3D(filters=nChannels2, kernel_size=(1, 1, 1), activation='relu'))
    S.add(Dropout(0.1))
    S.add(Conv3D(filters=featurenumbers*np.power(upsample, 3), kernel_size=(3, 3, 3), activation='softplus'))
    
    output_mean = H(x)
    output_std = S(x)
    outputs = concatenate([output_mean, output_std], axis = 4)                   
    print "Fitting"    
    
    regressor = Model(inputs=inputs,outputs=outputs)
    regressor.compile(optimizer=Adam(lr=0.0001), loss = scoring, metrics=['mse'])
    print regressor.summary()
    return regressor
