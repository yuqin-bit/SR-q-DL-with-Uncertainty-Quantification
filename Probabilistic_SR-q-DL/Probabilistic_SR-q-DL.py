# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 15:24:06 2016

@author: cyye
"""

import sys
import os
import nibabel as nib
import nibabel.processing
import time
from keras import backend as K
from data_loader import load_training_patch, load_test_patch, data_reshuffle_patch
from models import prob_srqdl
from input_parser import input_parser_mesc
from keras.callbacks import EarlyStopping
from keras.layers import merge, Dense, Input, add, Dropout, concatenate
import progressbar
import numpy as np
from numpy import moveaxis
from numpy import asarray
import pickle
from keras.models import save_model,load_model
from tensorflow.python.client import device_lib
import math
print(device_lib.list_local_devices())

def scoring(y_true, y_pred):
    tau = 1e-6
    mean = y_pred[:,:,:,:,:24]
    var = y_pred[:,:,:,:,24:]
    gt = y_true[:,:,:,:,:24]
    return K.log(var+tau)/2.0 + (mean - gt)*(mean - gt)/(2*var+tau)

####       
dwinames, masknames, featurenumbers, featurenames, \
testdwinames, testmasknames, patch_size_low, patch_size_high, \
upsample, directory, norm_microstructure= input_parser_mesc(sys.argv)


####
if os.path.exists(directory) == False:
    os.mkdir(directory)

#### load images ####
start = time.time()
print "Loading"    

with open(dwinames) as f:
    allDwiNames = f.readlines()
with open(masknames) as f:
    allMaskNames = f.readlines()
allFeatureNames = []
for feature_index in range(featurenumbers):
    tempFeatureNames = None
    with open(featurenames[feature_index]) as f:
        tempFeatureNames = f.readlines()
    allFeatureNames.append(tempFeatureNames)
allDwiNames = [x.strip('\n') for x in allDwiNames]
allMaskNames = [x.strip('\n') for x in allMaskNames]
for feature_index in range(featurenumbers):
    allFeatureNames[feature_index] = [x.strip('\n') for x in allFeatureNames[feature_index]]
        
        
####
dwiTraining, featurePatchTraining, scales = load_training_patch(allDwiNames, allMaskNames, allFeatureNames, 
                                                          featurenumbers, patch_size_high, patch_size_low, 
                                                          upsample, norm_microstructure)
print "scales:", scales.shape , scales
####

dwiTraining=asarray(dwiTraining)
dwiTraining=moveaxis(dwiTraining,1,4)
featurePatchTraining=asarray(featurePatchTraining)
featurePatchTraining=moveaxis(featurePatchTraining,1,4)

print "featurePatchTraining:",featurePatchTraining.shape
true = np.concatenate([featurePatchTraining, np.zeros(featurePatchTraining.shape)],axis = 4) 
print "true:",true.shape 

epoch = 20
for m in range(10):
    regressor = espcn_srep(dwiTraining.shape[1:], featurenumbers, upsample)          
    print "Fitting_model", str(m)    
    hist = regressor.fit(dwiTraining, true, batch_size=128, epochs=epoch, verbose=1, validation_split=0.1)
    print(hist.history)

    loss_txt_path=os.path.join(directory,"history"+ str(m) +".pkl")
    with open(loss_txt_path,'wb') as f:
        pickle.dump(hist.history,f)

    regressor.save(os.path.join(directory, "SRESPCN_ensemble_" + str(m) + ".h5"))
    
end = time.time()
print "Training took ", (end-start)

#### Test ####
print "Test Phase"    

start = time.time()
model_result = os.path.join(directory, "model_result")
if os.path.exists(model_result) == False:
    os.mkdir(model_result)

M = 10
predModels = []
for i in range(M):
    pred = load_model(os.path.join(directory, "SRESPCN_ensemble_" + str(i) + ".h5"), custom_objects={"tau": 1e-10, "scoring": scoring})
    predModels.append(pred)

with open(testdwinames) as f:
    allTestDwiNames = f.readlines()
with open(testmasknames) as f:
    allTestMaskNames = f.readlines()

allTestDwiNames = [x.strip('\n') for x in allTestDwiNames]
allTestMaskNames = [x.strip('\n') for x in allTestMaskNames]

#for iMask in range(len(allTestDwiNames)):
for iMask in progressbar.progressbar(range(len(allTestDwiNames))):
    print "Processing Subject: ", iMask
    #### load images ####
    #print "Loading"  
    dwi_nii = nib.load(allTestDwiNames[iMask])
    dwi = dwi_nii.get_data()
    mask_nii = nib.load(allTestMaskNames[iMask])
    mask = mask_nii.get_data()
    
    dwiTest, patchCornerList = load_test_patch(dwi, mask, patch_size_high, patch_size_low, upsample)
    
    dwiTest = asarray(dwiTest)
    dwiTest = moveaxis(dwiTest, 1, 4)
                        
    #print "Computing"
    rows = mask.shape[0]*2
    cols = mask.shape[1]*2
    slices = mask.shape[2]*2

    feature_sum = np.zeros([M, rows, cols, slices, featurenumbers])
    std_sum = np.zeros([M, rows, cols, slices, featurenumbers])
    for m in range(M):
        print "Model", m
        outputList = predModels[m].predict(dwiTest)
    
        outputList = asarray(outputList)
        outputList = moveaxis(outputList, 4, 1)
        
        featureList = outputList[:,:24,:,:,:]
        stdList = outputList[:,24:,:,:,:]

        features = data_reshuffle_patch(featureList, mask.shape, upsample, 
                                  patch_size_high, patchCornerList, featurenumbers, scales)
        feature_sum[m,:,:,:,:] = features

        stds = data_reshuffle_patch(stdList, mask.shape, upsample, 
                                  patch_size_high, patchCornerList, featurenumbers, scales)
        std_sum[m,:,:,:,:] = stds

        mask_upsampled_nii = nibabel.processing.resample_to_output(mask_nii, (mask_nii.header.get_zooms()[0]/upsample, mask_nii.header.get_zooms()[1]/upsample, mask_nii.header.get_zooms()[2]/upsample))

        hdr = dwi_nii.header
        hdr.set_qform(mask_upsampled_nii.header.get_qform()) 
        for feature_index in range(featurenumbers):
            feature_nii = nib.Nifti1Image(features[:,:,:,feature_index], hdr.get_base_affine(), hdr)
            feature_name = os.path.join(model_result, "feature_model_" + "%02d" % m + "_feature_" + "%02d" % feature_index + "_sub_" + "%02d" % iMask + ".nii.gz")
            feature_nii.to_filename(feature_name)

            std_nii = nib.Nifti1Image(stds[:,:,:,feature_index], hdr.get_base_affine(), hdr)
            std_name = os.path.join(model_result, "std_model_" + "%02d" % m + "_feature_" + "%02d" % feature_index + "_sub_" + "%02d" % iMask + ".nii.gz")
            std_nii.to_filename(std_name)


    mean_feature = np.mean(feature_sum, axis = 0)
    for feature_index in range(featurenumbers):
        feature_nii = nib.Nifti1Image(mean_feature[:,:,:,feature_index], hdr.get_base_affine(), hdr)
        feature_name = os.path.join(directory, "ESPCN_mean_feature_" + "%02d" % feature_index + "_sub_" + "%02d" % iMask + ".nii.gz")
        feature_nii.to_filename(feature_name)
    std_feature = np.sqrt((np.sum(feature_sum*feature_sum, axis = 0) +np.sum(std_sum*std_sum, axis = 0))/M - mean_feature*mean_feature)
    for feature_index in range(featurenumbers):
        feature_nii = nib.Nifti1Image(std_feature[:,:,:,feature_index], hdr.get_base_affine(), hdr)
        feature_name = os.path.join(directory, "ESPCN_std_feature_" + "%02d" % feature_index + "_sub_" + "%02d" % iMask + ".nii.gz")
        feature_nii.to_filename(feature_name)
    
end = time.time()
print "Test took ", (end-start)

