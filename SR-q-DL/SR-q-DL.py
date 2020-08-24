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
from data_loader import load_training_patch, load_test_patch, data_reshuffle_patch
from models import srqdl
from input_parser import input_parser_mesc
from keras.callbacks import EarlyStopping
import numpy as np
import progressbar
from numpy import moveaxis
from numpy import asarray
import pickle
from tensorflow.python.client import device_lib
import tensorflow as tf
print(device_lib.list_local_devices())

####     
dwinames, masknames, featurenumbers, featurenames, \
testdwinames, testmasknames, patch_size_low, patch_size_high, \
upsample, directory, norm_microstructure, nDict, nChannels1, nChannels2= input_parser_mesc(sys.argv)

####
if os.path.exists(directory) == False:
    os.mkdir(directory)

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
session = tf.Session(config=tf_config)

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

dwiTraining=asarray(dwiTraining)
dwiTraining=moveaxis(dwiTraining,1,4)
featurePatchTraining=asarray(featurePatchTraining)
featurePatchTraining=moveaxis(featurePatchTraining,1,4)

regressor = espcn_srep(dwiTraining.shape[1:], featurenumbers, upsample, nDict, nChannels1, nChannels2)

epoch = 20
#### fitting the model ####   
                
print "Fitting"    
hist = regressor.fit(dwiTraining, featurePatchTraining, batch_size=128, epochs=epoch, verbose=1, validation_split=0.1)
print(hist.history)

loss_txt_path=os.path.join(directory,'history.pkl')
with open(loss_txt_path,'wb') as f:
    pickle.dump(hist.history,f)

end = time.time()
regressor.save(os.path.join(directory,"model.h5"))
print "Training took ", (end-start)

#### Test #####
print "Test Phase"    

start = time.time()
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
    print "Loading"  
    dwi_nii = nib.load(allTestDwiNames[iMask])
    dwi = dwi_nii.get_data()
    mask_nii = nib.load(allTestMaskNames[iMask])
    mask = mask_nii.get_data()
    
    dwiTest, patchCornerList = load_test_patch(dwi, mask, patch_size_high, patch_size_low, upsample)
    
    dwiTest = asarray(dwiTest)
    dwiTest = moveaxis(dwiTest, 1, 4)
                        
    print "Computing"
    featureList = regressor.predict(dwiTest)
    
    featureList = asarray(featureList)
    featureList = moveaxis(featureList, 4, 1)
    
    features = data_reshuffle_patch(featureList, mask.shape, upsample, 
                              patch_size_high, patchCornerList, featurenumbers, scales)
    
    mask_upsampled_nii = nibabel.processing.resample_to_output(mask_nii, (mask_nii.header.get_zooms()[0]/upsample, mask_nii.header.get_zooms()[1]/upsample, 
                                                                          mask_nii.header.get_zooms()[2]/upsample))
    hdr = dwi_nii.header
    hdr.set_qform(mask_upsampled_nii.header.get_qform()) 
    for feature_index in range(featurenumbers):
        feature_nii = nib.Nifti1Image(features[:,:,:,feature_index], hdr.get_base_affine(), hdr)
        feature_name = os.path.join(directory, "ESPCN_feature_" + "%02d" % feature_index + "_sub_" + "%02d" % iMask + ".nii.gz")
        feature_nii.to_filename(feature_name)
    
end = time.time()
print "Test took ", (end-start)
    
