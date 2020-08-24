#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 15:49:54 2019

@author: pkuclosed
"""
import nibabel as nib
import numpy as np

####
def load_training_patch(allDwiNames, allMaskNames, allFeatureNames, featurenumbers, 
                  patch_size_high, patch_size_low, upsample, norm_microstructure, shift=None):
    #### counting ####
    if shift == None:
        shift = patch_size_high
        if patch_size_high > 5:
            shift = patch_size_high//3
    nPatch = 0
    nVox = 0
    for iMask in range(len(allMaskNames)):
        print "Counting High-Res Patches for Subject", iMask
        mask = nib.load(allMaskNames[iMask]).get_data()
        # number of patches
        for i in range(0, mask.shape[0], patch_size_high):
            for i1 in range(0, patch_size_high, shift):
                for j in range(0, mask.shape[1], patch_size_high):
                    for j1 in range(0, patch_size_high, shift):
                        for k in range(0, mask.shape[2], patch_size_high):
                            for k1 in range(0, patch_size_high, shift):
                                starti = min(mask.shape[0]-1, i+i1)
                                endi = min(mask.shape[0], i+i1+patch_size_high)
                                startj = min(mask.shape[1]-1, j+j1)
                                endj = min(mask.shape[1], j+j1+patch_size_high)
                                startk = min(mask.shape[2]-1, k+k1)
                                endk = min(mask.shape[2], k+k1+patch_size_high)
                                if np.sum(mask[starti:endi, startj:endj, startk:endk]) > 0.0*np.power(patch_size_high,3):
                                    nPatch = nPatch + 1
        print "Counting High-Res Voxels for Subject", iMask
        # number of voxels, used if normalizing microstructure
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                for k in range(mask.shape[2]):
                    if mask[i,j,k] > 0:
                        nVox = nVox + 1 
    dwi = nib.load(allDwiNames[0]).get_data()                   
    dwiTraining = np.zeros([nPatch, dwi.shape[3], patch_size_low, patch_size_low, patch_size_low])  
    featurePatchTraining = np.zeros([nPatch, featurenumbers*np.power(upsample,3), patch_size_high/upsample, 
                                     patch_size_high/upsample, patch_size_high/upsample])
    featureTraining = np.zeros([nVox, featurenumbers])
    
    # Normalize microstructure
    
    if norm_microstructure == 1:
        print "Normalizing Microstructure"
        nVox = 0
            
        for iMask in range(len(allDwiNames)):
            print "Examining High-Res Voxels for Subject:", iMask
            mask = nib.load(allMaskNames[iMask]).get_data()
            feature = []
            for feature_index in range(featurenumbers):
                tempFeature = nib.load(allFeatureNames[feature_index][iMask]).get_data()
                feature.append(tempFeature)
                for i in range(dwi.shape[0]):
                    for j in range(dwi.shape[1]):
                        for k in range(dwi.shape[2]):
                            if mask[i,j,k] > 0:
                                for feature_index in range(featurenumbers):
                                    featureTraining[nVox,feature_index] = feature[feature_index][i,j,k]
                                nVox = nVox + 1
        means = np.mean(featureTraining, axis = 0)               
        scales = np.log10(means)
        scalesint = np.floor(scales)
        scales = np.power(10,scalesint)
        print "scales:", scalesint, scales
    else:
        scales = np.ones(featurenumbers)
        print "scales:", scales
    
    # Setting data
    nPatch = 0
    for iMask in range(len(allDwiNames)):
        print "Setting Patch List for Subject:", iMask
        dwi_nii = nib.load(allDwiNames[iMask])
        dwi = dwi_nii.get_data()
        mask = nib.load(allMaskNames[iMask]).get_data()
        feature = []
        for feature_index in range(featurenumbers):
            tempFeature = nib.load(allFeatureNames[feature_index][iMask]).get_data()
            feature.append(tempFeature)
    
        for i in range(0, mask.shape[0], patch_size_high):
            for i1 in range(0, patch_size_high, shift):
                for j in range(0, mask.shape[1], patch_size_high):
                    for j1 in range(0, patch_size_high, shift):
                        for k in range(0, mask.shape[2], patch_size_high):
                            for k1 in range(0, patch_size_high, shift):
                                starti = min(mask.shape[0]-1, i+i1)
                                endi = min(mask.shape[0], i+i1+patch_size_high)
                                startj = min(mask.shape[1]-1, j+j1)
                                endj = min(mask.shape[1], j+j1+patch_size_high)
                                startk = min(mask.shape[2]-1, k+k1)
                                endk = min(mask.shape[2], k+k1+patch_size_high)
                                if np.sum(mask[starti:endi, startj:endj, startk:endk]) > 0.0*np.power(patch_size_high,3):
                                    for feature_index in range(featurenumbers):
                                        for ii in range(patch_size_high):
                                            for jj in range(patch_size_high):
                                                for kk in range(patch_size_high):
                                                    voxel_i = ii//upsample
                                                    channel_i = ii - voxel_i*upsample
                                                    voxel_j = jj//upsample
                                                    channel_j = jj - voxel_j*upsample
                                                    voxel_k = kk//upsample
                                                    channel_k = kk - voxel_k*upsample
                                                    channel = feature_index*np.power(upsample,3) + channel_i*upsample*upsample + \
                                                        channel_j*upsample + channel_k
                                                    if i + i1 + ii >= 0 and i + i1 + ii < mask.shape[0] \
                                                    and j + j1 + jj >= 0 and j + j1 + jj < mask.shape[1] \
                                                    and k + k1 + kk >= 0 and k + k1 + kk < mask.shape[2] \
                                                    and mask[i + i1 + ii, j + j1 + jj, k + k1 + kk] > 0:  
                                                        featurePatchTraining[nPatch, channel, voxel_i, voxel_j, voxel_k] \
                                                         = feature[feature_index][i+i1+ii,j+j1+jj,k+k1+kk]/scales[feature_index]  
                                                    else:
                                                        featurePatchTraining[nPatch, channel, voxel_i, voxel_j, voxel_k] = 0
                                    
        
                                    i_lr_center = (i+i1)/upsample + (patch_size_high/upsample-1)/2
                                    j_lr_center = (j+j1)/upsample + (patch_size_high/upsample-1)/2
                                    k_lr_center = (k+k1)/upsample + (patch_size_high/upsample-1)/2

                                    for ii in range(-patch_size_low//2 + 1,patch_size_low//2 + 1):
                                        for jj in range(-patch_size_low//2 + 1,patch_size_low//2 + 1):
                                            for kk in range(-patch_size_low//2 + 1,patch_size_low//2 + 1):   
                                                dwiTraining[nPatch, :, -(-patch_size_low//2 + 1) + ii, 
                                                        -(-patch_size_low//2 + 1) + jj, -(-patch_size_low//2 + 1) + kk] = 0                 
                                                if i_lr_center + ii >= 0 and i_lr_center + ii < dwi.shape[0] and j_lr_center + jj >= 0 and j_lr_center + jj < dwi.shape[1] \
                                                and k_lr_center + kk >= 0 and k_lr_center + kk < dwi.shape[2]: 
                                                    in_mask = False
                                                    for indx in range(upsample):
                                                        for indy in range(upsample):
                                                            for indz in range(upsample):
                                                                if mask[(i_lr_center + ii)*upsample + indx, (j_lr_center + jj)*upsample + indy
                                                                        , (k_lr_center + kk)*upsample + indz] > 0:
                                                                    in_mask = True
                                                                    break
                                                    if in_mask:
                                                        dwiTraining[nPatch, :, -(-patch_size_low//2 + 1) + ii, 
                                                                     -(-patch_size_low//2 + 1) + jj, -(-patch_size_low//2 + 1) + kk] \
                                                        = dwi[i_lr_center+ii, j_lr_center+jj, k_lr_center+kk, :]
                                    nPatch = nPatch + 1
                        
    return dwiTraining, featurePatchTraining, scales

####   
def load_test_patch(dwi, mask, patch_size_high, patch_size_low, upsample):
    #print "Counting High-Res Patches"
    nPatch = 0
    
    for i in range(0, mask.shape[0]*upsample, patch_size_high):
        for j in range(0, mask.shape[1]*upsample, patch_size_high):
            for k in range(0, mask.shape[2]*upsample, patch_size_high):
                if np.sum(mask[i//upsample:min(dwi.shape[0], i//upsample+patch_size_high//upsample),
                        j//upsample:min(dwi.shape[1], j//upsample+patch_size_high//upsample),
                        k//upsample:min(dwi.shape[2], k//upsample+patch_size_high//upsample)]) > 0.0*np.power(patch_size_high//upsample,3):
                #if mask[i//upsample,j//upsample,k//upsample] > 0:
                    nPatch = nPatch + 1
                    
    patchCornerList = np.zeros([nPatch, 3], int)
    dwiTest = np.zeros([nPatch, dwi.shape[3], patch_size_low, patch_size_low, patch_size_low])   

    #print "Setting Voxels"
    nPatch = 0
    for i in range(0, dwi.shape[0]*upsample, patch_size_high):
        for j in range(0, dwi.shape[1]*upsample, patch_size_high):
            for k in range(0, dwi.shape[2]*upsample, patch_size_high):
                #if mask[i//upsample,j//upsample,k//upsample] > 0:
                if np.sum(mask[i//upsample:min(dwi.shape[0], i//upsample+patch_size_high//upsample),
                        j//upsample:min(dwi.shape[1], j//upsample+patch_size_high//upsample),
                        k//upsample:min(dwi.shape[2], k//upsample+patch_size_high//upsample)]) > 0.0*np.power(patch_size_high//upsample,3):
                    patchCornerList[nPatch,0] = i
                    patchCornerList[nPatch,1] = j
                    patchCornerList[nPatch,2] = k
                    i_lr_center = i/upsample + (patch_size_high/upsample-1)/2
                    j_lr_center = j/upsample + (patch_size_high/upsample-1)/2
                    k_lr_center = k/upsample + (patch_size_high/upsample-1)/2
                    for ii in range(-patch_size_low//2 + 1, patch_size_low//2 + 1):
                        for jj in range(-patch_size_low//2 + 1, patch_size_low//2 + 1):
                            for kk in range(-patch_size_low//2 + 1, patch_size_low//2 + 1):
                                dwiTest[nPatch, :, -(-patch_size_low//2 + 1) + ii, 
                                            -(-patch_size_low//2 + 1) + jj, -(-patch_size_low//2 + 1) + kk] = 0 

                                if i_lr_center + ii >= 0 and i_lr_center + ii < dwi.shape[0] and j_lr_center + jj >= 0 and j_lr_center + jj < dwi.shape[1] \
                                and k_lr_center + kk >= 0 and k_lr_center + kk < dwi.shape[2] and mask[i_lr_center + ii, j_lr_center + jj, k_lr_center + kk]: 
                                    dwiTest[nPatch, :, -(-patch_size_low//2 + 1) + ii, 
                                            -(-patch_size_low//2 + 1) + jj, -(-patch_size_low//2 + 1) + kk]\
                                    = dwi[i_lr_center+ii, j_lr_center+jj, k_lr_center+kk, :]
                    nPatch = nPatch + 1
                    
    return dwiTest, patchCornerList

####
def data_reshuffle_patch(featureList, shape, upsample, patch_size_high, patchCornerList, featurenumbers, scales):
    rows = shape[0]*upsample
    cols = shape[1]*upsample
    slices = shape[2]*upsample
    
    features = np.zeros([rows,cols,slices,featurenumbers])
    for nPatch in range(patchCornerList.shape[0]):
        i = patchCornerList[nPatch,0]
        j = patchCornerList[nPatch,1]
        k = patchCornerList[nPatch,2]
        for feature_index in range(featurenumbers):
            for ii in range(patch_size_high):
                for jj in range(patch_size_high):
                    for kk in range(patch_size_high):
                        voxel_i = ii//upsample
                        channel_i = ii - voxel_i*upsample
                        voxel_j = jj//upsample
                        channel_j = jj - voxel_j*upsample
                        voxel_k = kk//upsample
                        channel_k = kk - voxel_k*upsample
                        channel = feature_index*np.power(upsample,3) + channel_i*upsample*upsample + \
                            channel_j*upsample + channel_k
                        if i + ii < shape[0]*upsample and j + jj < shape[1]*upsample and k + kk < shape[2]*upsample:  
                            features[i+ii,j+jj,k+kk,feature_index] = featureList[nPatch, channel, voxel_i, voxel_j, voxel_k]
    return features