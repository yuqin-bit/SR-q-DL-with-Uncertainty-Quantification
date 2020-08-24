#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 16:46:09 2019

@author: pkuclosed
"""

def input_parser_mesc(argv):
    
        dwinames = argv[1] # low resolution diffusion signals
        masknames = argv[2] # high resolution mask
        featurenumbers = int(argv[3]) # high resolution microstructure maps
        featurenames = []
        for feature_index in range(featurenumbers):
            featurenames.append(argv[4 + feature_index])
            
        testdwinames = argv[4 + featurenumbers] # low resolution diffusion signals
        testmasknames = argv[5 + featurenumbers] # low resolution brain mask
        
        patch_size_low = int(argv[6 + featurenumbers]) # low resolution patch_size = 11
        patch_size_high = int(argv[7 + featurenumbers]) # low resolution patch_size  = 7*2 = 14
        upsample = int(argv[8 + featurenumbers])
        directory = argv[9 + featurenumbers]
        
        norm_microstructure = False
        if len(argv) == 11 + featurenumbers:
            norm_microstructure = int(argv[10 + featurenumbers])
        if len(argv) == 14 + featurenumbers:
            nDict = int(argv[11 + featurenumbers])
            nChannels1 = int(argv[12 + featurenumbers])
            nChannels2 = int(argv[13 + featurenumbers])
        
    
    return dwinames, masknames, featurenumbers, featurenames, testdwinames, testmasknames, \
        patch_size_low, patch_size_high, upsample, directory, norm_microstructure
