# SR-q-DL-with-Uncertainty-Quantification

The demo includes both the training and test phase for the SR-*q*-DL and probabilistic SR-*q*-DL. Therefore, to run it, both the training and test data (which are images in the NIfTI format) should be prepared. The input diffusion signals should be normalized by the b0 signals.

There are a few dependencies that need to be installed:<br />
**numpy <br />
nibabel <br />
keras <br />
tensorflow <br />**

Here is how to run the scripts. For SR-*q*-DL, run <br />
> CUDA_VISIBLE_DEVICES=0 time python SR-q-DL.py < list of training normalized diffusion images> < list of training brain mask images > < number of microstructure measures to be estimated > < list of training microstructure 1 > ... < list of training microstructure N > < list of test normalized diffusion images > < list of test brain mask images > < input patch size > < output patch size > < upsampling rate > < output directory > < control variable for normalizing microstructure > < dictionary size N_d > < number N_1 of channels > < number N_2 of channels > <br />

Note that the control variable for normalizing microstructure is set to zero in this case.

For example, <br />
> CUDA_VISIBLE_DEVICES=0 time python SR-q-DL.py dwis_training.txt masks_training.txt 3 icvfs_training.txt isos_training.txt ods_training.txt dwis_test.txt masks_test.txt 5 2 2 output_directory 201 200 400 <br />

For probabilistic SR-*q*-DL, run <br />
> CUDA_VISIBLE_DEVICES=0 time python Probabilistic_SR-q-DL.py < list of training normalized diffusion images> < list of training brain mask images > < number of microstructure measures to be estimated > < list of training microstructure 1 > ... < list of training microstructure N > < list of test normalized diffusion images > < list of test brain mask images > < input patch size > < output patch size > < upsampling rate > < output directory > <br />


For example, <br />
> CUDA_VISIBLE_DEVICES=0 time python SR-q-DL.py dwis_training.txt masks_training.txt 3 icvfs_training.txt isos_training.txt ods_training.txt dwis_test.txt masks_test.txt 5 2 2 output_directory <br />
