#!/usr/bin/python3.6

# ------------------------------- Arguments description -------------------------------
# --gpu_id: the gpu device you want to use in current task
# --gpu_memory_fraction: upper bound of gpu memory fraction that can be used
# --root_path: the root path of test data folder
# --data_folder: the raw SIM image or WF iamge folder
# --load_weights_dir: the root path where the trained weights are saved
# --output_suffix: suffix of the output folder
# --model: network model to be trained, RCAN_NSM by default
# --load_weights_filter:  wildcard of the rDL model files to be loaded
# --input_y: the height of input images
# --input_x: the width of input images
# --input_z: the number of z-slices of single input volumes
# --n_channels: the number of input consecutive timepoints, 4 for TiS-rDL model, 1 for SiS-rDL model by default
# --img_background: a constant value of image background (related to the camera), 100 for LLSM data by default
# --norm_flag: 1 for min-max normalization, 0 for dividing 65535 (16-bit images)

# ------------------------------- Inference with TiS/SiS-rDL denoising model -------------------------------
python predict_rDL_TiS_SiS_Denoising.py --gpu_id '0' --gpu_memory_fraction 0.4 \
                                        --root_path "../data_test/SiS/" \
                                        --data_folder "Mitosis_ER" \
                                        --load_weights_dir "../trained_models/SiS_rDL_Denoising_Model/" \
                                        --output_suffix "" \
                                        --model "RCAN3D_NSM" \
                                        --load_weights_filter '*Latest.h5' \
                                        --input_y 64 --input_x 64 --input_z 16 --n_channels 1 \
                                        --img_background 100 \
                                        --norm_flag 1
