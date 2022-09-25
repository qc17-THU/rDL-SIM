#!/usr/bin/python3.6

# ------------------------------- Arguments description -------------------------------
# --gpu_id: the gpu device you want to use in current task
# --gpu_memory_fraction: upper bound of gpu memory fraction that can be used
# --root_path: the root path of test data folder
# --data_folder: the raw SIM image or WF iamge folder
# --save_weights_path: the root path where the trained weights are saved
# --save_weights_suffix: suffix of the folder where the trained weights are saved
# --model: network model to be trained, RCAN_NSM by default
# --load_weights_flag: whether to load existing weights, 1 for true, 0 for false
# --total_iterations: total training iterations
# --sample_interval: sampling interval for visualizing the network outputs during training
# --save_model_interval: iteration interval for saving current weights
# --input_y: the height of input images
# --input_x: the width of input images
# --input_z: the number of z-slices of single input volumes
# --n_channels: the number of input consecutive timepoints, 4 for TiS-rDL model, 1 for SiS-rDL model by default
# --img_background: a constant value of image background (related to the camera), 100 for LLSM data by default
# --norm_flag: 1 for min-max normalization, 0 for dividing 65535 (16-bit images)

# ------------------------------- train TiS-rDL denoising model -------------------------------
python train_rDL_TiS_Model.py --gpu_id '0' --gpu_memory_fraction 0.4 \
                              --root_path "../data_train/TiS/" \
                              --data_folder "SiT-Golgi" \
                              --save_weights_path "../trained_models/TiS_rDL_Denoising_Model/" \
                              --save_weights_suffix "" \
                              --model "RCAN3D_NSM" \
                              --load_weights_flag 0 \
                              --total_iterations 50000 --sample_interval 1000 \
                              --save_model_interval 1000 \
                              --batch_size 3 --init_lr 1e-4 \
                              --input_y 64 --input_x 64 --input_z 16 --n_channels 4 \
                              --img_background 100 \
                              --norm_flag 1

# ------------------------------- train SiS-rDL denoising model -------------------------------
python train_rDL_SiS_Model.py --gpu_id '4' --gpu_memory_fraction 0.4 \
                              --root_path "../data_train/SiS/" \
                              --data_folder "Mitosis_ER" \
                              --save_weights_path "../trained_models/SiS_rDL_Denoising_Model/" \
                              --save_weights_suffix "" \
                              --model "RCAN3D_NSM" \
                              --load_weights_flag 0 \
                              --total_iterations 50000 --sample_interval 1000 \
                              --save_model_interval 1000 \
                              --batch_size 3 --init_lr 1e-4 \
                              --input_y 64 --input_x 64 --input_z 8 --n_channels 1 \
                              --img_background 100 \
                              --norm_flag 1
