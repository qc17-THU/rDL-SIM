#!/usr/bin/python3.6

# ------------------------------- STEP 1: train SR inference module -------------------------------
# ------------------------------------ Arguments description --------------------------------------
# --gpu_id: the gpu device you want to use in current task
# --gpu_memory_fraction: upper bound of gpu memory fraction that can be used
# --root_path: the root path of train data folder
# --data_folder: the folder name of the augmented train data
# --save_weights_path: the root path where the trained weights are saved
# --save_weights_suffix: suffix of the folder where the trained weights are saved
# --load_weights_flag: whether to load existing weights, 1 for true, 0 for false
# --model_name: network model of the SR inference module, DFCAN by default
# --total_iterations: total training iterations
# --sample_interval: sampling interval for visualizing the network outputs during training
# --validate_interval: validation interval for evaluate the current models during training
# --validate_num: how many validation images used at validation stage
# --input_height: the height of input images
# --input_width: the width of input images
# --input_channels: num of input channels, 9 by default for linear SIM
# --scale_factor: upscaling factor for image super-resolution, 2 by default for linear SIM
# --norm_flag: 1 for min-max normalization, 0 for dividing 65535 (16-bit images)

# ----------------------------------- train SR inference module -----------------------------------
python train_SR_Inference_Module.py --gpu_id '0' --gpu_memory_fraction 0.4 \
                                     --root_path "../data_train/rDL-SIM/SR/" \
                                    --data_folder "Microtubules" \
                                    --save_weights_path "../trained_models/SR_Inference_Module/" \
                                    --save_weights_suffix "" \
                                    --load_weights_flag 0 \
                                    --model_name "DFCAN" \
                                    --total_iterations 100000 --sample_interval 1000 \
                                    --validate_interval 2000 --validate_num 1000 \
                                    --batch_size 4 --start_lr 1e-4 \
                                    --input_height 128 --input_width 128 --input_channels 9 \
                                    --scale_factor 2 --norm_flag 1

# ------------------------------- STEP 2: train rDL denoising module ------------------------------
# ------------------------------------ Arguments description --------------------------------------
# --gpu_id: the gpu device you want to use in current task
# --gpu_memory_fraction: upper bound of gpu memory fraction that can be used
# --root_path: the root path of train data folder
# --data_folder: the folder name of the augmented train data
# --save_weights_path: the root path where the trained weights are saved
# --save_weights_suffix: suffix of the folder where the trained weights are saved
# --denoise_model: the model lable of rDL denoising module
# --sr_model: the model lable of SR inference module
# --load_sr_module_dir: the directory where the weights of the SR inference modules are saved
# --load_sr_module_filter: wildcard of the SR inference module weights to be loaded
# --total_iterations: total training iterations
# --sample_interval: sampling interval for visualizing the network outputs during training
# --validate_interval: validation interval for evaluate the current models during training
# --validate_num: how many validation images used at validation stage
# --input_height: the height of input images
# --input_width: the width of input images
# --wave_length: the excitation wave length of SIM
# --excNA: the excitation NA of SIM

# ---------------------------------- train rDL denoising module -----------------------------------
python train_rDL_Denoising_Module.py --gpu_id '0' --gpu_memory_fraction 0.4 \
                                     --root_path "../data_train/rDL-SIM/DN/" \
                                     --data_folder "Microtubules" \
                                     --save_weights_path "../trained_models/rDL_Denoising_Module/" \
                                     --save_weights_suffix "" \
                                     --denoise_model "rDL_Denoiser" \
                                     --load_sr_module_dir "../trained_models/SR_Inference_Module/" \
                                     --load_sr_module_filter "*Best.h5" \
                                     --sr_model "DFCAN" \
                                     --total_iterations 200000 --sample_interval 1000 \
                                     --validate_interval 2000 --validate_num 500 \
                                     --input_height 128 --input_width 128 \
                                     --wave_length 488 --excNA 1.35

