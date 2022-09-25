#!/usr/bin/python3.6

# ------------------------------- Arguments description -------------------------------
# --gpu_id: the gpu device you want to use in current task
# --gpu_memory_fraction: upper bound of gpu memory fraction that can be used
# --root_path: the root path of test data folder
# --data_folder: folder name of raw SIM images to be denoised
# --denoise_model: the model lable of rDL denoising module
# --load_denoise_module_dir: the directory where the weights of the rDL denoising modules are saved
# --sr_model: the model lable of SR inference module
# --load_sr_module_dir: the directory where the weights of the SR inference modules are saved
# --input_height: the height of input images
# --input_width: the width of input images
# --wave_length: the excitation wave length of SIM
# --excNA: the excitation NA of SIM

# ------------------------------- Inference with rDL denoising model -------------------------------
python predict_rDL_Denoising.py --gpu_id '0' --gpu_memory_fraction 0.4 \
                                --root_path "../data_test/rDL-SIM/" \
                                --data_folder "Microtubules" \
                                --denoise_model "rDL_Denoiser" \
                                --load_denoise_module_dir "../trained_models/rDL_Denoising_Module/" \
                                --sr_model "DFCAN" \
                                --load_sr_module_dir "../trained_models/SR_Inference_Module/" \
                                --input_height 256 --input_width 256 \
                                --wave_length 488 --excNA 1.35
