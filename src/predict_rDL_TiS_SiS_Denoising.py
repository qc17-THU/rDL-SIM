import os
import argparse
import glob
import numpy as np
import datetime
import tifffile as tiff
import tensorflow as tf
from models import *
from tensorflow.keras import optimizers
from utils.utils import prctile_norm


# --------------------------------------------------------------------------------
#                                 define parameters
# --------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# directory parameters
parser.add_argument("--root_path", type=str, default="../data_test/SiS/")
parser.add_argument("--data_folder", type=str, default="Mitosis_ER")
parser.add_argument("--load_weights_dir", type=str, default="../trained_models/SiS_rDL_Denoising_Model/")
parser.add_argument("--output_suffix", type=str, default="")
# model parameters
parser.add_argument("--model", type=str, default="RCAN3D_NSM")
parser.add_argument("--load_weights_filter", type=str, default='*Latest.h5')
# training parameters
parser.add_argument("--gpu_id", type=str, default="4")
parser.add_argument("--gpu_memory_fraction", type=float, default=0.5)
# image parameters
parser.add_argument("--input_y", type=int, default=64)
parser.add_argument("--input_x", type=int, default=64)
parser.add_argument("--input_z", type=int, default=16)
parser.add_argument("--n_channels", type=int, default=1)
parser.add_argument("--img_background", type=int, default=100)
parser.add_argument("--norm_flag", type=int, default=0)

# --------------------------------------------------------------------------------
#                          instantiation for parameters
# --------------------------------------------------------------------------------
args = parser.parse_args()
root_path = args.root_path
data_folder = args.data_folder
load_weights_dir = args.load_weights_dir
output_suffix = args.output_suffix

model = args.model
load_weights_filter = args.load_weights_filter

gpu_id = args.gpu_id
gpu_memory_fraction = args.gpu_memory_fraction

input_y = args.input_y
input_x = args.input_x
input_z = args.input_z
n_channels = args.n_channels
img_background = args.img_background
norm_flag = args.norm_flag

# define and make output dir
if load_weights_dir[-1] != '/':
    load_weights_dir = load_weights_dir + '/'
load_weights_dir = load_weights_dir + data_folder + '/'
test_img_path = root_path + data_folder + '/'
output_path = root_path + data_folder + '_rDL_Denoised' + output_suffix + '/'
if not os.path.exists(output_path):
    os.mkdir(output_path)

# define GPU environment
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# --------------------------------------------------------------------------------
#                    select and define rDL-TiS denoising model
# --------------------------------------------------------------------------------
modelFns = {'RCAN3D': RCAN3D.RCAN3D, 'RCAN3D_NSM': RCAN3D_NSM.RCAN3D}
modelFN = modelFns[model]
g = modelFN((input_y, input_x, input_z, n_channels), n_ResGroup=3, n_RCAB=5)

# --------------------------------------------------------------------------------
#                                load weights
# --------------------------------------------------------------------------------
weights_file = glob.glob(load_weights_dir + load_weights_filter)
weights_file.sort()
weights_file.sort(key=lambda i: len(i))
print('Load existing weights: ' + weights_file[-1])
g.load_weights(weights_file[-1])

# --------------------------------------------------------------------------------
#                                  Training
# --------------------------------------------------------------------------------
start_time = datetime.datetime.now()
images_path = glob.glob(test_img_path + '*.tif')
for path in images_path:
    # ------------------------------------------------------------------------------
    #                              load training data
    # ------------------------------------------------------------------------------
    img = tiff.imread(path)
    img = img - img_background
    img[img < 0] = 0
    if norm_flag == 1:
        img = prctile_norm(np.array(img))
    else:
        img = np.array(img) / 65535

    img = np.reshape(img, (input_z,  n_channels, input_y, input_x), order='F')
    img = np.transpose(img, (2, 3, 0, 1))

    pred = np.transpose(g.predict(img[np.newaxis, ...]), (0, 3, 4, 1, 2))
    pred = np.reshape(pred, (input_z*n_channels, input_y, input_x), order='F')
    pred = (1e4 * prctile_norm(pred)).astype('uint16')

    cur_out_path = path.replace(test_img_path, output_path)
    tiff.imwrite(cur_out_path, pred, dtype='uint16')
