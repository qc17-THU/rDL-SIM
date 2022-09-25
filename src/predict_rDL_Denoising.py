import argparse
import glob
import cv2
import numpy as np
import os
import numpy.fft as F
import imageio
import tensorflow as tf
import tifffile as tiff
from models import *
from tensorflow.keras.models import load_model
from utils.loss import mse_ssim
from utils.utils import prctile_norm
from utils.read_mrc import read_mrc, write_mrc
from PIL import Image
from tensorflow.keras import optimizers
from sim_fitting.Parameters_2DSIM import parameters
from sim_fitting.CalModamp_2DSIM import cal_modamp
from sim_fitting.read_otf import read_otf
from scipy.interpolate import interp1d

# --------------------------------------------------------------------------------
#                                 define parameters
# --------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# directory parameters
parser.add_argument("--gpu_id", type=str, default="4")
parser.add_argument("--gpu_memory_fraction", type=float, default=0.5)
parser.add_argument("--root_path", type=str, default="../data_test/rDL-SIM/")
parser.add_argument("--data_folder", type=str, default="Microtubules")
parser.add_argument("--output_suffix", type=str, default="")
# model parameters
parser.add_argument("--load_denoise_module_dir", type=str, default="../trained_models/rDL_Denoising_Module/")
parser.add_argument("--load_denoise_module_filter", type=str, default="*Best.h5")
parser.add_argument("--denoise_model", type=str, default="rDL_Denoiser")
parser.add_argument("--load_sr_module_dir", type=str, default="../trained_models/SR_Inference_Module/")
parser.add_argument("--load_sr_module_filter", type=str, default="*Best.h5")
parser.add_argument("--sr_model", type=str, default="DFCAN")
# image parameters
parser.add_argument("--input_height", type=int, default=256)
parser.add_argument("--input_width", type=int, default=256)
parser.add_argument("--num_test", type=int, default=-1)
parser.add_argument("--num_average", type=int, default=1)
parser.add_argument("--frame_start", type=int, default=0)
# SIM parameters
parser.add_argument("--wave_length", type=int, default=488)
parser.add_argument("--excNA", type=float, default=1.35)
parser.add_argument("--OTF_path_488", type=str, default='./sim_fitting/OTF/TIRF488_cam1_0_z30_OTF2d.mrc')
parser.add_argument("--OTF_path_560", type=str, default='./sim_fitting/OTF/TIRF560_cam2_0_z21_OTF2d.mrc')
parser.add_argument("--OTF_path_647", type=str, default='./sim_fitting/OTF/TIRF647_cam2_0_z21_OTF2d.mrc')

# --------------------------------------------------------------------------------
#                          instantiation for parameters
# --------------------------------------------------------------------------------
args = parser.parse_args()
gpu_id = args.gpu_id
gpu_memory_fraction = args.gpu_memory_fraction
root_path = args.root_path
data_folder = args.data_folder
output_suffix = args.output_suffix

load_denoise_module_dir = args.load_denoise_module_dir
load_denoise_module_filter = args.load_denoise_module_filter
denoise_model = args.denoise_model
load_sr_module_dir = args.load_sr_module_dir
load_sr_module_filter = args.load_sr_module_filter
sr_model = args.sr_model

input_width = args.input_width
input_height = args.input_height
num_test = args.num_test
num_average = args.num_average
frame_start = args.frame_start

# make output dir
if load_denoise_module_dir[-1] != '/':
    load_denoise_module_dir = load_denoise_module_dir + '/'
if load_sr_module_dir[-1] != '/':
    load_sr_module_dir = load_sr_module_dir + '/'
load_denoise_module_dir = load_denoise_module_dir + data_folder + '/'
load_sr_module_dir = load_sr_module_dir + data_folder + '/'
test_images_path = root_path + data_folder
output_path = root_path + data_folder + '_rDL_Denoised' + output_suffix
if not os.path.exists(output_path):
    os.mkdir(output_path)

# define SIM parameters
wave_length = args.wave_length
excNA = args.excNA
OTF_path_488 = args.OTF_path_488
OTF_path_560 = args.OTF_path_560
OTF_path_647 = args.OTF_path_647
OTF_path_list = {488: OTF_path_488, 560: OTF_path_560, 647: OTF_path_647}
pParam = parameters(input_height, input_width, wave_length * 1e-3, excNA, setup=0)

# define GPU environment
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# --------------------------------------------------------------------------------
#                           select models and optimizer
# --------------------------------------------------------------------------------
modelFns = {'DFCAN': DFCAN.DFCAN, 'rDL_Denoiser': rDL_Denoiser.Denoiser, 'rDL_Denoiser_NSM': rDL_Denoiser_NSM.Denoiser}
modelFN_dn = modelFns[denoise_model]
modelFN_sr = modelFns[sr_model]

# --------------------------------------------------------------------------------
#                      define generator model and read weights
# --------------------------------------------------------------------------------
g = modelFN_dn((input_height, input_width, pParam.nphases))
weights_file = glob.glob(load_denoise_module_dir + load_denoise_module_filter)
weights_file.sort(key=len)
g.load_weights(weights_file[-1])
print('Load rDL denoise model: ' + weights_file[-1])
optimizer_g = optimizers.Adam(lr=1e-4, decay=0.5)
g.compile(loss=mse_ssim, optimizer=optimizer_g)

p = modelFN_sr((input_height, input_width, pParam.ndirs*pParam.nphases))
weights_file = glob.glob(load_sr_module_dir + load_sr_module_filter)
weights_file.sort(key=len)
p.load_weights(weights_file[-1])
print('Load SR inference model: ' + weights_file[-1])
optimizer_p = optimizers.Adam(lr=1e-4, decay=0.5)
p.compile(loss=mse_ssim, optimizer=optimizer_p)

# --------------------------------------------------------------------------------
#                           read and generate 2D OTF image
# --------------------------------------------------------------------------------
# define parameters
[Nx, Ny] = [pParam.Nx, pParam.Ny]
[dx, dy, dxy] = [pParam.dx, pParam.dy, pParam.dxy]
[dkx, dky, dkr] = [pParam.dkx, pParam.dky, pParam.dkr]
[nphases, ndirs] = [pParam.nphases, pParam.ndirs]
space = pParam.space
phase_space = 2 * np.pi / nphases
scale = pParam.scale

[Nx_hr, Ny_hr] = [Nx, Ny] * scale
[dx_hr, dy_hr] = [x / scale for x in [dx, dy]]

xx = dx_hr * np.arange(-Nx_hr / 2, Nx_hr / 2, 1)
yy = dy_hr * np.arange(-Ny_hr / 2, Ny_hr / 2, 1)
[X, Y] = np.meshgrid(xx, yy)

OTF_path = OTF_path_list[wave_length]
OTF, prol_OTF, PSF = read_otf(OTF_path_list[wave_length], Nx_hr, Ny_hr, dkx, dky, dkr)

# --------------------------------------------------------------------------------
#                               perform denoising
# --------------------------------------------------------------------------------
total_im = frame_start * nphases * ndirs
images_path = glob.glob(test_images_path + '/*.mrc')
images_path.sort()
if num_test == -1:
    num_test = len(images_path)
cycle_num = int(num_test / num_average)
for i in range(cycle_num):
    average_batch = np.zeros((Ny, Nx, ndirs * nphases))
    images_percycle = []
    images_path_percycle = []

    # calculate average images and modamps
    for j in range(num_average):
        path = images_path[i * num_average + j + frame_start]
        header, input_g = read_mrc(path, filetype='image')
        average_batch = average_batch + input_g
        images_percycle.append(input_g)
        images_path_percycle.append(path)

    average_batch = average_batch / num_average
    cur_k0, modamp = cal_modamp(np.array(average_batch), prol_OTF, pParam)

    cur_k0_angle = np.array(np.arctan(cur_k0[:, 1] / cur_k0[:, 0]))
    cur_k0_angle[1:3] = cur_k0_angle[1:3] + np.pi
    cur_k0_angle = -(cur_k0_angle - np.pi / 2)
    for nd in range(ndirs):
        if np.abs(cur_k0_angle[nd] - pParam.k0angle_g[nd]) > 0.05:
            cur_k0_angle[nd] = pParam.k0angle_g[nd]
    cur_k0 = np.sqrt(np.sum(np.square(cur_k0), 1))
    given_k0 = 1 / space
    cur_k0[np.abs(cur_k0 - given_k0) > 0.1] = given_k0

    print('process cycle %d / %d, modamp = %.7s, line spacing = %s' %
          (i + 1, cycle_num, np.mean(np.abs(modamp)), 1 / cur_k0))

    # perform denoising cycle by cycle
    for j in range(num_average):
        test_num = i * num_average + j + frame_start
        input_g = np.squeeze(images_percycle[j])
        input_g = input_g / 65535
        # input_g = prctile_norm(input_g)

        imSIM = p.predict(prctile_norm(input_g[np.newaxis, :]))
        imSIM = prctile_norm(np.squeeze(imSIM))
        imSIM = cv2.resize(imSIM, (Nx_hr, Ny_hr))

        # normalize the intensity cross three orientations
        input_g = input_g.transpose((2, 0, 1))
        datad = np.zeros((ndirs * nphases, input_height, input_width))
        meand1 = np.mean(input_g[0:nphases, :, :])
        for d in range(ndirs):
            datadi = input_g[d * nphases:(d + 1) * nphases, :, :]
            meandi = np.mean(datadi)
            datad[d * nphases:(d + 1) * nphases, :, :] = datadi * meand1 / meandi
        input_g = datad

        # generate pattern modulated images
        Gen_img = []
        phase_list = - np.angle(modamp)
        for d in range(ndirs):
            alpha = cur_k0_angle[d]
            for ph in range(nphases):
                kxL = cur_k0[d] * np.pi * np.cos(alpha)
                kyL = cur_k0[d] * np.pi * np.sin(alpha)
                kxR = -cur_k0[d] * np.pi * np.cos(alpha)
                kyR = -cur_k0[d] * np.pi * np.sin(alpha)
                phOffset = phase_list[d] + ph * phase_space
                interBeam = np.exp(1j * (kxL * X + kyL * Y + phOffset)) + np.exp(1j * (kxR * X + kyR * Y))
                temp = np.square(np.abs(interBeam))
                pattern = prctile_norm(temp)
                patterned_img = pattern * imSIM
                temp = F.fftshift(F.fft2(patterned_img)) * OTF
                Generated_img = np.abs(F.ifft2(F.ifftshift(temp)))
                Generated_img = cv2.resize(Generated_img, (Nx, Ny))
                Gen_img.append(prctile_norm(Generated_img))
        Gen_img = np.array(Gen_img)

        # perform denoise per orientation
        input_g_t = np.transpose(input_g, (1, 2, 0))
        Gen_img_t = np.transpose(Gen_img, (1, 2, 0))
        pred = []
        for d in range(ndirs):
            Gen = Gen_img_t[:, :, d * ndirs:(d + 1) * nphases]
            input = input_g_t[:, :, d * ndirs:(d + 1) * nphases]
            input1 = np.reshape(Gen, (1, input_height, input_width, nphases))
            input2 = np.reshape(input, (1, input_height, input_width, nphases))
            pr = np.squeeze(g.predict([input1, input2]))
            for pha in range(nphases):
                pred.append(np.squeeze(pr[:, :, pha]))
        pred = prctile_norm(np.array(pred))

        # save denoised images as .tif file
        pred = np.uint16(pred * 65535)
        out_file = images_path_percycle[j].replace(test_images_path, output_path)
        write_mrc(out_file, pred.transpose(1, 2, 0), header)
        # tiff.imwrite(out_file, pred)
