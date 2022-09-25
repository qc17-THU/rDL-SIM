import os
import glob
import datetime
import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from models import *
from utils.utils import ReduceLROnPlateau, data_loader, data_loader_rDL, prctile_norm, cal_comp
from utils.loss import mse_ssim
from scipy import signal
import numpy.fft as F
import cv2 as cv
from sim_fitting.Parameters_2DSIM import parameters
from sim_fitting.CalModamp_2DSIM import cal_modamp
from sim_fitting.read_otf import read_otf
from utils.read_mrc import read_mrc
from scipy.interpolate import interp1d
from PIL import Image


# --------------------------------------------------------------------------------
#                                 define parameters
# --------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# directory parameters
parser.add_argument("--root_path", type=str, default="../data_train/rDL-SIM/DN/")
parser.add_argument("--data_folder", type=str, default="Microtubules")
parser.add_argument("--save_weights_path", type=str, default="../trained_models/rDL_Denoising_Module/")
parser.add_argument("--save_weights_suffix", type=str, default="")
# model parameters
parser.add_argument("--load_weights_flag", type=int, default=0)
parser.add_argument("--denoise_model", type=str, default="rDL_Denoiser")
parser.add_argument("--load_sr_module_dir", type=str, default="../trained_models/SR_Inference_Module/")
parser.add_argument("--load_sr_module_filter", type=str, default="*Best.h5")
parser.add_argument("--sr_model", type=str, default="DFCAN")
# training parameters
parser.add_argument("--gpu_id", type=str, default="4")
parser.add_argument("--gpu_memory_fraction", type=float, default=0.5)
parser.add_argument("--mixed_precision", type=str, default="1")
parser.add_argument("--total_iterations", type=int, default=200000)
parser.add_argument("--sample_interval", type=int, default=1000)
parser.add_argument("--validate_interval", type=int, default=2000)
parser.add_argument("--validate_num", type=int, default=500)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--start_lr", type=float, default=1e-4)
parser.add_argument("--patience", type=int, default=15)
parser.add_argument("--lr_decay_factor", type=float, default=0.5)
# image parameters
parser.add_argument("--input_height", type=int, default=128)
parser.add_argument("--input_width", type=int, default=128)
# SIM parameters
parser.add_argument("--wave_length", type=int, default=488)
parser.add_argument("--excNA", type=float, default=1.35)
parser.add_argument("--ndirs", type=int, default=3)
parser.add_argument("--nphases", type=int, default=3)
parser.add_argument("--OTF_path_488", type=str, default='./sim_fitting/OTF/TIRF488_cam1_0_z30_OTF2d.mrc')
parser.add_argument("--OTF_path_560", type=str, default='./sim_fitting/OTF/TIRF560_cam2_0_z21_OTF2d.mrc')
parser.add_argument("--OTF_path_647", type=str, default='./sim_fitting/OTF/TIRF647_cam2_0_z21_OTF2d.mrc')

# --------------------------------------------------------------------------------
#                          instantiation for parameters
# --------------------------------------------------------------------------------
args = parser.parse_args()
root_path = args.root_path
data_folder = args.data_folder
save_weights_path = args.save_weights_path
save_weights_suffix = args.save_weights_suffix

load_weights_flag = args.load_weights_flag
denoise_model = args.denoise_model
load_sr_module_dir = args.load_sr_module_dir
load_sr_module_filter = args.load_sr_module_filter
sr_model = args.sr_model

gpu_id = args.gpu_id
gpu_memory_fraction = args.gpu_memory_fraction
mixed_precision = args.mixed_precision
total_iterations = args.total_iterations
sample_interval = args.sample_interval
validate_interval = args.validate_interval
validate_num = args.validate_num
batch_size = args.batch_size
start_lr = args.start_lr
lr_decay_factor = args.lr_decay_factor
patience = args.patience

input_height = args.input_height
input_width = args.input_width
nphases = args.nphases
ndirs = args.ndirs

# define and make output dir
save_weights_path = save_weights_path + data_folder + save_weights_suffix + "/"
save_weights_file = save_weights_path + data_folder + "_rDL_Denoise"
train_img_path = root_path + data_folder + "/train/"
train_gt_path = root_path + data_folder + "/train_gt/"
validate_img_path = root_path + data_folder + "/val/"
validate_gt_path = root_path + data_folder + "/val_gt/"
sample_path = save_weights_path + "sampled/"
log_path = save_weights_path + "graph/"
if not os.path.exists(save_weights_path):
    os.mkdir(save_weights_path)
if not os.path.exists(sample_path):
    os.mkdir(sample_path)
if not os.path.exists(log_path):
    os.mkdir(log_path)

# define SIM parameters
wave_length = args.wave_length
excNA = args.excNA
OTF_path_488 = args.OTF_path_488
OTF_path_560 = args.OTF_path_560
OTF_path_647 = args.OTF_path_647
OTF_path_list = {488: OTF_path_488, 560: OTF_path_560, 647: OTF_path_647}
pParam = parameters(input_height, input_width, wave_length * 1e-3, excNA, setup=0)

# define GPU environment
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = mixed_precision
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# --------------------------------------------------------------------------------
#                           select models and optimizer
# --------------------------------------------------------------------------------
modelFns = {'DFCAN': DFCAN.DFCAN, 'rDL_Denoiser': rDL_Denoiser.Denoiser, 'rDL_Denoiser_NSM': rDL_Denoiser_NSM.Denoiser}
modelFN_dn = modelFns[denoise_model]
modelFN_sr = modelFns[sr_model]

optimizer_dn = optimizers.Adam(lr=start_lr, beta_1=0.9, beta_2=0.999)
optimizer_sr = optimizers.Adam(lr=start_lr, beta_1=0.9, beta_2=0.999)

# --------------------------------------------------------------------------------
#                      define models and load weights
# --------------------------------------------------------------------------------
g = modelFN_dn((input_height, input_width, nphases))
if load_weights_flag:
    weights_file = glob.glob(save_weights_file + '*.h5')
    weights_file.sort()
    weights_file.sort(key=lambda i: len(i))
    print('Load existing weights: ' + weights_file[-1])
    g.load_weights(weights_file[-1])

g.compile(loss=mse_ssim, optimizer=optimizer_dn)
lr_controller = ReduceLROnPlateau(model=g, factor=lr_decay_factor, patience=patience, mode='min', min_delta=1e-4,
                                  cooldown=0, min_lr=max(start_lr*0.1, 1e-5), verbose=1)

p = modelFN_sr((input_height, input_width, ndirs*nphases))
weights_file = glob.glob(load_sr_module_dir + data_folder + "/" + load_sr_module_filter)
weights_file.sort()
weights_file.sort(key=lambda i: len(i))
print('Load weights of SR inference module: ' + weights_file[-1])
p.load_weights(weights_file[-1])
p.compile(loss=None, optimizer=optimizer_sr)


# --------------------------------------------------------------------------------
#                              write in tensorboard
# --------------------------------------------------------------------------------
def write_log(writer, names, logs, batch_no):
    with writer.as_default():
        tf.summary.scalar(names, logs, step=batch_no)
        writer.flush()


log_path = save_weights_path + '/graph'
if not os.path.exists(log_path):
    os.mkdir(log_path)
writer = tf.summary.create_file_writer(log_path)
train_names = 'train_loss'
val_names = ['val_mse', 'val_nrmse', 'val_SSIM', 'val_PSNR']


# --------------------------------------------------------------------------------
#                         predefine OTF and other parameters
# --------------------------------------------------------------------------------
# define parameters
[Nx, Ny] = [pParam.Nx, pParam.Ny]
[dx, dy, dxy] = [pParam.dx, pParam.dy, pParam.dxy]
[dkx, dky, dkr] = [pParam.dkx, pParam.dky, pParam.dkr]
[nphases, ndirs] = [pParam.nphases, pParam.ndirs]
space = pParam.space
scale = pParam.scale
phase_space = 2 * np.pi / nphases

[Nx_hr, Ny_hr] = [Nx, Ny] * scale
[dx_hr, dy_hr] = [x / scale for x in [dx, dy]]

xx = dx_hr * np.arange(-Nx_hr / 2, Nx_hr / 2, 1)
yy = dy_hr * np.arange(-Ny_hr / 2, Ny_hr / 2, 1)
[X, Y] = np.meshgrid(xx, yy)

# read OTF and PSF
OTF, prol_OTF, PSF = read_otf(OTF_path_list[wave_length], Nx_hr, Ny_hr, dkx, dky, dkr)


# --------------------------------------------------------------------------------
#                        Validation and sampling function
# --------------------------------------------------------------------------------
def validate(iter, sample=0):
    validate_path = glob.glob(validate_img_path + '*')
    mses, nrmses, psnrs, ssims = [], [], [], []
    if sample:
        validate_path = np.random.choice(validate_path, size=3, replace=False)
        input_show, pred_show, gt_show = [], [], []
    else:
        if validate_num < validate_path.__len__():
            validate_path = validate_path[0:validate_num]

    for path in validate_path:
        img_in, img_gt = data_loader_rDL([path], validate_img_path, validate_gt_path, 1)
        img_in = np.squeeze(img_in)
        img_gt = np.squeeze(img_gt)
        cur_k0, modamp = cal_modamp(np.array(img_in).transpose((1, 2, 0)).astype(np.float), prol_OTF, pParam)
        cur_k0_angle = np.array(np.arctan(cur_k0[:, 1] / cur_k0[:, 0]))
        cur_k0_angle[1:ndirs] = cur_k0_angle[1:ndirs] + np.pi
        cur_k0_angle = -(cur_k0_angle - np.pi / 2)
        for nd in range(ndirs):
            if np.abs(cur_k0_angle[nd] - pParam.k0angle_g[nd]) > 0.05:
                cur_k0_angle[nd] = pParam.k0angle_g[nd]
        cur_k0 = np.sqrt(np.sum(np.square(cur_k0), 1))
        given_k0 = 1 / pParam.space
        cur_k0[np.abs(cur_k0 - given_k0) > 0.1] = given_k0
        # img_in = img_in / 65535
        # img_gt = img_gt / 65535
        img_in = prctile_norm(img_in)
        img_gt = prctile_norm(img_gt)
        
        img_SR = p.predict(img_in.transpose((1, 2, 0)).reshape((1, input_height, input_width, -1)))
        img_SR = prctile_norm(np.squeeze(img_SR))
        img_SR = cv.resize(img_SR, (Ny_hr, Nx_hr))
        
        # intensity equalization for each orientation
        mean_th_in = np.mean(img_in[:nphases, :, :])
        for d in range(1, ndirs):
            data_d = img_in[d*nphases:(d+1)*nphases, :, :]
            img_in[d*nphases:(d+1)*nphases, :, :] = data_d * mean_th_in / np.mean(data_d)
        mean_th_gt = np.mean(img_gt[:nphases, :, :])
        for d in range(ndirs):
            data_d = img_gt[d*nphases:(d+1)*nphases, :, :]
            img_gt[d*nphases:(d+1)*nphases, :, :] = data_d * mean_th_gt / np.mean(data_d)
        
        # generate pattern-modulated images
        phase_list = -np.angle(modamp)
        img_gen = []
        for d in range(ndirs):
            alpha = cur_k0_angle[d]
            for i in range(nphases):
                kxL = cur_k0[d] * np.pi * np.cos(alpha)
                kyL = cur_k0[d] * np.pi * np.sin(alpha)
                kxR = -cur_k0[d] * np.pi * np.cos(alpha)
                kyR = -cur_k0[d] * np.pi * np.sin(alpha)
                phOffset = phase_list[d] + i * phase_space
                interBeam = np.exp(1j * (kxL * X + kyL * Y + phOffset)) + np.exp(1j * (kxR * X + kyR * Y))
                pattern = prctile_norm(np.square(np.abs(interBeam)))
                patterned_img_fft = F.fftshift(F.fft2(pattern * img_SR)) * OTF
                modulated_img = np.abs(F.ifft2(F.ifftshift(patterned_img_fft)))
                modulated_img = cv.resize(modulated_img, (Ny, Nx))
                img_gen.append(modulated_img)
        img_gen = prctile_norm(np.array(img_gen))

        # prediction and calculate multiple metrics
        img_in = np.transpose(img_in, (1, 2, 0))
        img_gen = np.transpose(img_gen, (1, 2, 0))
        input_MPE = img_gen[:, :, 0:nphases]
        input_PFE = img_in[:, :, 0:nphases]
        img_pred = g.predict([input_MPE[np.newaxis, ...], input_PFE[np.newaxis, ...]])
        img_pred = np.transpose(np.squeeze(img_pred.astype(np.float)), (2, 0, 1))
        img_gt = img_gt[0:nphases, :, :]
        mses, nrmses, psnrs, ssims = cal_comp(np.squeeze(img_gt), img_pred, mses, nrmses, psnrs, ssims)

        if sample:
            input_PFE = np.transpose(np.squeeze(input_PFE), (2, 0, 1))
            input_show.append(input_PFE[0, :, :])
            gt_show.append(img_gt[0, :, :])
            pred_show.append(img_pred[0, :, :])

    if sample:  # only sample
        r, c = 3, 3
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for row in range(r):
            axs[row, 1].set_title(
                ' PSNR=%.4f, SSIM=%.4f' % (psnrs[row * nphases], ssims[row * nphases]))
            for col, image in enumerate([input_show, pred_show, gt_show]):
                axs[row, col].imshow(np.squeeze(image[row]))
                axs[row, col].axis('off')
            cnt += 1
        fig.savefig(sample_path + '%d.png' % iter)
        plt.close()
    else:  # validation
        # if best, save _Best.iter, else save _Latest.iter
        if min(validate_nrmse) > np.mean(nrmses):
            g.save_weights(save_weights_file + '_Best.h5')
        else:
            g.save_weights(save_weights_file + '_Latest.h5')
        validate_nrmse.append(np.mean(nrmses))

        curlr = lr_controller.on_epoch_end(iter, np.mean(nrmses))
        write_log(writer, val_names[0], np.mean(mses), iter)
        write_log(writer, val_names[1], np.mean(nrmses), iter)
        write_log(writer, val_names[2], np.mean(ssims), iter)
        write_log(writer, val_names[3], np.mean(psnrs), iter)
        write_log(writer, 'lr', curlr, iter)


# --------------------------------------------------------------------------------
#                                 Model Training
# --------------------------------------------------------------------------------
start_time = datetime.datetime.now()
loss_perep = []
validate_nrmse = [np.Inf]
lr_controller.on_train_begin()
images_path = glob.glob(train_img_path + '*')

for iter in range(total_iterations):
    # ------------------------------------------------------------------------------
    #                     load data and calculate SIM  & phase
    # ------------------------------------------------------------------------------
    img_in, img_gt = data_loader_rDL(images_path, train_img_path, train_gt_path, batch_size)
    img_in = np.squeeze(img_in)
    img_gt = np.squeeze(img_gt)
    cur_k0, modamp = cal_modamp(np.array(img_gt).transpose((1, 2, 0)).astype(np.float), prol_OTF, pParam)
    while np.mean(np.abs(modamp)) < 0:
        img_in, img_gt = data_loader_rDL(images_path, train_img_path, train_gt_path, batch_size)
        cur_k0, modamp = cal_modamp(np.array(img_gt).transpose((1, 2, 0)).astype(np.float), prol_OTF, pParam)

    cur_k0_angle = np.array(np.arctan(cur_k0[:, 1] / cur_k0[:, 0]))
    cur_k0_angle[1:ndirs] = cur_k0_angle[1:ndirs] + np.pi
    cur_k0_angle = -(cur_k0_angle - np.pi/2)
    for nd in range(ndirs):
        if np.abs(cur_k0_angle[nd] - pParam.k0angle_g[nd]) > 0.05:
            cur_k0_angle[nd] = pParam.k0angle_g[nd]
    cur_k0 = np.sqrt(np.sum(np.square(cur_k0), 1))
    given_k0 = 1 / pParam.space
    cur_k0[np.abs(cur_k0 - given_k0) > 0.1] = given_k0
    # img_in = img_in / 65535
    # img_gt = img_gt / 65535
    img_in = prctile_norm(img_in)
    img_gt = prctile_norm(img_gt)

    im_toSIM = prctile_norm(img_in.transpose((1, 2, 0)))
    img_SR = p.predict(im_toSIM.reshape((1, input_height, input_width, 9)))
    img_SR = prctile_norm(np.squeeze(img_SR))
    img_SR = cv.resize(img_SR, (Ny_hr, Nx_hr))

    # ------------------------------------------------------------------------------
    #                   intensity equalization for each orientation
    # ------------------------------------------------------------------------------
    mean_th_in = np.mean(img_in[:nphases, :, :])
    for d in range(1, ndirs):
        data_d = img_in[d * nphases:(d + 1) * nphases, :, :]
        img_in[d * nphases:(d + 1) * nphases, :, :] = data_d * mean_th_in / np.mean(data_d)
    mean_th_gt = np.mean(img_gt[:nphases, :, :])
    for d in range(ndirs):
        data_d = img_gt[d * nphases:(d + 1) * nphases, :, :]
        img_gt[d * nphases:(d + 1) * nphases, :, :] = data_d * mean_th_gt / np.mean(data_d)

    # ------------------------------------------------------------------------------
    #                        generate pattern-modulated images
    # ------------------------------------------------------------------------------
    phase_list = - np.angle(modamp)
    img_gen = []
    for d in range(ndirs):
        alpha = cur_k0_angle[d]
        for i in range(nphases):
            kxL = cur_k0[d] * np.pi * np.cos(alpha)
            kyL = cur_k0[d] * np.pi * np.sin(alpha)
            kxR = -cur_k0[d] * np.pi * np.cos(alpha)
            kyR = -cur_k0[d] * np.pi * np.sin(alpha)
            phOffset = phase_list[d] + i * phase_space
            interBeam = np.exp(1j * (kxL * X + kyL * Y + phOffset)) + np.exp(1j * (kxR * X + kyR * Y))
            pattern = prctile_norm(np.square(np.abs(interBeam)))
            patterned_img_fft = F.fftshift(F.fft2(pattern * img_SR)) * OTF
            modulated_img = np.abs(F.ifft2(F.ifftshift(patterned_img_fft)))
            modulated_img = cv.resize(modulated_img, (Ny, Nx))
            img_gen.append(modulated_img)
    img_gen = prctile_norm(np.array(img_gen))

    # ------------------------------------------------------------------------------
    #                           train rDL denoising module
    # ------------------------------------------------------------------------------
    img_in = np.transpose(img_in, (1, 2, 0))
    img_gt = np.transpose(img_gt, (1, 2, 0))
    img_gen = np.transpose(img_gen, (1, 2, 0))
    input_MPE_batch = []
    input_PFE_batch = []
    gt_batch = []
    for i in range(ndirs):
        input_MPE_batch.append(img_gen[:, :, i * nphases:(i + 1) * nphases])
        input_PFE_batch.append(img_in[:, :, i * nphases:(i + 1) * nphases])
        gt_batch.append(img_gt[:, :, i * nphases:(i + 1) * nphases])
    input_MPE_batch = np.array(input_MPE_batch)
    input_PFE_batch = np.array(input_PFE_batch)
    gt_batch = np.array(gt_batch)

    loss_train = g.train_on_batch([input_MPE_batch, input_PFE_batch], gt_batch)
    loss_perep.append(loss_train)

    modamp_abs = np.mean(np.abs(modamp))
    elapsed_time = datetime.datetime.now() - start_time
    print("%d iter: time: %s, g_loss = %s, modamp = %s" % (iter + 1, elapsed_time, loss_train, modamp_abs))

    # ------------------------------------------------------------------------------
    #                          Sampling and Validation
    # ------------------------------------------------------------------------------
    if (iter + 1) % sample_interval == 0:
        images_path = glob.glob(train_img_path + '*')
        validate(iter + 1, sample=1)

    if (iter + 1) % validate_interval == 0:
        validate(iter + 1, sample=0)
        write_log(writer, train_names, np.mean(loss_perep), iter + 1)
        loss_perep = []


