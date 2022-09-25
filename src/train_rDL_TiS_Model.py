import os
import argparse
import glob
import numpy as np
import datetime
import tifffile as tiff
import tensorflow as tf
from models import *
from tensorflow.keras import optimizers
from utils.utils import ReduceLROnPlateau, prctile_norm


# --------------------------------------------------------------------------------
#                                 define parameters
# --------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# directory parameters
parser.add_argument("--root_path", type=str, default="../data_train/TiS/")
parser.add_argument("--data_folder", type=str, default="SiT-Golgi")
parser.add_argument("--save_weights_path", type=str, default="../trained_models/TiS_rDL_Denoising_Model/")
parser.add_argument("--save_weights_suffix", type=str, default="")
# model parameters
parser.add_argument("--model", type=str, default="RCAN3D_NSM")
parser.add_argument("--load_weights_flag", type=int, default=0)
# training parameters
parser.add_argument("--gpu_id", type=str, default="4")
parser.add_argument("--gpu_memory_fraction", type=float, default=0.5)
parser.add_argument("--mixed_precision", type=str, default="1")
parser.add_argument("--total_iterations", type=int, default=50000)
parser.add_argument("--sample_interval", type=int, default=1000)
parser.add_argument("--save_model_interval", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=3)
parser.add_argument("--init_lr", type=float, default=1e-4)
parser.add_argument("--lr_decay_factor", type=float, default=0.5)
parser.add_argument("--patience", type=int, default=10)
# image parameters
parser.add_argument("--input_y", type=int, default=64)
parser.add_argument("--input_x", type=int, default=64)
parser.add_argument("--input_z", type=int, default=16)
parser.add_argument("--n_channels", type=int, default=4)
parser.add_argument("--img_background", type=int, default=100)
parser.add_argument("--norm_flag", type=int, default=0)

# --------------------------------------------------------------------------------
#                          instantiation for parameters
# --------------------------------------------------------------------------------
args = parser.parse_args()
root_path = args.root_path
data_folder = args.data_folder
save_weights_path = args.save_weights_path
save_weights_suffix = args.save_weights_suffix

model = args.model
load_weights_flag = args.load_weights_flag

gpu_id = args.gpu_id
gpu_memory_fraction = args.gpu_memory_fraction
mixed_precision = args.mixed_precision
total_iterations = args.total_iterations
sample_interval = args.sample_interval
save_model_interval = args.save_model_interval
batch_size = args.batch_size
init_lr = args.init_lr
lr_decay_factor = args.lr_decay_factor
patience = args.patience

input_y = args.input_y
input_x = args.input_x
input_z = args.input_z
n_channels = args.n_channels
img_background = args.img_background
norm_flag = args.norm_flag

# define and make output dir
save_weights_path = save_weights_path + data_folder + save_weights_suffix + "/"
save_weights_file = save_weights_path + data_folder + "_rDL_Denoise"
train_img_path = root_path + data_folder + "/train/"
validate_img_path = root_path + data_folder + "/val/"
sample_path = save_weights_path + "sampled/"
log_path = save_weights_path + "graph/"
if not os.path.exists(save_weights_path):
    os.mkdir(save_weights_path)
if not os.path.exists(sample_path):
    os.mkdir(sample_path)
if not os.path.exists(log_path):
    os.mkdir(log_path)

# define GPU environment
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = mixed_precision
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# --------------------------------------------------------------------------------
#                    select and define rDL-TiS denoising model
# --------------------------------------------------------------------------------
modelFns = {'RCAN3D': RCAN3D.RCAN3D, 'RCAN3D_NSM': RCAN3D_NSM.RCAN3D}
modelFN = modelFns[model]
g = modelFN((input_y, input_x, input_z, n_channels), n_ResGroup=3, n_RCAB=5)
optimizer_g = optimizers.Adam(lr=init_lr, beta_1=0.9, beta_2=0.999, decay=1e-5)

# --------------------------------------------------------------------------------
#                         load weights and compile
# --------------------------------------------------------------------------------
if load_weights_flag:
    weights_file = glob.glob(save_weights_file + '*.h5')
    weights_file.sort()
    weights_file.sort(key=lambda i: len(i))
    print('Load existing weights: ' + weights_file[-1])
    g.load_weights(weights_file[-1])

g.compile(loss='mse', optimizer=optimizer_g)
lr_controller = ReduceLROnPlateau(model=g, factor=lr_decay_factor, patience=patience, mode='min', min_delta=1e-4,
                                  cooldown=0, min_lr=max(init_lr * 0.05, 1e-6), verbose=1)


# --------------------------------------------------------------------------------
#                              write in tensorboard
# --------------------------------------------------------------------------------
def write_log(writer, names, logs, batch_no):
    with writer.as_default():
        tf.summary.scalar(names, logs, step=batch_no)
        writer.flush()


writer = tf.summary.create_file_writer(log_path)
train_names = 'train_loss'


# --------------------------------------------------------------------------------
#                                Sample function
# --------------------------------------------------------------------------------
def sample(iter):
    path = glob.glob(validate_img_path + '*.tif')[0]
    img = tiff.imread(path).astype('float')
    img = img - img_background
    img[img < 0] = 0
    if norm_flag == 1:
        img = prctile_norm(np.array(img))
    else:
        img = np.array(img) / 65535

    img_in = np.reshape(img, (input_z, n_channels, input_y, input_x), order='F')
    img_in = np.transpose(prctile_norm(np.array(img_in)), (2, 3, 0, 1))

    pred = np.squeeze(g.predict(img_in[np.newaxis, ...]))[..., 1 + n_channels//2]
    pred = np.transpose(1e4 * prctile_norm(pred), (2, 0, 1)).astype('uint16')
    tiff.imwrite(sample_path + str(iter) + '.tif', pred, dtype='uint16')


# --------------------------------------------------------------------------------
#                                  Training
# --------------------------------------------------------------------------------
start_time = datetime.datetime.now()
# validate_nrmse = [np.Inf]
loss_perep = []
lr_controller.on_train_begin()
images_path = glob.glob(train_img_path + '*')
for iter in range(total_iterations):
    # ------------------------------------------------------------------------------
    #                              load training data
    # ------------------------------------------------------------------------------
    path = np.random.choice(images_path, size=batch_size)
    img_batch = []
    for curp in path:
        cur_img = tiff.imread(curp).astype('float')
        cur_img = cur_img - img_background
        cur_img[cur_img < 0] = 0
        if norm_flag == 1:
            cur_img = prctile_norm(np.array(cur_img))
        else:
            cur_img = np.array(cur_img) / 65535
        img_batch.append(cur_img)

    img_batch = np.reshape(img_batch, (batch_size, input_z, n_channels * 2, input_y, input_x), order='F')
    img_batch = np.transpose(img_batch, (0, 3, 4, 1, 2))

    # ------------------------------------------------------------------------------
    #                                 training
    # ------------------------------------------------------------------------------
    img_in = img_batch[..., ::2]
    img_gt = img_batch[..., 1::2]
    loss_train = g.train_on_batch(img_in, img_gt)
    loss_perep.append(loss_train)

    elapsed_time = datetime.datetime.now() - start_time
    print("%d iter: time: %s, g_loss = %s" % (iter + 1, elapsed_time, loss_train))

    # ------------------------------------------------------------------------------
    #                           save model and sampling
    # ------------------------------------------------------------------------------
    if (iter + 1) % save_model_interval == 0:
        g.save_weights(save_weights_file + '_Latest.h5')
        write_log(writer, train_names, np.mean(loss_perep), iter + 1)
        curlr = lr_controller.on_epoch_end(iter + 1, np.mean(loss_perep))
        write_log(writer, 'lr', curlr, iter + 1)
        loss_perep = []

    if (iter + 1) % sample_interval == 0:
        images_path = glob.glob(train_img_path + '*')
        sample(iter + 1)


