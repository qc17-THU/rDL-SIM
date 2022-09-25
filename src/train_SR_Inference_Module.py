import os
import glob
import datetime
import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from models import *
from utils.utils import ReduceLROnPlateau, data_loader, prctile_norm, cal_comp
from utils.loss import mse_ssim

# --------------------------------------------------------------------------------
#                                 define parameters
# --------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# directory parameters
parser.add_argument("--root_path", type=str, default="../data_train/rDL-SIM/SR/")
parser.add_argument("--data_folder", type=str, default="Microtubules")
parser.add_argument("--save_weights_path", type=str, default="../trained_models/SR_Inference_Module/")
parser.add_argument("--save_weights_suffix", type=str, default="")
# model parameters
parser.add_argument("--load_weights_flag", type=int, default=0)
parser.add_argument("--model_name", type=str, default="DFCAN")
# training parameters
parser.add_argument("--gpu_id", type=str, default="4")
parser.add_argument("--gpu_memory_fraction", type=float, default=0.5)
parser.add_argument("--mixed_precision", type=str, default="1")
parser.add_argument("--total_iterations", type=int, default=100000)
parser.add_argument("--sample_interval", type=int, default=1000)
parser.add_argument("--validate_interval", type=int, default=2000)
parser.add_argument("--validate_num", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--start_lr", type=float, default=1e-4)
parser.add_argument("--lr_decay_factor", type=float, default=0.5)
# image parameters
parser.add_argument("--input_height", type=int, default=128)
parser.add_argument("--input_width", type=int, default=128)
parser.add_argument("--input_channels", type=int, default=9)
parser.add_argument("--scale_factor", type=int, default=2)
parser.add_argument("--norm_flag", type=int, default=1)

# --------------------------------------------------------------------------------
#                          instantiation for parameters
# --------------------------------------------------------------------------------
args = parser.parse_args()
root_path = args.root_path
data_folder = args.data_folder
save_weights_path = args.save_weights_path
save_weights_suffix = args.save_weights_suffix

load_weights_flag = args.load_weights_flag
model_name = args.model_name

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

input_height = args.input_height
input_width = args.input_width
input_channels = args.input_channels
scale_factor = args.scale_factor
norm_flag = args.norm_flag

# define and make output dir
save_weights_path = save_weights_path + data_folder + save_weights_suffix + "/"
save_weights_file = save_weights_path + data_folder + "_SR"
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

# define GPU environment
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = mixed_precision
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# --------------------------------------------------------------------------------
#                           select models and optimizer
# --------------------------------------------------------------------------------
modelFns = {'DFCAN': DFCAN.DFCAN}
modelFN = modelFns[model_name]
optimizer = optimizers.Adam(lr=start_lr, beta_1=0.9, beta_2=0.999)

# --------------------------------------------------------------------------------
#                              define combined model
# --------------------------------------------------------------------------------
g = modelFN((input_height, input_width, input_channels), scale=scale_factor)
g.compile(loss=mse_ssim, optimizer=optimizer)
lr_controller = ReduceLROnPlateau(model=g, factor=lr_decay_factor, patience=10, mode='min', min_delta=1e-4,
                                    cooldown=0, min_lr=start_lr*0.1, verbose=1)

# --------------------------------------------------------------------------------
#                               define log writer
# --------------------------------------------------------------------------------
writer = tf.summary.create_file_writer(log_path)
train_names = 'training_loss'
val_names = ['val_loss', 'val_SSIM', 'val_PSNR', 'val_NRMSE']


def write_log(writer, names, logs, batch_no):
    with writer.as_default():
        tf.summary.scalar(names, logs, step=batch_no)
        writer.flush()


# --------------------------------------------------------------------------------
#                             if existed, load weights
# --------------------------------------------------------------------------------
if load_weights_flag:
    weights_file = glob.glob(save_weights_file + '*.h5')
    weights_file.sort()
    weights_file.sort(key=lambda i: len(i))
    print('Load existing weights: ' + weights_file[-1])
    g.load_weights(weights_file[-1])


# --------------------------------------------------------------------------------
#                             Sample and validate
# --------------------------------------------------------------------------------
def validate(iter, sample=0):
    validate_img_files = glob.glob(validate_img_path + '*')
    validate_img_files.sort()
    if sample == 1:  # sampling
        r, c = 3, 3
        mses, nrmses, psnrs, ssims = [], [], [], []
        img_show, gt_show, output_show = [], [], []
        validate_path = np.random.choice(validate_img_files, size=r)
        for path in validate_path:
            [img, gt] = data_loader([path], validate_img_path, validate_gt_path, input_height,
                                    input_width, 1, norm_flag=norm_flag, scale=scale_factor)
            img = prctile_norm(img)
            output = prctile_norm(np.squeeze(g.predict(img)))
            mses, nrmses, psnrs, ssims = cal_comp(gt, output, mses, nrmses, psnrs, ssims)
            img_show.append(np.squeeze(np.mean(img, 3)))
            gt_show.append(np.squeeze(gt))
            output_show.append(output)
        # show some examples
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for row in range(r):
            axs[row, 1].set_title('MSE=%.4f, SSIM=%.4f, PSNR=%.4f' % (mses[row], ssims[row], psnrs[row]))
            for col, image in enumerate([img_show, output_show, gt_show]):
                axs[row, col].imshow(np.squeeze(image[row]))
                axs[row, col].axis('off')
            cnt += 1
        fig.savefig(sample_path + '%d.png' % iter)
        plt.close()
    else:  # validation
        if validate_num < validate_img_files.__len__():
            validate_img_files = validate_img_files[0:validate_num]
        mses, nrmses, psnrs, ssims = [], [], [], []
        for file in validate_img_files:
            [img, gt] = data_loader([file], validate_img_path, validate_gt_path, input_height,
                                    input_width, 1, norm_flag=norm_flag, scale=scale_factor)
            img = prctile_norm(img)
            output = prctile_norm(np.squeeze(g.predict(img)))
            mses, nrmses, psnrs, ssims = cal_comp(gt, output, mses, nrmses, psnrs, ssims)

        # if best, save _Best.h5, else save _Latest.h5
        if min(validate_nrmse) > np.mean(nrmses):
            g.save_weights(save_weights_file + '_Best.h5')
        else:
            g.save_weights(save_weights_file + '_Latest.h5')
        validate_nrmse.append(np.mean(nrmses))

        # write log of mse, ssim, psnr, nrmse
        write_log(writer, val_names[0], np.mean(mses), iter)
        write_log(writer, val_names[1], np.mean(ssims), iter)
        write_log(writer, val_names[2], np.mean(psnrs), iter)
        write_log(writer, val_names[3], np.mean(nrmses), iter)


# --------------------------------------------------------------------------------
#                                    training
# --------------------------------------------------------------------------------
start_time = datetime.datetime.now()
loss_perep = []
validate_nrmse = [np.Inf]
lr_controller.on_train_begin()
images_path = glob.glob(train_img_path + '*')
for iter in range(total_iterations):
    # ---------------------------------------------
    #         train SR inference module
    # ---------------------------------------------
    img_in, img_gt = data_loader(images_path, train_img_path, train_gt_path, input_height, input_width,
                                 batch_size, norm_flag=norm_flag, scale=scale_factor)
    img_in = prctile_norm(img_in)
    img_gt = prctile_norm(img_gt)
    loss_train = g.train_on_batch(img_in, img_gt)
    loss_perep.append(loss_train)

    elapsed_time = datetime.datetime.now() - start_time
    print("%d iter: time: %s, g_loss = %s" % (iter + 1, elapsed_time, loss_train))

    # ---------------------------------------------
    #          sampling and validation
    # ---------------------------------------------
    if (iter + 1) % sample_interval == 0:
        images_path = glob.glob(train_img_path + '*')
        validate(iter + 1, sample=1)

    if (iter + 1) % validate_interval == 0:
        validate(iter + 1, sample=0)
        curlr = lr_controller.on_epoch_end(iter + 1, np.mean(loss_perep))
        write_log(writer, 'lr', curlr, iter + 1)
        write_log(writer, train_names, np.mean(loss_perep), iter + 1)
        loss_record = []
