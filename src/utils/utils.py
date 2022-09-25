import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import glob
import scipy
import cv2 as cv
import imageio
import warnings
import tifffile as tiff
from .read_mrc import read_mrc
from skimage.measure import compare_mse, compare_nrmse, compare_psnr, compare_ssim


# ---------------------------------------------------------------------------------------
#                                training strategy
# ---------------------------------------------------------------------------------------
class ReduceLROnPlateau():
    def __init__(self, model, curmonitor=np.Inf, factor=0.1, patience=10, mode='min',
                 min_delta=1e-4, cooldown=0, min_lr=0, verbose=1,
                 **kwargs):

        self.curmonitor = curmonitor
        if factor > 1.0:
            raise ValueError('ReduceLROnPlateau does not support a factor > 1.0.')

        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.model = model
        self.verbose = verbose
        self.monitor_op = None
        self._reset()

    def _reset(self):
        if self.mode == 'min':
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def update_monitor(self, curmonitor):
        self.curmonitor = curmonitor

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, curmonitor):
        curlr = K.get_value(self.model.optimizer.lr)
        self.curmonitor = curmonitor
        if self.curmonitor is None:
            warnings.warn('errro input of monitor', RuntimeWarning)
        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(self.curmonitor, self.best):
                self.best = self.curmonitor
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = float(K.get_value(self.model.optimizer.lr))
                    if old_lr > self.min_lr:
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        K.set_value(self.model.optimizer.lr, new_lr)
                        if self.verbose > 0:
                            print('\nEpoch %05d: ReduceLROnPlateau reducing '
                                  'learning rate to %s.' % (epoch + 1, new_lr))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0
        return curlr

    def in_cooldown(self):
        return self.cooldown_counter > 0


# ---------------------------------------------------------------------------------------
#                               Data processing tools
# ---------------------------------------------------------------------------------------
def data_loader(images_path, data_path, gt_path, height, width, batch_size, norm_flag=1, resize_flag=0, scale=2, wf=0):
    batch_images_path = np.random.choice(images_path, size=batch_size)
    image_batch = []
    gt_batch = []
    for path in batch_images_path:
        if path[-3:] == 'tif':
            curBatch = tiff.imread(path)
            path_gt = path.replace(data_path, gt_path)
            gt = imageio.imread(path_gt).astype(np.float)
        else:
            img_path = glob.glob(path + '/*.tif')
            img_path.sort()
            curBatch = []
            for cur in img_path:
                img = imageio.imread(cur).astype(np.float)
                if resize_flag == 1:
                    img = cv.resize(img, (height * scale, width * scale))
                curBatch.append(img)
            path_gt = path.replace(data_path, gt_path) + '.tif'
            gt = imageio.imread(path_gt).astype(np.float)

        if norm_flag:
            curBatch = prctile_norm(np.array(curBatch))
            gt = prctile_norm(gt)
        else:
            curBatch = np.array(curBatch) / 65535
            gt = gt / 65535
        image_batch.append(curBatch)
        gt_batch.append(gt)

    image_batch = np.array(image_batch)
    gt_batch = np.array(gt_batch)

    image_batch = np.transpose(image_batch, (0, 2, 3, 1))
    gt_batch = gt_batch.reshape((batch_size, width*scale, height*scale, 1))

    if wf == 1:
        image_batch = np.mean(image_batch, 3)
        for b in range(batch_size):
            image_batch[b, :, :] = prctile_norm(image_batch[b, :, :])
        image_batch = image_batch[:, :, :, np.newaxis]

    return image_batch, gt_batch


def data_loader_rDL(images_path, data_path, gt_path, batch_size=1):
    batch_images_path = np.random.choice(images_path, size=batch_size, replace=False)
    image_batch = []
    gt_batch = []
    for path in batch_images_path:
        if path[-3:] == 'tif':
            image = tiff.imread(path)
            path_gt = path.replace(data_path, gt_path)
            gt = tiff.imread(path_gt)
        else:
            imgfile = glob.glob(path + '/*.tif')
            imgfile.sort()
            image = []
            gt = []
            for file in imgfile:
                img = imageio.imread(file).astype(np.float)
                image.append(img)
            path_gt = path.replace(data_path, gt_path)
            imgfile = glob.glob(path_gt + '/*.tif')
            imgfile.sort()
            for file in imgfile:
                img = imageio.imread(file).astype(np.float)
                gt.append(img)
        image_batch.append(image)
        gt_batch.append(gt)

    image_batch = np.array(image_batch).astype(np.float)
    gt_batch = np.array(gt_batch).astype(np.float)

    return image_batch, gt_batch


def prctile_norm(x, min_prc=0, max_prc=100):
    y = (x-np.percentile(x, min_prc))/(np.percentile(x, max_prc)-np.percentile(x, min_prc)+1e-7)
    return y


def cal_comp(gt, pr, mses=None, nrmses=None, psnrs=None, ssims=None):
    if ssims is None:
        ssims = []
    if psnrs is None:
        psnrs = []
    if nrmses is None:
        nrmses = []
    if mses is None:
        mses = []
    gt, pr = np.squeeze(gt), np.squeeze(pr)
    gt = gt.astype(np.float32)
    if gt.ndim == 2:
        n = 1
        gt = np.reshape(gt, (1, gt.shape[0], gt.shape[1]))
        pr = np.reshape(pr, (1, pr.shape[0], pr.shape[1]))
    else:
        n = np.size(gt, 0)

    for i in range(n):
        mses.append(compare_mse(prctile_norm(np.squeeze(gt[i])), prctile_norm(np.squeeze(pr[i]))))
        nrmses.append(compare_nrmse(prctile_norm(np.squeeze(gt[i])), prctile_norm(np.squeeze(pr[i]))))
        psnrs.append(compare_psnr(prctile_norm(np.squeeze(gt[i])), prctile_norm(np.squeeze(pr[i])), 1))
        ssims.append(compare_ssim(prctile_norm(np.squeeze(gt[i])), prctile_norm(np.squeeze(pr[i]))))
    return mses, nrmses, psnrs, ssims
