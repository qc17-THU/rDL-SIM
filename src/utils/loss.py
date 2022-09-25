import tensorflow as tf
from tensorflow.keras import backend as K


# --------------------------------------------------------------------------------
#                              define loss
# --------------------------------------------------------------------------------
def mse_ssim(y_true, y_pred):

    ssim_para = 1e-1  # 1e-2
    mse_para = 1

    # normalization
    x = y_true
    y = y_pred
    x = (x - K.min(x)) / (K.max(x) - K.min(x))
    y = (y - K.min(y)) / (K.max(y) - K.min(y))

    # calculate loss
    ssim = ssim_para * (1 - K.mean(tf.image.ssim(x, y, 1)))
    mse = mse_para * K.mean(K.square(y - x))

    return mse + ssim


def mse_ssim_3d(y_true, y_pred):
    ssim_para = 1e-1  # 1e-2
    mse_para = 1

    # normalization
    x = K.permute_dimensions(y_true, (0, 4, 1, 2, 3))
    y = K.permute_dimensions(y_pred, (0, 4, 1, 2, 3))
    x = (x - K.min(x)) / (K.max(x) - K.min(x))
    y = (y - K.min(y)) / (K.max(y) - K.min(y))

    # calculate loss
    ssim_loss = ssim_para * (1 - K.mean(tf.image.ssim(x, y, 1)))
    mse_loss = mse_para * K.mean(K.square(y - x))

    return mse_loss + ssim_loss


def mae_ssim(y_true, y_pred):

    ssim_para = 1e-1 # 1e-2
    mse_para = 1

    # normalization
    x = y_true
    y = y_pred
    x = (x - K.min(x)) / (K.max(x) - K.min(x))
    y = (y - K.min(y)) / (K.max(y) - K.min(y))

    # calculate loss
    ssim = ssim_para * (1 - K.mean(tf.image.ssim(x, y, 1)))
    mae = mse_para * K.abs(y - x)

    return mae + ssim


def mse_gar(y_true, y_pred):
    gamma = 1

    # mse loss
    y_pred = y_pred[..., 0]
    y_gt = y_true[..., 0]
    mse_loss = K.mean(K.square(y_gt - y_pred))

    # gar regularization
    y_gar = y_true[..., 1]
    neighbor_loss = K.mean(K.square(y_gt - y_pred - y_gar))

    return mse_loss + gamma * neighbor_loss
