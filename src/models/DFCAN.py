from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Input, add, multiply, Lambda
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np


def gelu(x):
    cdf = 0.5 * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))
    return x * cdf


def apodize2d(img, napodize=10):
    bs, ny, nx, ch = img.get_shape().as_list()
    img_apo = img[:, napodize:ny-napodize, :, :]

    imageUp = img[:, 0:napodize, :, :]
    imageDown = img[:, ny-napodize:, :, :]
    diff = (imageDown[:, -1::-1, :, :] - imageUp) / 2
    l = np.arange(napodize)
    fact_raw = 1 - np.sin((l + 0.5) / napodize * np.pi / 2)
    fact = fact_raw[np.newaxis, :, np.newaxis, np.newaxis]
    fact = tf.convert_to_tensor(fact, dtype=tf.float32)
    fact = tf.tile(fact, [tf.shape(img)[0], 1, nx, ch])
    factor = diff * fact
    imageUp = tf.add(imageUp, factor)
    imageDown = tf.subtract(imageDown, factor[:, -1::-1, :, :])
    img_apo = tf.concat([imageUp, img_apo, imageDown], axis=1)

    imageLeft = img_apo[:, :, 0:napodize, :]
    imageRight = img_apo[:, :, nx-napodize:, :]
    img_apo = img_apo[:, :, napodize:nx-napodize, :]
    diff = (imageRight[:, :, -1::-1, :] - imageLeft) / 2
    fact = fact_raw[np.newaxis, np.newaxis, :, np.newaxis]
    fact = tf.convert_to_tensor(fact, dtype=tf.float32)
    fact = tf.tile(fact, [tf.shape(img)[0], ny, 1, ch])
    factor = diff * fact
    imageLeft = tf.add(imageLeft, factor)
    imageRight = tf.subtract(imageRight, factor[:, :, -1::-1, :])
    img_apo = tf.concat([imageLeft, img_apo, imageRight], axis=2)

    return img_apo


def fft2(input):
    input = apodize2d(input, napodize=10)
    temp = K.permute_dimensions(input, (0, 3, 1, 2))
    fft = tf.signal.fft2d(tf.complex(temp, tf.zeros_like(temp)))
    absfft = tf.pow(tf.abs(fft)+1e-8, 0.1)
    output = K.permute_dimensions(absfft, (0, 2, 3, 1))
    return output


def fftshift(input):
    bs, h, w, ch = input.get_shape().as_list()
    fs11 = input[:, -h // 2:h, -w // 2:w, :]
    fs12 = input[:, -h // 2:h, 0:w // 2, :]
    fs21 = input[:, 0:h // 2, -w // 2:w, :]
    fs22 = input[:, 0:h // 2, 0:w // 2, :]
    output = tf.concat([tf.concat([fs11, fs21], axis=1), tf.concat([fs12, fs22], axis=1)], axis=2)
    output = tf.image.resize(output, (128, 128))
    return output


def pixelshuffle(layer_in, scale):
    return tf.nn.depth_to_space(layer_in, block_size=scale)


def global_average_pooling(input):
    return tf.reduce_mean(input, axis=(1, 2), keepdims=True)


def FCALayer(input, channel, reduction=16):
    absfft1 = Lambda(fft2)(input)
    absfft1 = Lambda(fftshift)(absfft1)
    absfft2 = Conv2D(channel, kernel_size=3, activation='relu', padding='same')(absfft1)
    W = Lambda(global_average_pooling)(absfft2)
    W = Conv2D(channel // reduction, kernel_size=1, activation='relu', padding='same')(W)
    W = Conv2D(channel, kernel_size=1, activation='sigmoid', padding='same')(W)
    mul = multiply([input, W])
    return mul


def FCAB(input, channel):
    conv = Conv2D(channel, kernel_size=3, padding='same')(input)
    conv = Lambda(gelu)(conv)
    conv = Conv2D(channel, kernel_size=3, padding='same')(conv)
    conv = Lambda(gelu)(conv)
    att = FCALayer(conv, channel, reduction=16)
    output = add([att, input])
    return output


def ResidualGroup(input, channel):
    conv = input
    n_RCAB = 4
    for _ in range(n_RCAB):
        conv = FCAB(conv, channel)
    conv = add([conv, input])
    return conv


def DFCAN(input_shape, scale=2):
    inputs = Input(input_shape)
    conv = Conv2D(64, kernel_size=3, padding='same')(inputs)
    conv = Lambda(gelu)(conv)
    n_ResGroup = 4
    for _ in range(n_ResGroup):
        conv = ResidualGroup(conv, channel=64)
    conv = Conv2D(64 * (scale ** 2), kernel_size=3, padding='same')(conv)
    conv = Lambda(gelu)(conv)
    upsampled = Lambda(pixelshuffle, arguments={'scale': scale})(conv)
    conv = Conv2D(1, kernel_size=3, padding='same')(upsampled)
    output = Activation('sigmoid')(conv)
    model = Model(inputs=inputs, outputs=output)
    return model
