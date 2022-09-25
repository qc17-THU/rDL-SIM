from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, add, multiply, UpSampling3D, Conv3D, LeakyReLU, Lambda
import tensorflow as tf


def GlobalAveragePooling(input):
    return tf.reduce_mean(input, axis=(1, 2, 3), keepdims=True)


def CALayer(input, channel, reduction=16):
    W = Lambda(GlobalAveragePooling)(input)
    W = Conv3D(channel // reduction, kernel_size=1, activation='relu', padding='same')(W)
    W = Conv3D(channel, kernel_size=1, activation='sigmoid', padding='same')(W)
    mul = multiply([input, W])
    return mul


def RCAB(input, channel):
    conv = Conv3D(channel, kernel_size=3, padding='same')(input)
    conv = LeakyReLU(alpha=0.2)(conv)
    conv = Conv3D(channel, kernel_size=3, padding='same')(conv)
    conv = LeakyReLU(alpha=0.2)(conv)
    att = CALayer(conv, channel, reduction=16)
    output = add([att, input])
    return output


def ResidualGroup(input, channel, n_RCAB=5):
    conv = input
    for _ in range(n_RCAB):
        conv = RCAB(conv, channel)
    return conv


def RCAN3D(input_shape, n_ResGroup=4, n_RCAB=5):

    inputs = Input(input_shape)
    conv = Conv3D(64, kernel_size=3, padding='same')(inputs)
    for _ in range(n_ResGroup):
        conv = ResidualGroup(conv, 64, n_RCAB=n_RCAB)

    conv = Conv3D(256, kernel_size=3, padding='same')(conv)
    conv = LeakyReLU(alpha=0.2)(conv)
    conv = Conv3D(input_shape[3], kernel_size=3, padding='same')(conv)
    output = LeakyReLU(alpha=0.2)(conv)

    model = Model(inputs=inputs, outputs=output)

    return model

