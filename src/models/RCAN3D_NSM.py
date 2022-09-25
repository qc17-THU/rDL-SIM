from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, add, multiply, UpSampling3D, Conv3D, LeakyReLU, Lambda, Layer
from tensorflow.keras.initializers import constant as const
import tensorflow as tf


class NoiseSuppressionModule(Layer):

    def __init__(self, init_cutoff_freq=4.1, dxy=0.0926, init_slop=100):
        super(NoiseSuppressionModule, self).__init__()
        self.cutoff_freq = self.add_weight(shape=(1,), initializer=const(init_cutoff_freq),
                                           trainable=True, name='cutoff_freq')
        self.slop = self.add_weight(shape=(1,), initializer=const(init_slop),
                                    trainable=True, name='slop')
        self.dxy = tf.Variable(initial_value=dxy, trainable=False, name='dxy')

    def call(self, inputs):
        bs, ny, nx, nz, ch = inputs.get_shape().as_list()
        ny = tf.cast(ny, tf.float32)
        nx = tf.cast(nx, tf.float32)
        dkx = tf.divide(1, tf.multiply(nx, self.dxy))
        dky = tf.divide(1, tf.multiply(ny, self.dxy))

        y = tf.multiply(tf.cast(tf.range(-ny // 2, ny // 2), tf.float32), dky)
        x = tf.multiply(tf.cast(tf.range(-nx // 2, nx // 2), tf.float32), dkx)
        [X, Y] = tf.meshgrid(x, y)
        rdist = tf.sqrt(tf.square(X) + tf.square(Y))

        otf_mask = tf.sigmoid(tf.multiply(self.cutoff_freq - rdist, self.slop))
        otf = otf_mask
        otf_mask = tf.expand_dims(tf.expand_dims(tf.expand_dims(otf_mask, 0), 0), 0)
        otf_mask = tf.tile(otf_mask, (1, nz, ch, 1, 1))
        otf_mask = tf.complex(otf_mask, tf.zeros_like(otf_mask))

        inputs = tf.complex(inputs, tf.zeros_like(inputs))
        inputs = tf.transpose(inputs, [0, 3, 4, 1, 2])
        fft_feature = tf.signal.fftshift(tf.signal.fft2d(inputs))
        output = tf.signal.ifft2d(tf.signal.fftshift(tf.multiply(otf_mask, fft_feature)))
        output = tf.transpose(output, [0, 3, 4, 1, 2])
        output = tf.math.real(output)

        return output


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
    conv_input = Conv3D(64, kernel_size=3, padding='same')(inputs)
    NSM = NoiseSuppressionModule()
    inputs_ns = NSM(inputs)
    conv = Conv3D(64, kernel_size=3, padding='same')(inputs_ns)
    conv = conv + conv_input
    for _ in range(n_ResGroup):
        conv = ResidualGroup(conv, 64, n_RCAB=n_RCAB)

    conv = Conv3D(256, kernel_size=3, padding='same')(conv)
    conv = LeakyReLU(alpha=0.2)(conv)
    conv = Conv3D(input_shape[3], kernel_size=3, padding='same')(conv)
    output = LeakyReLU(alpha=0.2)(conv)

    model = Model(inputs=inputs, outputs=output)

    return model

