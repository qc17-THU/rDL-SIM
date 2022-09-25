from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, add, Lambda, multiply
from tensorflow.keras.layers import UpSampling2D, AveragePooling2D, Conv2D
from tensorflow.keras.layers import LeakyReLU


def CALayer(input, input_height, input_width, channel, reduction=16):
    W = AveragePooling2D(pool_size=(input_height, input_width))(input)
    W = Conv2D(channel // reduction, kernel_size=1, activation='relu', padding='same')(W)
    W = Conv2D(channel, kernel_size=1, activation='sigmoid', padding='same')(W)
    W = UpSampling2D(size=(input_height, input_width))(W)
    mul = multiply([input, W])
    return mul


def RCAB(input, input_height, input_width, channel):
    conv = Conv2D(channel, kernel_size=3, padding='same')(input)
    conv = LeakyReLU(alpha=0.2)(conv)
    conv = Conv2D(channel, kernel_size=3, padding='same')(conv)
    conv = LeakyReLU(alpha=0.2)(conv)
    att = CALayer(conv, input_height, input_width, channel, reduction=16)
    output = add([att, input])
    return output


def ResidualGroup(input, input_height, input_width, channel):
    conv = input
    n_RCAB = 5
    for _ in range(n_RCAB):
        conv = RCAB(conv, input_height, input_width, channel)
    output = add([conv, input])
    return output


def Denoiser(input_shape, n_rg=(2, 5, 5)):

    inputs1 = Input(input_shape)
    inputs2 = Input(input_shape)
    # --------------------------------------------------------------------------------
    #                      extract features of generated image
    # --------------------------------------------------------------------------------
    conv0 = Conv2D(64, kernel_size=3, padding='same')(inputs1)
    conv = LeakyReLU(alpha=0.2)(conv0)
    for _ in range(n_rg[0]):
        conv = ResidualGroup(conv, input_shape[0], input_shape[1], 64)
    conv = add([conv, conv0])
    conv = Conv2D(64, kernel_size=3, padding='same')(conv)
    conv1 = LeakyReLU(alpha=0.2)(conv)

    # --------------------------------------------------------------------------------
    #                      extract features of noisy image
    # --------------------------------------------------------------------------------
    conv0 = Conv2D(64, kernel_size=3, padding='same')(inputs2)
    conv = LeakyReLU(alpha=0.2)(conv0)
    for _ in range(n_rg[1]):
        conv = ResidualGroup(conv, input_shape[0], input_shape[1], 64)
    conv = add([conv, conv0])
    conv = Conv2D(64, kernel_size=3, padding='same')(conv)
    conv2 = LeakyReLU(alpha=0.2)(conv)

    # --------------------------------------------------------------------------------
    #                              merge features
    # --------------------------------------------------------------------------------
    weight1 = Lambda(lambda x: x*1)
    weight2 = Lambda(lambda x: x*1)
    conv1 = weight1(conv1)
    conv2 = weight2(conv2)

    conct = add([conv1, conv2])
    conct = Conv2D(64, kernel_size=3, padding='same')(conct)
    conct = LeakyReLU(alpha=0.2)(conct)
    conv = conct

    for _ in range(n_rg[2]):
        conv = ResidualGroup(conv, input_shape[0], input_shape[1], 64)
    conv = add([conv, conct])

    conv = Conv2D(256, kernel_size=3, padding='same')(conv)
    conv = LeakyReLU(alpha=0.2)(conv)

    CA = CALayer(conv, input_shape[0], input_shape[1], 256, reduction=16)
    conv = Conv2D(input_shape[2], kernel_size=3, padding='same')(CA)

    output = LeakyReLU(alpha=0.2)(conv)

    model = Model(inputs=[inputs1, inputs2], outputs=output)
    return model
