# -*- coding: utf-8 -*-
"""
@Author ：HP
@Time ： 2022年08月28日
论文：EEG-TCNet: An Accurate Temporal Convolutional Network for Embedded Motor-Imagery Brain–Machine Interfaces
TF:https://github.com/iis-eth-zurich/eeg-tcnet
此为EEGNetTCNet(Fixed)
"""
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, AveragePooling2D
from tensorflow.keras.layers import Conv1D, Conv2D, SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization, LayerNormalization, Flatten
from tensorflow.keras.layers import Add, Concatenate, Lambda, Input, Permute
from tensorflow.keras.constraints import max_norm

# %% Temporal convolutional (TC) block used in the ATCNet model
def TCN_block(input_layer, input_dimension, depth, kernel_size, filters, dropout=0.3, activation='elu'):
    """ TCN_block from Bai et al 2018 Temporal Convolutional Network (TCN)

        Notes
        THe original code available at https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
        This implementation has a slight modification from the original code
        and it is taken from the code by Ingolfsson et al at https://github.com/iis-eth-zurich/eeg-tcnet
        See details at https://arxiv.org/abs/2006.00622

        References
        .. Bai, S., Kolter, J. Z., & Koltun, V. (2018).
           An empirical evaluation of generic convolutional and recurrent networks for sequence modeling.
           arXiv preprint arXiv:1803.01271.
    """

    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',padding='causal', kernel_initializer='he_uniform')(input_layer)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)

    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',padding='causal', kernel_initializer='he_uniform')(block)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)

    if (input_dimension != filters):
        conv = Conv1D(filters, kernel_size=1, padding='same')(input_layer)
        added = Add()([block, conv])
    else:
        added = Add()([block, input_layer])
    out = Activation(activation)(added)

    for i in range(depth - 1):
        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2 ** (i + 1), activation='linear',padding='causal', kernel_initializer='he_uniform')(out)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)

        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2 ** (i + 1), activation='linear',padding='causal', kernel_initializer='he_uniform')(block)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        added = Add()([block, out])
        out = Activation(activation)(added)

    return out

#此数据集的Hz=250，不必考虑 kernLength
def EEGNet(input_layer, F1, kernLength, D, Chans, dropout=0.2, activation='elu'):
    """ EEGNet model from Lawhern et al 2018 See details at https://arxiv.org/abs/1611.08024
    The original code for this model is available at: https://github.com/vlawhern/arl-eegmodels
        Notes The initial values in this model are based on the values identified by the authors
        References
           EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces. arXiv preprint arXiv:1611.08024.
    """
    F2 = F1 * D
    block1 = Conv2D(F1, (kernLength, 1), padding='same', use_bias=False)(input_layer)
    block1 = BatchNormalization(axis=-1)(block1)

    block2 = DepthwiseConv2D((1, Chans), use_bias=False,depthwise_constraint=max_norm(1.))(block1)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation(activation)(block2)

    block2 = AveragePooling2D((8, 1), data_format='channels_last')(block2)
    block2 = Dropout(dropout)(block2)

    block3 = SeparableConv2D(F2, (16, 1), use_bias=False, padding='same')(block2)
    block3 = BatchNormalization(axis=-1)(block3)
    block3 = Activation(activation)(block3)

    block3 = AveragePooling2D((8, 1), data_format='channels_last')(block3)
    block3 = Dropout(dropout)(block3)

    return block3

#Reproduced EEGTCNet model: https://arxiv.org/abs/2006.00622 论文中参数
def EEGTCNet(n_classes, Chans, Samples, F1=8, D=2):
    """ EEGTCNet model from Ingolfsson et al 2020.See details at https://arxiv.org/abs/2006.00622
    The original code for this model is available at https://github.com/iis-eth-zurich/eeg-tcnet
        Notes
           Eeg-tcnet: An accurate temporal convolutional network for embedded motor-imagery brain–machine interfaces.
           In 2020 IEEE International Conference on Systems,
    """
    input1 = Input(shape=(Chans, Samples, 1))
    input2 = Permute((2, 1, 3))(input1)

    numFilters = F1
    F2 = numFilters * D

    EEGNet_sep = EEGNet(input_layer=input2, F1=F1, kernLength=32, D=D, Chans=Chans)
    block2 = Lambda(lambda x: x[:, :, -1, :])(EEGNet_sep)

    outs = TCN_block(input_layer=block2, input_dimension=F2, depth=2, kernel_size=4, filters=12)
    out = Lambda(lambda x: x[:, -1, :])(outs)

    dense = Dense(n_classes, name='dense', kernel_constraint=max_norm(0.25))(out)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)

model=EEGTCNet(2,64,128)
print(model.summary())