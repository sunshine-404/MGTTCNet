# -*- coding: utf-8 -*-
"""
@Author ：HP
@Time ： 2022年10月06日
"""
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, AveragePooling2D
from tensorflow.keras.layers import Conv1D, Conv2D, SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import Add, Concatenate, BatchNormalization, Lambda, Input, Permute
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import Conv1D, Dropout, MultiHeadAttention, LayerNormalization

def Inception(input_layer,Chans, drop_rate,activation):
    # ========================================================================================
    # EEG-Inception 128 Hz->64,32,16  #250Hz->250,125,62,31
    # ========================================================================================
    block1 = Conv2D(4, (1, 250), padding='same')(input_layer)
    block1 = BatchNormalization()(block1)

    block1 = DepthwiseConv2D((Chans, 1), padding='valid', depth_multiplier=2, depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation(activation)(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = Dropout(drop_rate)(block1)

    # ================================
    block2 = Conv2D(4, (1, 125), padding='same')(input_layer)
    block2 = BatchNormalization()(block2)

    block2 = DepthwiseConv2D((Chans, 1), padding='valid', depth_multiplier=2, depthwise_constraint=max_norm(1.))(block2)
    block2 = BatchNormalization()(block2)
    block2 = Activation(activation)(block2)
    block2 = AveragePooling2D((1, 4))(block2)
    block2 = Dropout(drop_rate)(block2)

    # ================================
    block3 = Conv2D(4, (1, 62), padding='same')(input_layer)
    block3 = BatchNormalization()(block3)

    block3 = DepthwiseConv2D((Chans, 1), padding='valid', depth_multiplier=2, depthwise_constraint=max_norm(1.))(block3)
    block3 = BatchNormalization()(block3)
    block3 = Activation(activation)(block3)
    block3 = AveragePooling2D((1, 4))(block3)
    block3 = Dropout(drop_rate)(block3)

    # 拼接================================
    block = Concatenate(axis=-1)([block1, block2, block3])
    block = AveragePooling2D((1, 4))(block)
    #block = Dropout(drop_rate)(block)

    return block

def TCN_block(input_layer, input_dimension, depth, kernel_size, filters, dropout, activation):
    """ TCN_block from Bai et al 2018 Temporal Convolutional Network (TCN)
        Notes
        -----
        THe original code available at https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
        This implementation has a slight modification from the original code
        and it is taken from the code by Ingolfsson et al at https://github.com/iis-eth-zurich/eeg-tcnet
        See details at https://arxiv.org/abs/2006.00622
        References
        ----------
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

    for i in range(depth - 1): #depth=2,i=0,此时只有一次
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

def ITransTCNet_noTrans(n_classes, in_chans, in_samples):

    input_1 = Input(shape=(in_chans,in_samples,1))

    #Inception模块
    block1 = Inception(input_layer=input_1, Chans=16, drop_rate=0.5,activation='elu')
    block1 = Lambda(lambda x: x[:, -1, :, :])(block1)

    # Temporal convolutional network (TCN)
    block3 = TCN_block(input_layer=block1, input_dimension=24, depth=2, kernel_size=4, filters=32,dropout=0.3, activation='elu')

    # Get feature maps of the last sequence
    block3 = Lambda(lambda x: x[:, -1, :])(block3)

    softmax = Dense(n_classes, activation="softmax")(block3)

    return Model(inputs=input_1, outputs=softmax)

model=ITransTCNet_noTrans(2,16,1000)
print(model.summary())