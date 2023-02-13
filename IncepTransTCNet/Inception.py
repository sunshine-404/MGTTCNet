# -*- coding: utf-8 -*-
"""
@Author ：HP
@Time ： 2022年09月16日
"""
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, AveragePooling2D
from tensorflow.keras.layers import Conv1D, Conv2D, SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import Add, Concatenate, BatchNormalization, Lambda, Input, Permute
from tensorflow.keras.constraints import max_norm

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
    block12 = Concatenate(axis=-1)([block1, block2])
    block13 = Concatenate(axis=-1)([block1, block3])
    block23 = Concatenate(axis=-1)([block2, block3])
    block = Concatenate(axis=-1)([block12, block13, block23])

    #block = Concatenate(axis=-1)([block1, block2, block3])
    block = AveragePooling2D((1, 8))(block)
    block = Dropout(drop_rate)(block)

    return block

def InceptionNet(n_classes, in_chans, in_samples):
    input_1 = Input(shape=(in_chans,in_samples,1))

    #Inception模块
    block1 = Inception(input_layer=input_1, Chans=16, drop_rate=0.5, activation='elu')
    embedded = Flatten()(block1)
    out = Dense(n_classes, activation='softmax')(embedded)

    return Model(inputs=input_1, outputs=out)

model=InceptionNet(2,16,2000)
print(model.summary())