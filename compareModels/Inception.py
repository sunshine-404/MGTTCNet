# -*- coding: utf-8 -*-
"""
@Author ：HP
@Time ： 2022年08月27日
EEG-Inception: A Novel Deep Convolutional Neural Network for Assistive ERP-based Brain-Computer Interfaces
无code，InceptionTCN作者复现
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9311146
"""
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, SpatialDropout2D, BatchNormalization
from tensorflow.keras.layers import Conv2D, Flatten, DepthwiseConv2D,Concatenate, AveragePooling2D, MaxPooling2D

import warnings
warnings.filterwarnings('ignore')

def EEG_Incep(nb_classes,Chans, Samples, drop_rate=0.5):
    Input_block = Input(shape=(Chans, Samples, 1))
    # ========================================================================================
    # EEG-Inception
    # ========================================================================================
    #原文中128 Hz->64,32,16
    block1 = Conv2D(8, (1, 125), padding='same')(Input_block)  #250Hz->125,62,31,15,7
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = Dropout(drop_rate)(block1)
    block1 = DepthwiseConv2D((Chans, 1), padding='valid', depth_multiplier=2)(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = Dropout(drop_rate)(block1)

    # ================================
    block2 = Conv2D(8, (1, 62), padding='same')(Input_block)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = Dropout(drop_rate)(block2)
    block2 = DepthwiseConv2D((Chans, 1), padding='valid', depth_multiplier=2)(block2)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = Dropout(drop_rate)(block2)

    # ================================
    block3 = Conv2D(8, (1, 31), padding='same')(Input_block)
    block3 = BatchNormalization()(block3)
    block3 = Activation('elu')(block3)
    block3 = Dropout(drop_rate)(block3)
    block3 = DepthwiseConv2D((Chans, 1), padding='valid', depth_multiplier=2)(block3)
    block3 = BatchNormalization()(block3)
    block3 = Activation('elu')(block3)
    block3 = Dropout(drop_rate)(block3)

    # 拼接================================
    block = Concatenate(axis=-1)([block1, block2, block3])
    block = AveragePooling2D((1, 4))(block)

    # ================================
    block1_1 = Conv2D(8, (1, 31), padding='same')(block)
    block1_1 = BatchNormalization()(block1_1)
    block1_1 = Activation('elu')(block1_1)
    block1_1 = Dropout(drop_rate)(block1_1)

    # ================================
    block2_1 = Conv2D(8, (1, 15), padding='same')(block)
    block2_1 = BatchNormalization()(block2_1)
    block2_1 = Activation('elu')(block2_1)
    block2_1 = Dropout(drop_rate)(block2_1)

    # ================================
    block3_1 = Conv2D(8, (1, 7), padding='same')(block)
    block3_1 = BatchNormalization()(block3_1)
    block3_1 = Activation('elu')(block3_1)
    block3_1 = Dropout(drop_rate)(block3_1)

    # 拼接================================
    block_new = Concatenate(axis=-1)([block1_1, block2_1, block3_1])
    block_new = AveragePooling2D((1, 2))(block_new)

    # ================================
    block_new = Conv2D(12, (1, 8), padding='same')(block_new)
    block_new = BatchNormalization()(block_new)
    block_new = Activation('elu')(block_new)
    block_new = Dropout(drop_rate)(block_new)
    block_new = AveragePooling2D((1, 2))(block_new)

    block_new = Conv2D(6, (1, 4), padding='same')(block_new)
    block_new = BatchNormalization()(block_new)
    block_new = Activation('elu')(block_new)
    block_new = Dropout(drop_rate)(block_new)
    block_new = AveragePooling2D((1, 2))(block_new)

    embedded = Flatten()(block_new)
    out = Dense(nb_classes, activation='softmax')(embedded)

    return Model(inputs=Input_block, outputs=out)

model = EEG_Incep(2, 16, 1000)
print(model.summary())