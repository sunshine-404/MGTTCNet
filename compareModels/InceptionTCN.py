# -*- coding: utf-8 -*-
"""
@Author ：HP
@Time ： 2022年08月27日
EEG-ITNet: An Explainable Inception Temporal Convolutional Network for Motor Imagery Classification
Tensorflow code  https://github.com/AbbasSalami/EEG-ITNet
"""
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, SpatialDropout1D, SpatialDropout2D, BatchNormalization
from tensorflow.keras.layers import Flatten, InputSpec, Layer, Concatenate, AveragePooling2D, MaxPooling2D, Reshape, Permute
from tensorflow.keras.layers import Conv2D, Add, LSTM , SeparableConv2D, DepthwiseConv2D, ConvLSTM2D, LayerNormalization
from tensorflow.keras.constraints import max_norm, unit_norm

import warnings
warnings.filterwarnings('ignore')

n_ff = [2, 4, 8]

def EEG_ITNet(nb_classes,Chans, Samples, drop_rate=0.5):
    Input_block = Input(shape=(Chans, Samples, 1))

    #原文中128 Hz->64,32,16  #250Hz->125,62,31
    block1 = Conv2D(n_ff[0], (1, 31), use_bias=False, activation='linear', padding='same', name='Spectral_filter_1')(Input_block)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False, padding='valid', depth_multiplier=1, activation='linear',depthwise_constraint=max_norm(max_value=1), name='Spatial_filter_1')(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)

    # ================================
    block2 = Conv2D(n_ff[1], (1, 62), use_bias=False, activation='linear', padding='same', name='Spectral_filter_2')(Input_block)
    block2 = BatchNormalization()(block2)
    block2 = DepthwiseConv2D((Chans, 1), use_bias=False, padding='valid', depth_multiplier=1, activation='linear',depthwise_constraint=max_norm(max_value=1), name='Spatial_filter_2')(block2)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)

    # ================================
    block3 = Conv2D(n_ff[2], (1, 125), use_bias=False, activation='linear', padding='same', name='Spectral_filter_3')(Input_block)
    block3 = BatchNormalization()(block3)
    block3 = DepthwiseConv2D((Chans, 1), use_bias=False, padding='valid', depth_multiplier=1, activation='linear',depthwise_constraint=max_norm(max_value=1), name='Spatial_filter_3')(block3)
    block3 = BatchNormalization()(block3)
    block3 = Activation('elu')(block3)

    # 合并================================
    block = Concatenate(axis=-1)([block1, block2, block3])
    block = AveragePooling2D((1, 4))(block)
    block_in = Dropout(drop_rate)(block)  # (1,125,14)

    # 1)*****************************************************************
    paddings = tf.constant([[0, 0], [0, 0], [3, 0], [0, 0]])  # tf.constant()用来定义tensor常量
    # tf.pad
    # padding它必须是 [N, 2] 形式，N代表张量的阶， 2代表必须是2列
    # padding里面每个[a, b]  都代表在相应的维上，前加 a个(行) 0，后加b个(行) 0
    # padding=[[3,1],[3,5]] 第一个维度上前加3行0，后加1行0；第二维度上，前加3个0，后加5个0

    block = tf.pad(block_in, paddings, "CONSTANT")  # (1,125,14)->(1,128,14)
    block = DepthwiseConv2D((1, 4), padding="valid", depth_multiplier=1, dilation_rate=(1, 1))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)

    block = tf.pad(block, paddings, "CONSTANT")  # (1,125,14)->(1,128,14)
    block = DepthwiseConv2D((1, 4), padding="valid", depth_multiplier=1, dilation_rate=(1, 1))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)

    block_out = Add()([block_in, block])  # (1,125,14)

    # 2)*****************************************************************
    paddings = tf.constant([[0, 0], [0, 0], [6, 0], [0, 0]])

    block = tf.pad(block_out, paddings, "CONSTANT")  # (1,125,14)->(1,131,14)
    block = DepthwiseConv2D((1, 4), padding="valid", depth_multiplier=1, dilation_rate=(1, 2))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)

    block = tf.pad(block, paddings, "CONSTANT")  # (1,125,14)->(1,131,14)
    block = DepthwiseConv2D((1, 4), padding="valid", depth_multiplier=1, dilation_rate=(1, 2))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)

    block_out = Add()([block_out, block])  # (1,125,14)

    # 3)*****************************************************************
    paddings = tf.constant([[0, 0], [0, 0], [12, 0], [0, 0]])

    block = tf.pad(block_out, paddings, "CONSTANT")  # (1,125,14)->(1,137,14)
    block = DepthwiseConv2D((1, 4), padding="valid", depth_multiplier=1, dilation_rate=(1, 4))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)

    block = tf.pad(block, paddings, "CONSTANT")  # (1,125,14)->(1,137,14)
    block = DepthwiseConv2D((1, 4), padding="valid", depth_multiplier=1, dilation_rate=(1, 4))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)

    block_out = Add()([block_out, block])  # (1,125,14)

    # 4)*****************************************************************
    paddings = tf.constant([[0, 0], [0, 0], [24, 0], [0, 0]])

    block = tf.pad(block_out, paddings, "CONSTANT")  # (1,125,14)->(1,149,14)
    block = DepthwiseConv2D((1, 4), padding="valid", depth_multiplier=1, dilation_rate=(1, 8))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)

    block = tf.pad(block, paddings, "CONSTANT")  # (1,125,14)->(1,149,14)
    block = DepthwiseConv2D((1, 4), padding="valid", depth_multiplier=1, dilation_rate=(1, 8))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)

    block_out = Add()([block_out, block])  # (1,125,14)

    # ================================
    block = block_out
    # ================================

    block = Conv2D(28, (1, 1))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = AveragePooling2D((1, 4), data_format='channels_last')(block)
    block = Dropout(drop_rate)(block)
    embedded = Flatten()(block)

    out = Dense(nb_classes, activation='softmax', kernel_constraint=max_norm(0.25))(embedded)
    return Model(inputs=Input_block, outputs=out)

model = EEG_ITNet(2,16,1000)
print(model.summary())