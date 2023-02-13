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
    block = AveragePooling2D((1, 8))(block)
    #block = Dropout(drop_rate)(block)

    return block

def transformer_encoder(inputs, head_size, num_heads, ff_dim=4, dropout=0.5):
    #ff_dim: Hidden layer size in feed forward network inside transformer

    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="elu")(res)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = LayerNormalization(epsilon=1e-6)(x)

    return x + res

def ITransTCNet_noTCN(n_classes, in_chans, in_samples,
                  num_transformer_blocks,head_size, num_heads):

    input_1 = Input(shape=(in_chans,in_samples,1))

    #Inception模块
    block1 = Inception(input_layer=input_1, Chans=16, drop_rate=0.5,activation='elu') #gelu
    block1 = Lambda(lambda x: x[:, -1, :, :])(block1)

    #Transformer_encoder
    for _ in range(num_transformer_blocks):
        block2 = transformer_encoder(block1, head_size, num_heads)

    # Get feature maps of the last sequence
    block3 = Lambda(lambda x: x[:, -1, :])(block2)

    softmax = Dense(n_classes, activation="softmax")(block3)

    return Model(inputs=input_1, outputs=softmax)

model=ITransTCNet_noTCN(2,16,1000,
                num_transformer_blocks=2,head_size=16, num_heads=4)
print(model.summary())