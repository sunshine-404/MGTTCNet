# -*- coding: utf-8 -*-
"""
@Author ：HP
@Time ： 2022年09月03日
"""
from tensorflow.keras.models import Model
from keras.layers import add,Flatten,Dense, Multiply, Activation,Concatenate
from keras.layers import Conv1D, Conv2D, AveragePooling2D, MaxPooling2D,SeparableConv2D,GlobalAveragePooling1D
from keras.layers import Dropout,BatchNormalization, Add, Lambda, DepthwiseConv2D, Input, Permute
from keras.constraints import max_norm
from keras import backend as K
K.set_image_data_format('channels_last')

import warnings
warnings.filterwarnings('ignore')

#原论文中核大小为采样率的一半
def EEGNet4_8(nb_classes, Chans, Samples, F1=4, F2=8, D=2, dropoutRate=0.5):
    input_main = Input((Chans, Samples, 1))

    block1 = Conv2D(F1, (1, 125), padding='same', use_bias=False)(input_main)
    block1 = BatchNormalization()(block1)

    block1 = DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=D, depthwise_constraint=max_norm(1.))(block1)
    # 使用depth_multiplier选项可以准确地映射到在一个时间卷积中学习的空间滤波器的数量。2*8=16
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = Dropout(dropoutRate)(block1)

    block1 = SeparableConv2D(F2, (1, 16), use_bias=False, padding='same')(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 8))(block1)
    block1 = Dropout(dropoutRate)(block1)

    flatten = Flatten(name='flatten')(block1)
    dense = Dense(nb_classes, name='dense', kernel_constraint=max_norm(0.25))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)

model_EEGNet4_8 = EEGNet4_8(nb_classes=2, Chans=22, Samples=1125, dropoutRate=0.5)
#print(model_EEGNet4_8.summary())

def EEGNet8_16(nb_classes, Chans, Samples, F1=8, F2 = 16, D=2, dropoutRate=0.5):
    input_main = Input((Chans, Samples, 1))
    block1 = Conv2D(F1, (1, 125), padding='same', use_bias=False)(input_main)
    block1 = BatchNormalization()(block1)

    block1 = DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=D,depthwise_constraint = max_norm(1.))(block1)
    # 使用depth_multiplier选项可以准确地映射到在一个时间卷积中学习的空间滤波器的数量。2*8=16
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = Dropout(dropoutRate)(block1)

    block1 = SeparableConv2D(F2, (1, 16), use_bias=False, padding='same')(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 8))(block1)
    block1 = Dropout(dropoutRate)(block1)

    flatten = Flatten(name='flatten')(block1)
    dense = Dense(nb_classes, name='dense',kernel_constraint=max_norm(0.25))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)

model_EEGNet8_16=EEGNet8_16(nb_classes=2,Chans=22, Samples=1125)
print(model_EEGNet8_16.summary())

def ShallowConvNet(nb_classes, Chans, Samples, dropoutRate=0.5):
    # start the model
    input_main = Input((Chans, Samples, 1))
    block1 = Conv2D(40, (1, 25), input_shape=(Chans, Samples, 1), padding='same',kernel_constraint = max_norm(2., axis=(0,1,2)))(input_main)
    block1 = Conv2D(40, (Chans, 1), use_bias=False,kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
    block1 = Activation(square)(block1)

    block1 = AveragePooling2D(pool_size=(1, 75), strides=(1, 15))(block1)
    block1 = Activation(log)(block1)
    block1 = Dropout(dropoutRate)(block1)

    flatten = Flatten()(block1)
    dense = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)
    return Model(inputs=input_main, outputs=softmax)

# need these for ShallowConvNet
def square(x):
    return K.square(x)

def log(x):
    return K.log(K.clip(x, min_value=1e-7, max_value=10000))

# model_ShallowConvNet = ShallowConvNet(nb_classes=2, Chans=16, Samples=1000)
# print(model_ShallowConvNet.summary())

def DeepConvNet(nb_classes, Chans, Samples, dropoutRate=0.5):
    # start the model
    input_main = Input((Chans, Samples, 1))
    block1 = Conv2D(25, (1, 10),input_shape=(Chans, Samples, 1),kernel_constraint = max_norm(2., axis=(0,1,2)))(input_main)
    block1 = Conv2D(25, (Chans, 1),kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = Conv2D(50, (1, 10),kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block2)
    block2 = Dropout(dropoutRate)(block2)

    block3 = Conv2D(100, (1, 10),kernel_constraint = max_norm(2., axis=(0,1,2)))(block2)
    block3 = BatchNormalization()(block3)
    block3 = Activation('elu')(block3)
    block3 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block3)
    block3 = Dropout(dropoutRate)(block3)

    block4 = Conv2D(200, (1, 10),kernel_constraint = max_norm(2., axis=(0,1,2)))(block3)
    block4 = BatchNormalization()(block4)
    block4 = Activation('elu')(block4)
    block4 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block4)
    block4 = Dropout(dropoutRate)(block4)

    flatten = Flatten()(block4)

    dense = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)

model_DeepConvNet=DeepConvNet(nb_classes =2, Chans = 22, Samples = 1125)
#print(model_DeepConvNet.summary())