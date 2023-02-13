# -*- coding: utf-8 -*-
"""
@Author ：HP
@Time ： 2022年09月04日
"""
from IncepTransTCNet.ITransTCNet import ITransTCNet
from IncepTransTCNet.ITransTCNet_noTCN import ITransTCNet_noTCN
from IncepTransTCNet.ITransTCNet_noTrans import ITransTCNet_noTrans
from IncepTransTCNet.Incep_TCN_Trans import ITCNTrans
from IncepTransTCNet.saveML import ITransTCNet_L
from alation.IATCNet_noAtten import IATCNet_noAtten
from alation.IATCNet_noSW import IATCNet_noSW
from alation.IATCNet_noTCN import IATCNet_noTCN
from alation.Inception import InceptionNet
from atten.IncepAtten import IncepAttenNet
from atten.InceptionAtten import InceptionAtten
from compareModels.CNN_LSTM import CNNLSTM
from compareModels.baseModel import EEGNet4_8, EEGNet8_16, ShallowConvNet, DeepConvNet
from compareModels.ATCNet import ATCNet
from compareModels.ChroNet import ChroNet
from compareModels.EEGNeX_8_32 import EEGNeX_8_32
from compareModels.EEGNetTCNet import EEGTCNet
from compareModels.EEGNetTCNet_Fusion import TCNet_Fusion
from compareModels.IATCNet import IATCNet
from compareModels.Inception import EEG_Incep
from compareModels.InceptionTCN import EEG_ITNet
from paperModel.EEGNetInception import EEGNetInception
from picture.draw_conf import draw_confusion
from picture.plot_acc import plot_accuracy, plot_loss
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import warnings

from picture.plot_conf import draw_confusion_matrix
from tranformer.Inception_Transf_encode import ITransformerNet

warnings.filterwarnings('ignore')

# plt.rcParams两行是用于解决标签不能显示汉字的问题
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import os
import time
import numpy as np

# from tensorflow.keras.optimizers import Adam
# import tensorflow as tf
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# from tensorflow.keras import backend as K

from tensorflow.keras.optimizers import Adam
#from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import cohen_kappa_score, classification_report, ConfusionMatrixDisplay
# kappa score是一个介于(-1, 1)之间的数. score>0.8意味着好的分类；0或更低意味着不好（实际是随机标签）

def getData():
    # 加载数据
    # data_array = np.load(file="/root/autodl-tmp/dataset_GKF/data8s.npy")
    # label_array = np.load(file="/root/autodl-tmp/dataset_GKF/label8s.npy")

    # 没有重叠的4s数据
    #data_array = np.load(file="/root/autodl-tmp/dataset01/data4s.npy")
    #label_array = np.load(file="/root/autodl-tmp/dataset01/label4s.npy")

    #有重叠的4s数据 (14781, 16, 1000) (14781, 2) (14781,)
    data_array = np.load(file="/root/autodl-tmp/dataset4s_GKF/data4s.npy")
    label_array = np.load(file="/root/autodl-tmp/dataset4s_GKF/label4s.npy")

    # 分别加载5种不同频带数据16

    #(10374, 16, 1000) (3012, 16, 1000) (1488, 16, 1000)
    # # data_array = np.load(file="/root/autodl-tmp/band_data/delta4s.npy")
    # # label_array = np.load(file="/root/autodl-tmp/band_data/delta_label4s.npy")
    #
    # #(10290, 16, 1000) (2988, 16, 1000) (1476, 16, 1000)
    # # data_array = np.load(file="/root/autodl-tmp/band_data/theta4s.npy")
    # # label_array = np.load(file="/root/autodl-tmp/band_data/theta_label4s.npy")
    #
    # #(10385, 16, 1000) (3015, 16, 1000) (1489, 16, 1000)
    # # data_array = np.load(file="/root/autodl-tmp/band_data/alpha4s.npy")
    # # label_array = np.load(file="/root/autodl-tmp/band_data/alpha_label4s.npy")
    #
    # #(10295, 16, 1000) (2989, 16, 1000) (1476, 16, 1000)
    # # data_array = np.load(file="/root/autodl-tmp/band_data/beta4s.npy")
    # # label_array = np.load(file="/root/autodl-tmp/band_data/beta_label4s.npy")
    #
    # #(10252, 16, 1000) (2977, 16, 1000) (1470, 16, 1000)
    # data_array = np.load(file="/root/autodl-tmp/band_data/gamma4s.npy")
    # label_array = np.load(file="/root/autodl-tmp/band_data/gamma_label4s.npy")

    # 划分训练集、测试集
    # random_state为同一数值时，数据划分结果就一样，random_state为不同数值时，数据划分结果就不一样。
    X_train, X_test, y_train, y_test = train_test_split(data_array, label_array, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.225, random_state=42)

    # 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    #print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)
    print(X_train.shape, X_val.shape, X_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test

def getModel(model_name):
    # 对比的模型
    if (model_name == 'EEGNet4_8'):
        model = EEGNet4_8(2, 16, 1000)
    elif (model_name == 'EEGNet8_16'):
        model = EEGNet8_16(2, 16, 1000)
    elif (model_name == 'ShallowConvNet'):
        model = ShallowConvNet(2, 16, 1000)
    elif (model_name == 'DeepConvNet'):
        model = DeepConvNet(2, 16, 1000)

    elif (model_name == 'ChroNet'):
        model = ChroNet(2, 16, 1000)

    #一篇DP论文中的模型 论文中acc高，但此处acc太低，无法比较
    elif (model_name == 'CNNLSTM'):
        model = CNNLSTM(2, 16, 1000)

    elif (model_name == 'EEGNetTCNet'):
        model = EEGTCNet(2, 16, 1000)
    elif (model_name == 'EEGNetTCNet_Fusion'):
        model = TCNet_Fusion(2, 16, 1000)

    elif (model_name == 'EEGNetX8_32'):
        model = EEGNeX_8_32(2, 16, 1000)

    elif (model_name == 'Inception'):
        model = EEG_Incep(2, 16, 1000)
    elif (model_name == 'InceptionTCN'):
        model = EEG_ITNet(2, 16, 1000)

    #改进的模型
    elif (model_name == 'ATCNet'):
        model = ATCNet(2, 16, 1000)
    elif (model_name == 'IATCNet'):
        model = IATCNet(2, 16, 1000)

    #IATCNet的消融实验 Inception+SW+ATTen+TCN
    elif (model_name == 'InceptionNet'):
        model = InceptionNet(2, 16, 1000)
    elif (model_name == 'IATCNet_noSW'):
        model = IATCNet_noSW(2, 16, 1000)
    elif (model_name == 'IATCNet_noTCN'):
        model = IATCNet_noTCN(2, 16, 1000)
    elif (model_name == 'IATCNet_noAtten'):
        model = IATCNet_noAtten(2, 16, 1000)

    elif (model_name == 'EEGNetInception'):
        model = EEGNetInception(2, 16, 1000)

    # #注意力模型 mha, se, eca, cbam, danet
    # elif (model_name == 'IncepAttenNet'):
    #     model = IncepAttenNet(2, 16, 2000)
    #
    # elif (model_name == 'InceptionAtten'):
    #     model = InceptionAtten(2, 16, 2000)

    # Inception+Transformer+TCN
    elif (model_name == 'ITransTCNet'):
        model = ITransTCNet(2, 16, 1000,
                           num_transformer_blocks=2,head_size=16, num_heads=6)

    # ITransTCNet的消融实验 Inception+Trans+TCN
    elif (model_name == 'InceptionNet'):
        model = InceptionNet(2, 16, 1000)
    elif (model_name == 'ITransTCNet_noTrans'):
        model = ITransTCNet_noTrans(2, 16, 1000)
    elif (model_name == 'IATCNet_noTCN'):
        model = ITransTCNet_noTCN(2, 16, 1000,
                                  num_transformer_blocks=2,head_size=16, num_heads=6)
    elif (model_name == 'ITCNTransNet'):
        model = ITCNTrans(2, 16, 1000,
                                  num_transformer_blocks=2,head_size=16, num_heads=6)
     #尺度动态卷积模块中参数L的分析
    elif (model_name == 'ITransTCNet_L'):
        model = ITransTCNet_L(2, 16, 1000,
                           num_transformer_blocks=2,head_size=16, num_heads=6)

    else:
        raise Exception("'{}' model is not supported model yet!".format(model_name))

    return model

def train_test(train_conf, results_path):
    # 获取训练集、测试集
    X_train, X_val, X_test, y_train, y_val, y_test = getData()

    # Get the current 'IN' time to calculate the overall training time
    in_run = time.time()

    # Get training hyperparamters
    model_name = train_conf.get('model_name')
    batch_size = train_conf.get('batch_size')
    epochs = train_conf.get('epochs')
    lr = train_conf.get('lr')
    win_num = train_conf.get('win_num')

    opt = Adam(learning_rate=lr)

    # Train the model
    model = getModel(train_conf.get('model_name'))
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    #保存最好的模型
    #5个不同频段 delta,theta,alpha,beta,gamma
    filepath = results_path + '/saveML/{}/checkpointL4.h5'.format(model_name)
    callbacks = [ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')]

    history_model = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs ,
                              callbacks=callbacks, verbose=1, validation_data=(X_val, y_val))

    # make prediction on test set.
    probs = model.predict(X_test)
    preds = probs.argmax(axis=-1)
    acc = np.mean(preds == y_test.argmax(axis=-1))

    print("模型名称:{} 参数设置:batch_size={}，epoch={}".format(model_name, batch_size, epochs))
    print("{} Classification accuracy: {:0.4f}".format(model_name, acc))

    # Get the current 'OUT' time to calculate the 'run' training time
    out_run = time.time()
    # Print & write performance measures for each run
    print("{} Train Time: {:0.4f}".format(model_name, (out_run - in_run) / 60))

    # 不保存模型直接报告评价指标值
    # predicted = model.predict(X_test)
    #
    # test_acc = accuracy_score(y_test.argmax(axis=1), predicted.argmax(axis=-1))  # 注：此时传进来的y_test是数组类型
    # test_auc = roc_auc_score(y_test.argmax(axis=1), predicted.argmax(axis=-1))
    # test_reca = recall_score(y_test.argmax(axis=1), predicted.argmax(axis=-1))
    # test_pre = precision_score(y_test.argmax(axis=1), predicted.argmax(axis=-1))
    #
    # precision = precision_score(y_test.argmax(axis=1), predicted.argmax(axis=-1))
    # recall = recall_score(y_test.argmax(axis=1), predicted.argmax(axis=-1))
    # test_f1 = 2 * precision * recall / (precision + recall)
    #
    # # 此种方式为0 1 0 1格式
    # ee_metrics_out = confusion_matrix(y_test.argmax(axis=1), predicted.argmax(axis=-1))
    # test_sens = ee_metrics_out[1][1] / (ee_metrics_out[1][0] + ee_metrics_out[1][1])
    # test_spec = ee_metrics_out[0][0] / (ee_metrics_out[0][0] + ee_metrics_out[0][1])
    #
    # test_kappa = cohen_kappa_score(y_test.argmax(axis=1), predicted.argmax(axis=-1))
    #
    # print("{} accuracy: {:0.4f}".format(model_name, (test_acc)))
    # print("{} auc: {:0.4f} ".format(model_name, (test_auc)))
    # print("{} precision: {:0.4f}".format(model_name, (test_pre)))
    # print("{} recall: {:0.4f}".format(model_name, (test_reca)))
    # print("{} f1score: {:0.4f}".format(model_name, (test_f1)))
    # print("{} sensitivity: {:0.4f}".format(model_name, (test_sens)))
    # print("{} specificity: {:0.4f}".format(model_name, (test_spec)))
    # print("{} kappa: {:0.4f}".format(model_name, (test_kappa)))

    plot_accuracy(model_name, history_model.history['val_accuracy'], history_model.history['accuracy'])
    plot_loss(model_name, history_model.history['val_loss'], history_model.history['loss'])
    #draw_confusion_matrix(model, X_test, y_test, model_name)

#在保存的模型上做测试Evaluation
def test(model, train_conf, results_path, title_name):
    # 获取训练集、测试集
    X_train, X_val, X_test, y_train, y_val, y_test = getData()

    # Get training hyperparamters
    model_name = train_conf.get('model_name')

    # Load the best model out of multiple random runs (experiments).
    model.load_weights(results_path +'/saveML/{}/checkpointL4.h5'.format(model_name))

    # 报告测试集上的评价指标值
    predicted = model.predict(X_test)

    test_acc = accuracy_score(y_test.argmax(axis=1), predicted.argmax(axis=-1))  # 注：此时传进来的y_test是数组类型
    test_auc = roc_auc_score(y_test.argmax(axis=1), predicted.argmax(axis=-1))
    test_reca = recall_score(y_test.argmax(axis=1), predicted.argmax(axis=-1))
    test_pre = precision_score(y_test.argmax(axis=1), predicted.argmax(axis=-1))

    precision = precision_score(y_test.argmax(axis=1), predicted.argmax(axis=-1))
    recall = recall_score(y_test.argmax(axis=1), predicted.argmax(axis=-1))
    test_f1 = 2 * precision * recall / (precision + recall)

    # 此种方式为0 1 0 1格式
    ee_metrics_out = confusion_matrix(y_test.argmax(axis=1), predicted.argmax(axis=-1))
    test_sens = ee_metrics_out[1][1] / (ee_metrics_out[1][0] + ee_metrics_out[1][1])
    test_spec = ee_metrics_out[0][0] / (ee_metrics_out[0][0] + ee_metrics_out[0][1])

    test_kappa = cohen_kappa_score(y_test.argmax(axis=1), predicted.argmax(axis=-1))

    print("{} accuracy: {:0.4f}".format(model_name, (test_acc)))
    print("{} auc: {:0.4f} ".format(model_name, (test_auc)))
    print("{} precision: {:0.4f}".format(model_name, (test_pre)))
    print("{} recall: {:0.4f}".format(model_name, (test_reca)))
    print("{} f1score: {:0.4f}".format(model_name, (test_f1)))
    print("{} sensitivity: {:0.4f}".format(model_name, (test_sens)))
    print("{} specificity: {:0.4f}".format(model_name, (test_spec)))
    print("{} kappa: {:0.4f}".format(model_name, (test_kappa)))

    #绘制混淆矩阵
    #draw_confusion_matrix(model, X_test, y_test, model_name)
    draw_confusion(model, X_test, y_test, title_name)

def run():
    # Create a folder to store the results of the experiment
    results_path = os.getcwd() + "/results"
    if not os.path.exists(results_path):
        os.makedirs(results_path)  # Create a new directory if it does not exist

    # Set training hyperparamters
    train_conf = {'model_name': 'ITransTCNet_L', 'batch_size':64, 'epochs': 100, 'lr': 0.001}
    model_conf = {'title_name': 'ITransTCNet_L'} #此处为修改混淆矩阵的图片名

    # 训练模型 Train the model
    train_test(train_conf, results_path)

    # Load the best model Evaluate the model based on the weights saved in the '/results' folder
    model = getModel(train_conf.get('model_name'))
    title_name = model_conf.get('title_name')

    test(model, train_conf, results_path, title_name)

if __name__ == "__main__":
    run()