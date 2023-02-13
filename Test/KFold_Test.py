# -*- coding: utf-8 -*-
"""
@Author ：HP
@Time ： 2022年09月04日
"""
from IncepTransTCNet.ITransTCNet import ITransTCNet
from IncepTransTCNet.ITransTCNet_noTCN import ITransTCNet_noTCN
from IncepTransTCNet.ITransTCNet_noTrans import ITransTCNet_noTrans
from IncepTransTCNet.Incep_TCN_Trans import ITCNTrans
from alation.Inception import InceptionNet
from compareModels.CNN_LSTM import CNNLSTM
from compareModels.baseModel import EEGNet4_8, EEGNet8_16, ShallowConvNet, DeepConvNet
from compareModels.ChroNet import ChroNet
from compareModels.EEGNeX_8_32 import EEGNeX_8_32
from compareModels.EEGNetTCNet import EEGTCNet
from compareModels.EEGNetTCNet_Fusion import TCNet_Fusion
from compareModels.Inception import EEG_Incep
from compareModels.InceptionTCN import EEG_ITNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
from picture.plot_conf import draw_KFold_confusion_matrix
warnings.filterwarnings('ignore')
import os
import time
import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')
from sklearn.model_selection import StratifiedKFold
#StratifiedKFold用法类似Kfold，但是它是分层采样，确保训练集，验证集中各类别样本的比例与原始数据集中相同。因此一般使用StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import f1_score, cohen_kappa_score, classification_report, ConfusionMatrixDisplay
# kappa score是一个介于(-1, 1)之间的数. score>0.8意味着好的分类；0或更低意味着不好（实际是随机标签）

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
        model = ChroNet(2, 16, 2000)

    #一篇DP论文中的模型
    elif (model_name == 'CNNLSTM'):
        model = CNNLSTM(2, 16, 2000)

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

    else:
        raise Exception("'{}' model is not supported model yet!".format(model_name))

    return model

def train_test(train_conf, results_path):
    ee_accuracy = []
    ee_aucC =[]
    ee_precision = []
    ee_recall = []

    ee_f1score = []
    ee_macro_f1score = []
    ee_sensitivity = []
    ee_specificity = []

    ee_kappa = []

    # 加载数据没有重叠的4s数据
    # data_array = np.load(file="/root/autodl-tmp/dataset01/data10s.npy")
    # label_array = np.load(file="/root/autodl-tmp/dataset01/label10s.npy")

    # 加载数据有重叠75%的4s数据
    # data_array = np.load(file="/root/autodl-tmp/dataset4s_GKF/data4s.npy")
    # label_array = np.load(file="/root/autodl-tmp/dataset4s_GKF/label4s.npy")

    # 分别加载5种不同频带数据16

    # (10374, 16, 1000) (3012, 16, 1000) (1488, 16, 1000)
    # data_array = np.load(file="/root/autodl-tmp/band_data/delta4s.npy")
    # label_array = np.load(file="/root/autodl-tmp/band_data/delta_label4s.npy")

    # (10290, 16, 1000) (2988, 16, 1000) (1476, 16, 1000)
    # data_array = np.load(file="/root/autodl-tmp/band_data/theta4s.npy")
    # label_array = np.load(file="/root/autodl-tmp/band_data/theta_label4s.npy")

    # (10385, 16, 1000) (3015, 16, 1000) (1489, 16, 1000)
    # data_array = np.load(file="/root/autodl-tmp/band_data/alpha4s.npy")
    # label_array = np.load(file="/root/autodl-tmp/band_data/alpha_label4s.npy")

    # (10295, 16, 1000) (2989, 16, 1000) (1476, 16, 1000)
    # data_array = np.load(file="/root/autodl-tmp/band_data/beta4s.npy")
    # label_array = np.load(file="/root/autodl-tmp/band_data/beta_label4s.npy")

    # (10252, 16, 1000) (2977, 16, 1000) (1470, 16, 1000)
    data_array = np.load(file="/root/autodl-tmp/band_data/gamma4s.npy")
    label_array = np.load(file="/root/autodl-tmp/band_data/gamma_label4s.npy")

    # 运行开始时间
    in_run = time.time()

    n_splits = train_conf.get('n_splits')
    # 存放10折交叉验证的混淆矩阵
    matrix = np.zeros([n_splits, 2, 2])

    # K折交叉验证
    sfk = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    #Whether to shuffle each class's samples before splitting into batches.
    #Note that the samples within each split will not be shuffled. default=False
    #当shuffle=False时，将保留数据集排序中的顺序依赖关系。也就是说，某些验证集中来自类 k 的所有样本在 y 中是连续的。
    for (train_index, test_index) ,k in zip(sfk.split(data_array, label_array.argmax(1)),range(n_splits)):
        print('************************第{}折交叉验证***************************'.format(k+1))
        X_train, X_test = data_array[train_index], data_array[test_index]
        y_train, y_test = label_array[train_index], label_array[test_index]

        # 划分训练集、训练集
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True, random_state=42)
        #Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None. default=True

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

        print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

        # Get training hyperparamters
        model_name = train_conf.get('model_name')
        batch_size = train_conf.get('batch_size')
        epochs = train_conf.get('epochs')
        lr = train_conf.get('lr')
        opt = Adam(learning_rate=lr)

        # Train the model
        model = getModel(train_conf.get('model_name'))
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        #保存最好的模型
        #filepath = results_path + '/saveModel/{}/{}-Fold/checkpoint.h5'.format(model_name, k + 1)
        filepath = results_path + '/saveModel/band_model/gamma/{}/{}-Fold/checkpoint.h5'.format(model_name,k+1)
        callbacks = [ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')]

        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs ,callbacks=callbacks, verbose=1, validation_data=(X_val, y_val))

        print('------------------不保存模型时的测试结果---------------------')
        probss = model.predict(X_test)
        acc = np.mean(probss.argmax(axis=-1) == y_test.argmax(axis=-1))
        print("{} Classification accuracy: {:0.4f}".format(model_name, acc))

        test_kappa = cohen_kappa_score(y_test.argmax(axis=1), probss.argmax(axis=-1))
        print("{} kappa: {:0.4f}".format(model_name, (test_kappa)))

        # 加载保存模型，然后在保存模型上测试
        #model.load_weights(results_path + '/saveModel/{}/{}-Fold/checkpoint.h5'.format(model_name, k + 1))
        model.load_weights(results_path + '/saveModel/band_model/gamma/{}/{}-Fold/checkpoint.h5'.format(model_name, k + 1))

        print('------------------保存模型时的测试结果---------------------')
        # 报告测试集上的评价指标值
        probs = model.predict(X_test)

        ee_accu = accuracy_score(y_test.argmax(axis=-1), probs.argmax(axis=-1))
        ee_accuracy.append(ee_accu)
        print("第{}折时，{} Classification accuracy: {:0.4f}".format(k + 1, model_name, ee_accu))

        ee_auc = roc_auc_score(y_test.argmax(axis=1), probs.argmax(axis=-1))
        ee_aucC.append(ee_auc)
        print("第{}折时，{} auc: {:0.4f}".format(k + 1, model_name, ee_auc))

        ee_prec = precision_score(y_test.argmax(axis=-1), probs.argmax(axis=-1))
        ee_precision.append(ee_prec)
        print("第{}折时，{} precision: {:0.4f}".format(k + 1, model_name, ee_prec))

        ee_reca = recall_score(y_test.argmax(axis=-1), probs.argmax(axis=-1))
        ee_recall.append(ee_reca)
        print("第{}折时，{} recall: {:0.4f}".format(k + 1, model_name, ee_reca))

        ee_f1 = f1_score(y_test.argmax(axis=-1), probs.argmax(axis=-1))
        ee_f1score.append(ee_f1)
        print("第{}折时，{} f1score: {:0.4f}".format(k + 1, model_name, ee_f1))

        ee_macro_f1 = f1_score(y_test.argmax(axis=-1), probs.argmax(axis=-1), average='macro')
        ee_macro_f1score.append(ee_macro_f1)
        print("第{}折时，{} macro_f1score: {:0.4f}".format(k + 1, model_name, ee_macro_f1))

        # 此种方式为0 1 0 1格式
        ee_metrics_out = confusion_matrix(y_test.argmax(axis=-1), probs.argmax(axis=-1))

        ee_sens = ee_metrics_out[1][1] / (ee_metrics_out[1][0] + ee_metrics_out[1][1])
        ee_sensitivity.append(ee_sens)
        print("第{}折时，{} sensitivity: {:0.4f}".format(k + 1, model_name, ee_sens))

        ee_spec = ee_metrics_out[0][0] / (ee_metrics_out[0][0] + ee_metrics_out[0][1])
        ee_specificity.append(ee_spec)
        print("第{}折时，{} specificity: {:0.4f}".format(k + 1, model_name, ee_spec))

        ee_kap = cohen_kappa_score(y_test.argmax(axis=1), probs.argmax(axis=-1))
        ee_kappa.append(ee_kap)
        print("第{}折时，{} kappa: {:0.4f}".format(k + 1, model_name, ee_kap))

        # matrix[k, :, :] = confusion_matrix(y_test.argmax(axis=1), probs.argmax(axis=1))  #显示个数
        matrix[k, :, :] = confusion_matrix(y_test.argmax(axis=1), probs.argmax(axis=1), normalize='pred')  # 显示概率

    #报告评价指标值
    model_name = train_conf.get('model_name')
    print("*****************{}performance report********************".format(model_name))

    # K折运行结束时间
    out_run = time.time()
    run_time = (out_run - in_run) / 60
    print("{} Train Time: {:0.4f}".format(model_name, run_time))

    ee_accuracy = np.array(ee_accuracy)
    print("K fold average mean accuracy: {:0.4f} std:{:0.4f}".format(ee_accuracy.mean(), ee_accuracy.std()))

    ee_aucC=np.array(ee_aucC)
    print("K fold average mean auc: {:0.4f} std:{:0.4f}".format(ee_aucC.mean(), ee_aucC.std()))

    ee_precision = np.array(ee_precision)
    print("K fold average mean precision: {:0.4f} std:{:0.4f}".format(ee_precision.mean(), ee_precision.std()))

    ee_recall = np.array(ee_recall)
    print("K fold average mean recall: {:0.4f} std:{:0.4f}".format(ee_recall.mean(), ee_recall.std()))

    ee_f1score = np.array(ee_f1score)
    print("K fold average mean f1score: {:0.4f} std:{:0.4f}".format(ee_f1score.mean(), ee_f1score.std()))

    ee_macro_f1score = np.array(ee_macro_f1score)
    print("K fold average mean macro_f1score: {:0.4f} std:{:0.4f}".format(ee_macro_f1score.mean(),ee_macro_f1score.std()))

    ee_sensitivity = np.array(ee_sensitivity)
    print("K fold average mean sensitivity: {:0.4f} std:{:0.4f}".format(ee_sensitivity.mean(), ee_sensitivity.std()))

    ee_specificity = np.array(ee_specificity)
    print("K fold average mean specificity: {:0.4f} std:{:0.4f}".format(ee_specificity.mean(), ee_specificity.std()))

    ee_kappa=np.array(ee_kappa)
    print("K fold average mean kappa: {:0.4f} std:{:0.4f}".format(ee_kappa.mean(), ee_kappa.std()))

    #绘制10折交叉验证的混淆矩阵图
    draw_KFold_confusion_matrix(matrix.mean(0), model_name)

def run():
    # Create a folder to store the results of the experiment
    results_path = os.getcwd() + "/results"
    #results_path = "H:/eegpaper/111/sunproject01/tsne/results"
    if not os.path.exists(results_path):
        os.makedirs(results_path)  # Create a new directory if it does not exist

    # Set training hyperparamters
    train_conf = {'model_name': 'ITransTCNet', 'batch_size': 64, 'epochs': 100, 'n_splits':10, 'lr': 0.001}

    # 训练模型 Train the model
    train_test(train_conf, results_path)

if __name__ == "__main__":
    run()