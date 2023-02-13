# -*- coding: utf-8 -*-
"""
@Author ：HP
@Time ： 2022年06月06日
"""
from glob import glob
from autoreject import AutoReject
from tensorflow.keras.utils import to_categorical
import scipy.io
import mne
import numpy as np

import warnings
warnings.filterwarnings('ignore')

# Electrodes to be considered -
# Source - https://www.egi.com/images/HydroCelGSN_10-10.pdf

Electrode_map = {'C3': 'E36', 'C4': 'E104', 'F3': 'E24', 'F4': 'E124', 'F7': 'E33', 'F8': 'E122',
                 'FP1': 'E22', 'FP2': 'E9', 'O1': 'E70', 'O2': 'E83', 'P3': 'E52', 'P4': 'E92',
                 'T3-T7': 'E45', 'T4-T8': 'E108', 'T5-P7': 'E58', 'T6-P8': 'E96'}
Electrode_understanding = {'Left-Central': 'E36', 'Right-Central': 'E104', 'Front-Left': 'E24', 'Front-Right': 'E124',
                           'Front-Far left': 'E33', 'Front-Far Right': 'E122',
                           'Forehead (above left eye)': 'E22', 'Forehead (above Right eye)': 'E9',
                           'Back-left to Center': 'E70', 'Back-Right to Center': 'E83', 'Back-Above E70': 'E52',
                           'Back-Above E83': 'E92', 'Side Left': 'E45', 'Side Right': 'E108', 'Left': 'E58',
                           'Right': 'E96'}

# 读数据
def data_process_chan(path):
    # Inputs:path = Location of the raw file
    # Outputs:Returns the .raw object for EEG

    # 读入数据
    ch_names = ['E' + str(i + 1) for i in range(128)]
    sampling_freq = 250
    mat = scipy.io.loadmat(path)
    key = list(mat.keys())

    data = key[3]
    a_mat = mat[data]
    info = mne.create_info(ch_names=ch_names, ch_types='eeg', verbose=None, sfreq=sampling_freq)
    df = mne.io.RawArray(a_mat[:-1, :] / 1e6, info)  # 128*...的矩阵
    # mne的rawarray方法，会默认你的数据单位是伏特，但实际你的数据单位是微伏，所以要在这里除以10的6次方，转换成伏特

    # Raw对象主要用来存储连续型数据
    # mne.io.RawArray类来手动创建Raw  使用mne.io.RawArray创建Raw对象时，其构造函数只接受矩阵和info对象
    # 创建info结构,内容包括：通道名称和通道类型 设置采样频率为:sfreq

    # 设置montage通道文件
    montage = mne.channels.make_standard_montage('GSN-HydroCel-128')
    df.set_montage(montage)

    # 滤波数据
    simulated_raw = df.copy().load_data()
    simulated_raw = simulated_raw.notch_filter(freqs=(50))  # 陷波滤波器去掉工频
    simulated_raw_filter = simulated_raw.filter(l_freq=1, h_freq=40, method='fir', fir_window='hamming')  # 高低通滤波

    # 选取特定通道
    raw_rest_16 = simulated_raw_filter.pick_channels(list(Electrode_map.values()))

    # 分段数据
    epochs = mne.make_fixed_length_epochs(raw_rest_16, duration=8.0, overlap=6, preload=True)
    # epoch_data = epochs.get_data()

    # 伪迹处理之后
    ar = AutoReject()
    epochs_ar = ar.fit_transform(epochs)
    epoch_data = epochs_ar.get_data()

    return epoch_data

patient_files_path=glob('/root/landata/datamat/mdd/*.mat')
print(len(patient_files_path))
healthy_files_path=glob('/root/landata/datamat/hc/*.mat')
print(len(healthy_files_path))

patients_epochs_array=[data_process_chan(subject) for subject in patient_files_path]
control_epochs_array=[data_process_chan(subject) for subject in healthy_files_path]

patients_epochs_labels=[len(i)*[1] for i in patients_epochs_array] #患者组1.对照组0
control_epochs_labels=[len(i)*[0] for i in control_epochs_array]
print(len(patients_epochs_labels),len(control_epochs_labels))

data_list=control_epochs_array+patients_epochs_array
label_list=control_epochs_labels+patients_epochs_labels
groups_list=[[i]*len(j) for i, j in enumerate(data_list)]
#print(len(data_list),len(label_list),,len(groups_list)) #6 6

data_array=np.vstack(data_list)   #水平(按列顺序)把数组给堆叠起来，vstack()函数正好和它相反
label_array=np.hstack(label_list) #垂直(按行顺序)把数组给堆叠起来，
group_array=np.concatenate(groups_list)

label_array=to_categorical(label_array)#独热码转换

#此时数据格式为(n,chan,timepoint)
#data_array=np.moveaxis(data_array,1,2)

print(data_array.shape,label_array.shape,group_array.shape)

np.save(file="/root/autodl-tmp/dataset02/data8s.npy",arr=data_array)
data_array=np.load(file="/root/autodl-tmp/dataset02/data8s.npy")

np.save(file="/root/autodl-tmp/dataset02/label8s.npy",arr=label_array)
label_array=np.load(file="/root/autodl-tmp/dataset02/label8s.npy")

np.save(file="/root/autodl-tmp/dataset02/group8s.npy",arr=group_array)
group_array=np.load(file="/root/autodl-tmp/dataset02/group8s.npy")

print(data_array.shape,label_array.shape,group_array.shape)