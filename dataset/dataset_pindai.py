# -*- coding: utf-8 -*-
"""
@Author ：HP
@Time ： 2022年06月06日
"""
from glob import glob
from autoreject import AutoReject
from keras.utils import to_categorical
import scipy.io
import mne
import numpy as np
import scipy.signal as signal
from scipy.signal import butter, lfilter
import warnings
warnings.filterwarnings('ignore')

# Electrodes to be considered -
# Source - https://www.egi.com/images/HydroCelGSN_10-10.pdf

def butter_bandpass(lowcut, highcut, fs, order=5): # fs为采样频率
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass') # 分子b，分母a
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)    # 这个y的格式和data的格式一样
    return y

Electrode_map = {'C3': 'E36', 'C4': 'E104', 'F3': 'E24', 'F4': 'E124', 'F7': 'E33', 'F8': 'E122',
                 'FP1': 'E22', 'FP2': 'E9', 'O1': 'E70', 'O2': 'E83', 'P3': 'E52', 'P4': 'E92',
                 'T3-T7': 'E45', 'T4-T8': 'E108', 'T5-P7': 'E58', 'T6-P8': 'E96'}

# 读数据
def data_process_pindai(path, pin_name):
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
    df = mne.io.RawArray(a_mat[:-1, :]/ 1e6, info)  # 128*...的矩阵
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
    epochs = mne.make_fixed_length_epochs(raw_rest_16, duration=4.0, overlap=3, preload=True)
    # epoch_data = epochs.get_data()

    # 伪迹处理之后
    ar = AutoReject()
    epochs_ar = ar.fit_transform(epochs)
    epoch_data = epochs_ar.get_data()

    # 获取不同频段信号 delta, theta, alpha, beta, gamma
    if (pin_name == 'delta'):
        data = butter_bandpass_filter(epoch_data, 1, 4, fs=250, order=5)
    elif (pin_name == 'theta'):
        data = butter_bandpass_filter(epoch_data, 4, 8, fs=250, order=5)
    elif (pin_name == 'alpha'):
        data = butter_bandpass_filter(epoch_data, 8, 13, fs=250, order=5)
    elif (pin_name == 'beta'):
        data = butter_bandpass_filter(epoch_data, 13, 30, fs=250, order=5)
    elif (pin_name == 'gamma'):
        data = butter_bandpass_filter(epoch_data, 30, 40, fs=250, order=5)
    else:
        raise Exception("'{}' is not supported yet!".format(pin_name))

    return data
#/root/autodl-tmp/landata/datamat/mdd/*.mat'
patient_files_path=glob('/root/autodl-tmp/band_data/datamat/mdd/*.mat')
healthy_files_path=glob('/root/autodl-tmp/band_data/datamat/hc/*.mat')
print(len(healthy_files_path),len(patient_files_path))

######################################## delta #####################################################
patients_delta_array=[data_process_pindai(subject,'delta') for subject in patient_files_path]
control_delta_array=[data_process_pindai(subject,'delta') for subject in healthy_files_path]

patients_delta_labels=[len(i)*[1] for i in patients_delta_array] #患者组1.对照组0
control_delta_labels=[len(i)*[0] for i in control_delta_array]
print(len(patients_delta_labels),len(control_delta_labels))

data_delta_list=control_delta_array+patients_delta_array
label_delta_list=control_delta_labels+patients_delta_labels
print(len(data_delta_list),len(label_delta_list))

data_delta_array=np.concatenate(data_delta_list)   #水平(按列顺序)把数组给堆叠起来，vstack()函数正好和它相反
label_delta_array=np.concatenate(label_delta_list) #垂直(按行顺序)把数组给堆叠起来，
label_delta_array=to_categorical(label_delta_array)#独热码转换

#此时数据格式为(n,chan,timepoint)
print(data_delta_array.shape,label_delta_array.shape)

np.save(file="/root/autodl-tmp/band_data/delta4s.npy",arr=data_delta_array)
data_delta=np.load(file="/root/autodl-tmp/band_data/delta4s.npy")

np.save(file="/root/autodl-tmp/band_data/delta_label4s.npy",arr=label_delta_array)
label_delta=np.load(file="/root/autodl-tmp/band_data/delta_label4s.npy")

print(data_delta.shape,label_delta.shape)

######################################## theta #####################################################
patients_theta_array=[data_process_pindai(subject,'theta') for subject in patient_files_path]
control_theta_array=[data_process_pindai(subject,'theta') for subject in healthy_files_path]

patients_theta_labels=[len(i)*[1] for i in patients_theta_array] #患者组1.对照组0
control_theta_labels=[len(i)*[0] for i in control_theta_array]
print(len(patients_theta_labels),len(control_theta_labels))

data_theta_list=control_theta_array+patients_theta_array
label_theta_list=control_theta_labels+patients_theta_labels
print(len(data_theta_list),len(label_theta_list))

data_theta_array=np.concatenate(data_theta_list)   #水平(按列顺序)把数组给堆叠起来，vstack()函数正好和它相反
label_theta_array=np.concatenate(label_theta_list) #垂直(按行顺序)把数组给堆叠起来，
label_theta_array=to_categorical(label_theta_array)#独热码转换

#此时数据格式为(n,chan,timepoint)
print(data_theta_array.shape,label_theta_array.shape)

np.save(file="/root/autodl-tmp/band_data/theta4s.npy",arr=data_theta_array)
data_theta=np.load(file="/root/autodl-tmp/band_data/theta4s.npy")

np.save(file="/root/autodl-tmp/band_data/theta_label4s.npy",arr=label_theta_array)
label_theta=np.load(file="/root/autodl-tmp/band_data/theta_label4s.npy")

print(data_theta.shape,label_theta.shape)

######################################## alpha #####################################################
patients_alpha_array=[data_process_pindai(subject,'alpha') for subject in patient_files_path]
control_alpha_array=[data_process_pindai(subject,'alpha') for subject in healthy_files_path]

patients_alpha_labels=[len(i)*[1] for i in patients_alpha_array] #患者组1.对照组0
control_alpha_labels=[len(i)*[0] for i in control_alpha_array]
print(len(patients_alpha_labels),len(control_alpha_labels))

data_alpha_list=control_alpha_array+patients_alpha_array
label_alpha_list=control_alpha_labels+patients_alpha_labels
print(len(data_alpha_list),len(label_alpha_list))

data_alpha_array=np.concatenate(data_alpha_list)   #水平(按列顺序)把数组给堆叠起来，vstack()函数正好和它相反
label_alpha_array=np.concatenate(label_alpha_list) #垂直(按行顺序)把数组给堆叠起来，
label_alpha_array=to_categorical(label_alpha_array)#独热码转换

#此时数据格式为(n,chan,timepoint)
print(data_alpha_array.shape,label_alpha_array.shape)

np.save(file="/root/autodl-tmp/band_data/alpha4s.npy",arr=data_alpha_array)
data_alpha=np.load(file="/root/autodl-tmp/band_data/alpha4s.npy")

np.save(file="/root/autodl-tmp/band_data/alpha_label4s.npy",arr=label_alpha_array)
label_alpha=np.load(file="/root/autodl-tmp/band_data/alpha_label4s.npy")

print(data_alpha.shape,label_alpha.shape)

######################################## beta #####################################################
patients_beta_array=[data_process_pindai(subject,'beta') for subject in patient_files_path]
control_beta_array=[data_process_pindai(subject,'beta') for subject in healthy_files_path]

patients_beta_labels=[len(i)*[1] for i in patients_beta_array] #患者组1.对照组0
control_beta_labels=[len(i)*[0] for i in control_beta_array]
print(len(patients_beta_labels),len(control_beta_labels))

data_beta_list=control_beta_array+patients_beta_array
label_beta_list=control_beta_labels+patients_beta_labels
print(len(data_beta_list),len(label_beta_list))

data_beta_array=np.concatenate(data_beta_list)   #水平(按列顺序)把数组给堆叠起来，vstack()函数正好和它相反
label_beta_array=np.concatenate(label_beta_list) #垂直(按行顺序)把数组给堆叠起来，
label_beta_array=to_categorical(label_beta_array)#独热码转换

#此时数据格式为(n,chan,timepoint)
print(data_beta_array.shape,label_beta_array.shape)

np.save(file="/root/autodl-tmp/band_data/beta4s.npy",arr=data_beta_array)
data_beta=np.load(file="/root/autodl-tmp/band_data/beta4s.npy")

np.save(file="/root/autodl-tmp/band_data/beta_label4s.npy",arr=label_beta_array)
label_beta=np.load(file="/root/autodl-tmp/band_data/beta_label4s.npy")

print(data_beta.shape,label_beta.shape)

######################################## gamma #####################################################
patients_gamma_array=[data_process_pindai(subject,'gamma') for subject in patient_files_path]
control_gamma_array=[data_process_pindai(subject,'gamma') for subject in healthy_files_path]

patients_gamma_labels=[len(i)*[1] for i in patients_gamma_array] #患者组1.对照组0
control_gamma_labels=[len(i)*[0] for i in control_gamma_array]
print(len(patients_gamma_labels),len(control_gamma_labels))

data_gamma_list=control_gamma_array+patients_gamma_array
label_gamma_list=control_gamma_labels+patients_gamma_labels
print(len(data_gamma_list),len(label_gamma_list))

data_gamma_array=np.concatenate(data_gamma_list)   #水平(按列顺序)把数组给堆叠起来，vstack()函数正好和它相反
label_gamma_array=np.concatenate(label_gamma_list) #垂直(按行顺序)把数组给堆叠起来，
label_gamma_array=to_categorical(label_gamma_array)#独热码转换

#此时数据格式为(n,chan,timepoint)
print(data_gamma_array.shape,label_gamma_array.shape)

np.save(file="/root/autodl-tmp/band_data/gamma4s.npy",arr=data_gamma_array)
data_gamma=np.load(file="/root/autodl-tmp/band_data/gamma4s.npy")

np.save(file="/root/autodl-tmp/band_data/gamma_label4s.npy",arr=label_gamma_array)
label_gamma=np.load(file="/root/autodl-tmp/band_data/gamma_label4s.npy")

print(data_gamma.shape,label_gamma.shape)