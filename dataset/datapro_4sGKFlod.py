# -*- coding: utf-8 -*-
"""
@Author ：HP
@Time ： 2022年06月06日
"""
from glob import glob
import scipy.io
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
    epochs = mne.make_fixed_length_epochs(raw_rest_16, duration=4.0, overlap=3, preload=True)
    #epoch_data = epochs.get_data()

    #伪迹处理之后
    ar = AutoReject()
    epochs_ar = ar.fit_transform(epochs)
    epoch_data = epochs_ar.get_data()

    return epoch_data

patient_files_path=glob('/root/autodl-tmp/landata/datamat/mdd/*.mat')
healthy_files_path=glob('/root/autodl-tmp/landata/datamat/hc/*.mat')
print(len(patient_files_path),len(healthy_files_path))

patients_epochs_array=[data_process_chan(subject) for subject in patient_files_path]
control_epochs_array=[data_process_chan(subject) for subject in healthy_files_path]

patients_epochs_labels=[len(i)*[1] for i in patients_epochs_array] #患者组1.对照组0
control_epochs_labels=[len(i)*[0] for i in control_epochs_array]
print(len(patients_epochs_labels),len(control_epochs_labels))

data_list=control_epochs_array+patients_epochs_array
label_list=control_epochs_labels+patients_epochs_labels
groups_list=[[i]*len(j) for i, j in enumerate(data_list)]

#按组划分数据
data_array1=np.concatenate((data_list[0],data_list[29],data_list[1],data_list[30],data_list[2],data_list[31]))
data_array2=np.concatenate((data_list[3],data_list[32],data_list[4],data_list[33],data_list[5],data_list[34]))

data_array3=np.concatenate((data_list[6],data_list[35],data_list[7],data_list[36],data_list[8],data_list[37]))
data_array4=np.concatenate((data_list[9],data_list[38],data_list[10],data_list[39],data_list[11],data_list[40]))
data_array5=np.concatenate((data_list[12],data_list[41],data_list[13],data_list[42],data_list[14],data_list[43]))

data_array6=np.concatenate((data_list[15],data_list[44],data_list[16],data_list[45]))

data_array7=np.concatenate((data_list[17],data_list[46],data_list[18],data_list[47],data_list[19]))
data_array8=np.concatenate((data_list[20],data_list[48],data_list[21],data_list[49],data_list[22]))
data_array9=np.concatenate((data_list[23],data_list[50],data_list[24],data_list[51],data_list[25]))

data_array10=np.concatenate((data_list[26],data_list[52],data_list[27],data_list[52]))

data_array=np.concatenate((data_array1,data_array2,data_array3,data_array4,data_array5,data_array6,data_array7,data_array8,data_array9,data_array10))

label_list1=label_list[0]+label_list[29]+label_list[1]+label_list[30]+label_list[2]+label_list[31]
label_list2=label_list[3]+label_list[32]+label_list[4]+label_list[33]+label_list[5]+label_list[34]

label_list3=label_list[6]+label_list[35]+label_list[7]+label_list[36]+label_list[8]+label_list[37]
label_list4=label_list[9]+label_list[38]+label_list[10]+label_list[39]+label_list[11]+label_list[40]
label_list5=label_list[12]+label_list[41]+label_list[13]+label_list[42]+label_list[14]+label_list[43]

label_list6=label_list[15]+label_list[44]+label_list[16]+label_list[45]

label_list7=label_list[17]+label_list[46]+label_list[18]+label_list[47]+label_list[19]
label_list8=label_list[20]+label_list[48]+label_list[21]+label_list[49]+label_list[22]
label_list9=label_list[23]+label_list[50]+label_list[24]+label_list[51]+label_list[25]

label_list10=label_list[26]+label_list[52]+label_list[27]+label_list[52]

label_array=np.concatenate((label_list1,label_list2,label_list3,label_list4,label_list5,label_list6,label_list7,label_list8,label_list9,label_list10))
label_array=to_categorical(label_array)#独热码转换

group1=groups_list[0]+groups_list[29]+groups_list[1]+groups_list[30]+groups_list[2]+groups_list[31]
group2=groups_list[3]+groups_list[32]+groups_list[4]+groups_list[33]+groups_list[5]+groups_list[34]

group3=groups_list[6]+groups_list[35]+groups_list[7]+groups_list[36]+groups_list[8]+groups_list[37]
group4=groups_list[9]+groups_list[38]+groups_list[10]+groups_list[39]+groups_list[11]+groups_list[40]
group5=groups_list[12]+groups_list[41]+groups_list[13]+groups_list[42]+groups_list[14]+groups_list[43]

group6=groups_list[15]+groups_list[44]+groups_list[16]+groups_list[45]

group7=groups_list[17]+groups_list[46]+groups_list[18]+groups_list[47]+groups_list[19]
group8=groups_list[20]+groups_list[48]+groups_list[21]+groups_list[49]+groups_list[22]
group9=groups_list[23]+groups_list[50]+groups_list[24]+groups_list[51]+groups_list[25]

group10=groups_list[26]+groups_list[52]+groups_list[27]+groups_list[52]

group_label1=[1 for i in enumerate(group1)]
group_label2=[2 for i in enumerate(group2)]
group_label3=[3 for i in enumerate(group3)]
group_label4=[4 for i in enumerate(group4)]
group_label5=[5 for i in enumerate(group5)]
group_label6=[6 for i in enumerate(group6)]
group_label7=[7 for i in enumerate(group7)]
group_label8=[8 for i in enumerate(group8)]
group_label9=[9 for i in enumerate(group9)]
group_label10=[10 for i in enumerate(group10)]

group_array=np.concatenate((group_label1,group_label2,group_label3,group_label4,group_label5,group_label6,group_label7,group_label8,group_label9,group_label10))

print(data_array.shape,label_array.shape,group_array.shape)

np.save(file="/root/autodl-tmp/dataset4s_GKF/data4s.npy",arr=data_array)
data_array=np.load(file="/root/autodl-tmp/dataset4s_GKF/data4s.npy")

np.save(file="/root/autodl-tmp/dataset4s_GKF/label4s.npy",arr=label_array)
label_array=np.load(file="/root/autodl-tmp/dataset4s_GKF/label4s.npy")

np.save(file="/root/autodl-tmp/dataset4s_GKF/group4s.npy",arr=group_array)
group_array=np.load(file="/root/autodl-tmp/dataset4s_GKF/group4s.npy")

print(data_array.shape,label_array.shape,group_array.shape)