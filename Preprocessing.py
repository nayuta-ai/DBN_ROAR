import numpy as np
import scipy.io as sio
from scipy.stats import rankdata
from scipy.fftpack import fft,ifft
# DataLoad
dataset_dir = "/home/DEAP/s01.mat"
emotion = "valence"
def baseline_remove(datasets):
    data_in = datasets[0:40,0:32,0:8064]
    base_signal = (data_in[0:40,0:32,0:128]+data_in[0:40,0:32,128:256]+data_in[0:40,0:32,256:384])/3
    data = data_in[0:40,0:32,384:8064]
    ### baseline removal
    for i in range(0,60):
        data[0:40,0:32,i*128:(i+1)*128]=data[0:40,0:32,i*128:(i+1)*128]-base_signal
    return data

def label_preprocess(emotion):
    for i in range(0,40):
        if emotion[i]>5:
            emotion[i]=1
        else:
            emotion[i]=0
    return emotion

def DEAP_preprocess(dir,emotion):
    data =sio.loadmat(dir)
    datasets=data['data']
    labels=data['labels']
    data = baseline_remove(datasets)
    labels = labels.transpose(1,0)
    if emotion == "valence":
        label = labels[0]
    elif emotion == "arousal":
        label = labels[1]
    else:
        print("label is not founded")
    label = label_preprocess(label)
    data_eeg=np.zeros([40,60,32,128])
    label_eeg=np.zeros([40,60,1])
    for i in range(0,40):
        for j in range(0,60):
            data_eeg[i][j]=data[i,0:32,i*128:(i+1)*128]
            label_eeg[i][j]=label[i]
    data_eeg = data_eeg.reshape(-1,32,128)
    label_eeg = label_eeg.astype(np.int64).reshape(-1)
    return data_eeg, label_eeg
"""
data,label = DEAP_preprocess(dataset_dir,emotion)
print(data.shape)
print(label.shape)
"""