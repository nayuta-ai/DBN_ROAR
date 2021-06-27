import os
import pickle
import math
import mne
import scipy.stats
import numpy as np
import scipy.io as sio
from scipy.stats import rankdata
from scipy.fftpack import fft,ifft
from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

from Preprocessing import DEAP_preprocess
from Feature_extraction import feature_extract
from train import train_DBN
from plot import topomap,curve
from ROAR import ROAR_DBN2

# DataLoad
dataset_dir = "/home/DEAP/s01.mat"
emotion = "valence"
n_iter = 100
data,label = DEAP_preprocess(dataset_dir,emotion)
param = {'stftn':128,'fStart':[4,8,14,31],'fEnd':[7,13,30,50],'window':1,'fs':128}
de = feature_extract(data,param)
acc, feature = ROAR_DBN2(de[3],label,n_iter=n_iter)
curve(acc)
with open("./data/feature.pickle",mode="wb") as f:
    pickle.dump(feature,f)