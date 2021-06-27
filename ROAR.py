import numpy as np
from scipy.stats import rankdata
from train import train_DBN

def ROAR_DBN2(data,label,n_iter=2):
    # Integrated Gradients
    acc=[]
    feature_importance=[]
    gradient,accuracy = train_DBN(data,label,n_iter=n_iter)
    acc.append(accuracy)
    feature_importance.append(gradient)
    rank = rankdata(gradient)
    min_index=np.where(rank==32)
    new_data = data.transpose(1,0)
    new_label = label
    min = new_data[min_index]
    for i in range(1,33):
        print("{}trial".format(i))
        trans=np.where(rank<=i)
        for i in trans:
            new_data[i]=min
            new_label[i] = label[i]
        pre_data=new_data.transpose(1,0)
        gradient,accuracy = train_DBN(pre_data,new_label,n_iter=n_iter)
        acc.append(accuracy)
        feature_importance.append(gradient)
    return acc,feature_importance