import numpy as np
from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

def train_DBN(data,label,n_iter=10):
    X_train, X_test, Y_train, Y_test = train_test_split(data, label,
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    random_state=0)
    X_train = (X_train - np.min(X_train, 0)) / (np.max(X_train, 0) + 0.0001)  # 0-1 scaling
    X_test = (X_test - np.min(X_test, 0)) / (np.max(X_test, 0) + 0.0001)  # 0-1 scaling
    logistic = linear_model.LogisticRegression(C=100)
    rbm1 = BernoulliRBM(n_components=200, learning_rate=0.06, n_iter=n_iter, verbose=1, random_state=101)
    rbm2 = BernoulliRBM(n_components=200, learning_rate=0.06, n_iter=n_iter, verbose=1, random_state=101)
    DBN2 = Pipeline(steps=[('rbm1', rbm1),('rbm2', rbm2),('logistic', logistic)])
    DBN2.fit(X_train, Y_train)
    accuracy = metrics.accuracy_score(
            Y_test,
            DBN2.predict(X_test))
    print("Logistic regression using RBM features Accuracy:\n%s\n" % (accuracy))
    coef1 = rbm1.components_.transpose(1,0)
    coef2 = rbm2.components_.transpose(1,0)
    coef3 = logistic.coef_.transpose(1,0)
    IG = np.dot(np.dot(coef1,coef2),coef3)
    return IG,accuracy
