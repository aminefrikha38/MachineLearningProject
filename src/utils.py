import numpy as np

def one_hot(y, num_classes=10):
    Y = np.zeros((y.shape[0], num_classes))
    Y[np.arange(y.shape[0]), y] = 1
    return Y

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy(Y, P):
    return -np.mean(np.sum(Y * np.log(P + 1e-8), axis=1))

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)