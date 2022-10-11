import numpy as np
import scipy.io as sio
import random_
import torch


def divide(task_size):
    N = len(task_size)
    N =N * 0.8
    num = 1

    for i in task_size:
        i = i.tolist()
        if num == 1:
            X_train = i
        if num <= N and num > 1:
            X_train = np.vstack((X_train, i))
        if num == (N + 1):
            X_test = i
        if num > (N + 1):
            X_test = np.vstack((X_test, i))
        num = num + 1
    return X_train, X_test