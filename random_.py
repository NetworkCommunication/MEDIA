import numpy as np
import scipy.io as sio
import random
import torch
import weighted_sum
import divide
import accuracy


def rdm(X_test):
    num=1
    for i in range( len(X_test) ):
        Y_local = []
        for i in range(20):
            y_ = random.random()
            if y_>0.5:
                Y_local.append(1)
            elif y_<0.5:
                Y_local.append(0)

        if num == 1:
            Y_l = Y_local
        elif num > 1:
            Y_l = np.vstack((Y_l, Y_local))
        num = num + 1
    return  Y_l

t_setx=sio.loadmat('./data/t_setx.mat')['t_setx']
t_setx=np.array(t_setx)
d_set6=sio.loadmat('./data/d_set6.mat')['d_set6']
d_set6=np.array(d_set6)
X_train, t_set = divide.divide(t_setx)
Y_train, d_set = divide.divide(d_set6)


y_local = rdm(t_set)
per = weighted_sum.per_q(t_set, y_local)
p_a= accuracy.per_acu(d_set , y_local)