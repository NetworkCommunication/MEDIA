import numpy as np
import scipy.io as sio
import random_
import torch
import weighted_sum
import divide

def greedy (tasks):
    for i in range(3**10):
        a=i
        ds=[]
        for j in range(10):
            b=int(a%3)
            a=int(a/3)
            if b==0:
                ds.append(0)
                ds.append(0)
            if b==1:
                ds.append(0)
                ds.append(1)
            if b==2:
                ds.append(1)
                ds.append(1)

        if i==0:
            Q=weighted_sum.Q_value_x(tasks, ds)
            decisions=ds
        elif i>0:
            if Q>weighted_sum.Q_value_x(tasks, ds):
                Q = weighted_sum.Q_value_x(tasks, ds)
                decisions = ds
    #print(Q)

    return decisions

t_setx=sio.loadmat('./data/t_setx.mat')['t_setx']
t_setx=np.array(t_setx)
d_set6=sio.loadmat('./data/d_set6.mat')['d_set6']
d_set6=np.array(d_set6)
X_train, t_set = divide.divide(t_setx)
Y_train, d_set = divide.divide(d_set6)

num=1
for i in t_set:
    y=greedy(i)
    if num==1:
        Y=y
    elif num>1:
        Y = np.vstack((Y, y))
    num=num+1

d_set=Y
per = weighted_sum.per_q(t_set, d_set)