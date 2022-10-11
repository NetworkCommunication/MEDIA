import torch
import scipy.io as sio
import inc_model
import time
import numpy as np
import random

#执行时间

def task_d(n):
    tasks=[]
    for i in range(n):
        tasks.append(15)
        tasks.append(20)
    return tasks

def ex_time(tasks, model):
    x=tasks
    x = torch.tensor(x).double()
    start = time.time()
    for i in range(1000):
        a = model.forward(x)
        a = torch.sigmoid(a)
        a.tolist()
    end = time.time()
    ys = end - start
    ys/=1000
    return ys

def rdm(n):
    num=1
    start = time.time()
    for i in range( 1000 ):
        Y_local = []
        for i in range( n):
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

    end = time.time()
    ys = end - start
    ys /= 1000
    return  ys

model1 = inc_model.Activation_Net(10*2, 120, 80, 10*2)
model2 = inc_model.Activation_Net(20*2, 120, 80, 20*2)
model3 = inc_model.Activation_Net(30*2, 120, 80, 30*2)
model4 = inc_model.Activation_Net(40*2, 120, 80, 40*2)
model5 = inc_model.Activation_Net(50*2, 120, 80, 50*2)

tasks1=task_d(10)
tasks2=task_d(20)
tasks3=task_d(30)
tasks4=task_d(40)
tasks5=task_d(50)


ys=ex_time(tasks1, model1)
ys1 = ys

ys=ex_time(tasks2, model2)
ys1 = np.vstack((ys1, ys))

ys=ex_time(tasks3, model3)
ys1 = np.vstack((ys1, ys))

ys=ex_time(tasks4, model4)
ys1 = np.vstack((ys1, ys))

ys=ex_time(tasks5, model5)
ys1 = np.vstack((ys1, ys))

print(ys1)



#print(ys1)

ys=rdm(10)
ys2=ys
ys=rdm(20)
ys2 = np.vstack((ys2, ys))
ys=rdm(30)
ys2 = np.vstack((ys2, ys))
ys=rdm(40)
ys2 = np.vstack((ys2, ys))
ys=rdm(50)
ys2 = np.vstack((ys2, ys))
print(ys2)

