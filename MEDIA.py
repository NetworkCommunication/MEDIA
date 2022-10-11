import torch
import scipy.io as sio
import numpy as np
import divide
import accuracy
import weighted_sum

from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import inc_model


model = inc_model.Activation_Net(10*2, 120, 80, 10*2)

if torch.cuda.is_available():
    model = model.cuda()


t_setx=sio.loadmat('./data/t_setx.mat')['t_setx']
t_setx=np.array(t_setx)
d_set6=sio.loadmat('./data/d_set6.mat')['d_set6']
d_set6=np.array(d_set6)
X_train, X_test = divide.divide(t_setx)
Y_train, Y_test = divide.divide(d_set6)

model.incremental_update(X_train, Y_train, X_test, Y_test)

num =0
for x in X_test:
    x= torch.tensor(x).double()
    a=model.forward(x)
    a=torch.sigmoid(a)
    a.tolist()
    y=[]
    for i in a:
        if i>=0.5:
            y.append(int(1))
        elif i<0.5:
            y.append(int(0))
    if num==0:
        Y=y
    elif num>0:
        Y=np.vstack( (Y,y) )
    num = num + 1


per = weighted_sum.per_q(X_test, Y)
p_a= accuracy.per_acu(Y_test , Y)

model.update_fre(p_a)

#torch.save(model, './model/incemental.pth')








