import torch
import time
from torch import nn, optim
from torch.autograd import Variable

import numpy as np
import copy
import scipy.io as sio

torch.set_default_tensor_type(torch.DoubleTensor)


class Activation_Net(nn.Module):


    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Activation_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))

        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

        self.batch_size = 64
        self.learning_rate = 0.02
        self.num_epoches = 20





    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

    def fenge(self, task_size):
        N = len(task_size)
        N = 6000
        num = 1

        for i in task_size:
            i = i.tolist()
            if num == 1:
                X_train = i
            if num <=  N and num > 1:
                X_train = np.vstack((X_train, i))
            if num == ( N + 1):
                X_test = i
            if num > ( N + 1):
                X_test = np.vstack((X_test, i))
            num = num + 1
        return X_train, X_test


    def update(self, X_train, Y_train):
        # 定义损失函数和优化器
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.BCELoss()
        # optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        # optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        # optimizer = optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        optimizer = optim.Adam(self.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)

        epoch = 0

        for s in range(100):
            for task_size, label in zip(X_train, Y_train):
                task_size = torch.tensor(task_size).double()
                label = torch.tensor(label).double()
                # task_size = task_size.view(task_size.size(0), -1)
                if torch.cuda.is_available():
                    task_size = task_size.cuda()
                    label = label.cuda()
                else:
                    task_size = Variable(task_size)
                    label = Variable(label)
                logist_0 = self.forward(task_size)
                logist = torch.sigmoid(logist_0)

                cls_loss = criterion(logist, label)

                loss = cls_loss

                print_loss = loss.data.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch += 1



                if epoch % 10000 == 0:
                    print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))



