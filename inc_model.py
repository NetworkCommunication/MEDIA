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
        self.num_update = 0
        self.frequency = 10

    def distillation_loss(self, logist, lable, T):

        label = Variable(lable)
        # outputs = torch.log_softmax(logist / T, dim=1)  # 计算softmax值的log值
        outputs = torch.sigmoid(logist / T)

        outputs = torch.log(outputs)
        label = torch.sigmoid(label / T)  # softmax(label)

        outputs = -torch.sum(outputs * label, dim=0, keepdim=False)
        # outputs = -torch.mean(outputs, dim=0, keepdim=False)

        return Variable(outputs / 3)

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

    def update_fre(self,acc):
        if acc >= 0.90:
            self.frequency=1
        elif acc<0.90:
            self.frequency=10


    ##增量式训练
    def incremental_update(self, train_data, lable_data, X_test, Y_test):
        N = len(train_data)
        #print(N)
        num = 1
        train_num = 6  ##训练集的数量
        a_cc=[]

        for i, j in zip(train_data, lable_data):
            i = i.tolist()
            j = j.tolist()

            if num <= (N + 1):  # 训练集分割
                if num % (N / train_num) == 1:
                    X_train = i
                    X_lable = j
                else:
                    X_train = np.vstack((X_train, i))
                    X_lable = np.vstack((X_lable, j))
                if num % (N / train_num) == 0:
                    if num > 1:

                        start = time.time()
                        self.update(X_train, X_lable)
                        #self.num_update = self.num_update + 1
                        end = time.time()
                        ys = end - start
                        num_corr = 0  # 正确的数量
                        for t, y in zip(X_test, Y_test):
                            t = torch.tensor(t).double()
                            dec = self.forward(t)
                            dec = torch.sigmoid(dec)
                            dec.tolist()
                            for i_, j_ in zip(dec, y):
                                if (i_ - j_) < 0.5 and (i_ - j_) > -0.5:
                                    num_corr = num_corr + 1

                        corr = num_corr / (20 * len(X_test))
                        print('阶段：', self.num_update, '准确率：', corr)
                        a_cc.append(corr)
                        if self.num_update == 1:
                            a_c = corr
                            tm = ys
                        if self.num_update > 1:
                            a_c = np.vstack((a_c, corr))
                            tm = np.vstack((tm, ys))

            num = num + 1

        #a_c = np.array(a_c)
        #print(a_c)
        #print(tm)
        #sio.savemat('./data/e7_3.mat', {'e7_3': a_c})
        #sio.savemat('./data/e6_3.mat', {'e6_3': tm})
        print(a_cc)

    def update(self, X_train, Y_train):
        # 定义损失函数和优化器
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.BCELoss()  # 损失函数选择二进制交叉熵损失函数
        # optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)   #随机梯度下降算法
        # optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)  #优化器选亚当算法
        # optimizer = optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        optimizer = optim.Adam(self.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)
        # Save a copy to compute distillation outputs 保存副本以计算蒸馏输出
        prev_model = copy.deepcopy(self)

        # prev_model.cuda()          # 指定GPU

        # 训练模型
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
                logist = torch.sigmoid(logist_0)  # 对输出使用sigmoid激活函数

                cls_loss = criterion(logist, label)  # 计算损失函数

                if self.num_update > 0:  # 判断是否是第一次训练
                    dist_target = prev_model.forward(task_size)  # 旧模型对新任务的逻辑值
                    logits_dist = logist_0  # 新模型对新任务的输出（切片后的，为了匹配旧模型的输出格式）
                    dist_loss = self.distillation_loss(logits_dist, dist_target, 2)  # 计算蒸馏损失函数 2

                    loss = dist_loss + cls_loss

                else:  # 如果是第一次训练，直接使用交叉熵损失函数训练

                    loss = cls_loss

                print_loss = loss.data.item()

                optimizer.zero_grad()  # 优化器初始化
                loss.backward()
                optimizer.step()
                epoch += 1



                if epoch % 10000 == 0:
                    print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))



