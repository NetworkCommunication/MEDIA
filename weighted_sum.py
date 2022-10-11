import numpy as np
import scipy.io as sio
import random_
import torch

def Q_value_x(tasks, decisions):



    re = 200 / 8
    rc = 100 / 8
    Tl = 0
    Te = 0
    Tc = 0
    El = 0
    Ee = 0
    Ec = 0
    num = 1
    for task, decision in zip(tasks, decisions):
        if num % 2 == 1:
            task_1 = task
            decision_1 = decision
        if num % 2 == 0:
            task_2 = task * 50 * 10**6
            decision_2 = decision
            # 开始计算
            if decision_1 == 1:
                # 本地
                Tl = Tl + task_2 / (150 * 10**6)
                El = El + task_2 * 6 / (100 * 10**6)

            else:
                if decision_1 == 0 and decision_2 == 1:
                    # 边缘
                    Te = Te + task_2/(1000 * 10**6) + task_1/re
                    Ee = Ee + task_2 * 3 / (100 * 10**6) + task_1 * 5
                if decision_1 == 0 and decision_2 == 0:
                    # 中心
                    Tc = Tc + task_2 / (2000 * 10**6) + task_1 / re + task_1 / rc
                    Ec = Ec + task_2 * 2 / (100 * 10**6) + task_1 * 5 + task_1 * 10

        num = num + 1


    T = max(Tl, Te + Tc)
    E = (El + Ee + Ec)/50
    Q=0.5*T + 0.5*E
    return Q

def per_q(X_test, Y_test):
    num=1
    per=[]

    for x,y in zip( X_test,Y_test ):
        if num%100==1 :
            q=Q_value_x(x,y)
        elif num%100==0:
            q=q/100

            per.append(q)
        elif num>1:
            q= q + Q_value_x(x,y)
        num = num+1
    num=1
    for a in per:
        if num ==1:
            per1=a
        elif num >1:
            per1=np.vstack((per1, a))
        num=num+1

    return per1
