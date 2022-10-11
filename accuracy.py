import numpy as np
import scipy.io as sio
import random
import torch

def per_acu(Y_label, Y_test):
    num=1
    acu=[]

    for label,test in zip( Y_label,Y_test ):
        a=0
        for l, t in zip(label, test):
            if l==t:
                a=a+1
        a=a/len(label)
        if num%100==1 :
            per_a=a
        elif num%100==0:
            per_a=per_a/100

            acu.append(per_a)
        elif num>1:
            per_a= per_a + a
        num = num+1
    num=1
    for a_ in acu:
        if num ==1:
            acu1=a_
        elif num >1:
            acu1=np.vstack((acu1, a_))
        num=num+1

    return acu1

