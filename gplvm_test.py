#coding:utf-8
from __future__ import print_function
import gpflow
from gpflow import kernels
from gpflow import ekernels
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# %matplotlib inline
import pods
pods.datasets.overide_manual_autherorize = True # dont ask to authorize
np.random.seed(42)
gpflow.settings.numerics.quadrature = 'error'  # throw error if quadrature is used for kernel expectations

data = pods.datasets.oil_100()
Y = data['X']
print('Number of points * number of dimensions', Y.shape)
data['citation']


# create model
Q = 6
M = 20
N = Y.shape[0]
# PCA降维，提取前面5维向量作为基 100*5
X_mean = gpflow.gplvm.PCA_reduce(Y, Q)

print('X_mean: ', X_mean.shape)
# permutation:生成随机序列, 然后取前20, 20*5
# 所谓inducing points可能就是一些假设存在于潜在空间中的点吧
Z = np.random.permutation(X_mean.copy())[:M]

print('Z: ', Z.shape)

#slice(0,3): 截取序号0,1,2,共3个元素的切片
#slice(3,5): 截取序号3,4,共2个元素的切片，一共也是5维
k = ekernels.Add([ekernels.RBF(3, ARD=False, active_dims=[0,1,2]) ,
         ekernels.Linear(3, ARD=False, active_dims=[3, 4, 5])  ])
#k = ekernels.RBF(5, ARD=False, active_dims=[0,1,2,3,4]) 

m = gpflow.gplvm.BayesianGPLVM(X_mean=X_mean, X_var=0.1*np.ones((N, Q)), Y=Y, kern=k, M=M, Z=Z)

linit = m.compute_log_likelihood()
m.optimize(maxiter=4)

assert(m.compute_log_likelihood() > linit)

XPCAplot = gpflow.gplvm.PCA_reduce(data['X'], 2)
f, ax = plt.subplots(1,2, figsize=(10,6))
labels=data['Y'].argmax(axis=1)
colors = cm.rainbow(np.linspace(0, 1, len(np.unique(labels))))

for i, c in zip(np.unique(labels), colors):
    ax[0].scatter(XPCAplot[labels==i,0], XPCAplot[labels==i,1], color=c, label=i)
    ax[0].set_title('PCA')
    ax[1].scatter(m.X_mean.value[labels==i,1], m.X_mean.value[labels==i,2], color=c, label=i)
    ax[1].set_title('Bayesian GPLVM')
    
plt.show()

print('finished')


