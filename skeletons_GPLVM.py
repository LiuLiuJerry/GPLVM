#coding:utf-8
from __future__ import print_function
import gpflow
from gpflow import kernels
from gpflow import ekernels
#import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

# %matplotlib inline
import pods
import argparse
import h5py

import gen_horse
import ioUtil

parser = argparse.ArgumentParser()
parser.add_argument("--hdf5", default='Skeletons/horse_kpts.hdf5' )
parser.add_argument("--name", default='skeleton')


FLAGS = parser.parse_args()

def plot_skeleton(xyz, ax):
    ax.scatter(xyz[:,0], xyz[:,2], xyz[:,1])
    #key points
    key_pts = xyz[0:3,:]
    ax.plot(key_pts[:,0], key_pts[:,2], key_pts[:,1])
    #head
    head = xyz[3:4,:]
    head = np.row_stack([key_pts[2,:],head])
    print(head)
    ax.plot(head[:,0], head[:,2], head[:,1])

    #legs
    legf_l = xyz[4:7,:]
    legf_r = xyz[7:10,:]
    legb_l = xyz[10:13,:]
    legb_r = xyz[13:16,:]

    legf_l = np.row_stack([key_pts[1,:],legf_l])
    legf_r = np.row_stack([key_pts[1,:],legf_r])
    legb_l = np.row_stack([key_pts[0,:],legb_l]) 
    legb_r = np.row_stack([key_pts[0,:],legb_r])

    ax.plot(legf_l[:,0], legf_l[:,2], legf_l[:,1])
    ax.plot(legf_r[:,0], legf_r[:,2], legf_r[:,1]) 
    ax.plot(legb_l[:,0], legb_l[:,2], legb_l[:,1])   
    ax.plot(legb_r[:,0], legb_r[:,2], legb_r[:,1])  

    #tail
    tail = xyz[16:]
    tail = np.row_stack([key_pts[0,:],tail]);
    return ax.plot(tail[:,0], tail[:,2], tail[:,1])

def on_button_press(event):
    print(event.inaxes)
    global ax_chosen
    global ax_horse
    global ax_skeleton
    if event.inaxes == None:
        return 

    for i in range(3):
        if  np.in1d(event.inaxes, f_axes[i]): 
            ax_chosen.spines['top'].set_visible(False)
            ax_chosen.spines['bottom'].set_visible(False)
            print('choose axes')
            ax_chosen = f_axes[i][0]
            ax_skeleton = f_axes[i][1]
            ax_horse = f_axes[i][2]
            ax_chosen.spines['top'].set_visible(True)
            ax_chosen.spines['bottom'].set_visible(True)

            event.canvas.draw()
            return

    x = event.xdata
    y = event.ydata
    newX = [[x,y]]
    print('position: ', x, y)
    ax = event.inaxes
    ax.plot([x], [y],'r.',markersize=5)

    print(newX)
    mu_fFull, var_fFull = m.predict_f_full_cov(newX)
    newxyz = np.reshape(mu_fFull, (D, 3))
    # new shapes : get horses here
    nametosave = str(x)+'_'+str(y)+'.ply'
    horse_xyz = gen_horse.gen_horses(sess, model, newxyz, mustSavePly, nametosave)

    ax_skeleton.clear()
    ax_horse.clear()
    ax_skeleton.axis('off') 
    ax_horse.axis('off') 

    plot_skeleton(newxyz, ax_skeleton)
    ax_horse.scatter(horse_xyz[:,0], horse_xyz[:,2], horse_xyz[:,1], s = 1)

    event.canvas.draw()

np.random.seed(42)
gpflow.settings.numerics.quadrature = 'error'  # throw error if quadrature is used for kernel expectations

data = ioUtil.load_skeletons(FLAGS.hdf5, 'names')
Y = data.pointSet_in
print('Number of points * number of dimensions', Y.shape)

# create model
Q = 2 #pca降维的维度
M = 10 #支撑点的数目
N = Y.shape[0]
D = Y.shape[1]
Y = np.reshape(Y, (N, D*3))

# PCA降维，提取前面5维向量作为基 100*5
X_mean = gpflow.gplvm.PCA_reduce(Y, Q)

print('X_mean: ', X_mean.shape)

# GPLVM
# permutation:生成随机序列, 然后取前20, 20*5
# 所谓inducing points可能就是一些假设存在于潜在空间中的点吧
Z = np.random.permutation(X_mean.copy())[:M]
k = ekernels.Add([ekernels.RBF(3, ARD=False, active_dims=[0,1,2]) ,
         ekernels.Linear(3, ARD=False, active_dims=[3, 4, 5])  ])
#k = ekernels.RBF(5, ARD=False, active_dims=[0,1,2,3,4]) 

m = gpflow.gplvm.BayesianGPLVM(X_mean=X_mean, X_var=0.1*np.ones((N, Q)), Y=Y, kern=k, M=M, Z=Z)

linit = m.compute_log_likelihood()
m.optimize(maxiter=4)

print('X_mean by GPLVM: ', m.X_mean.value.shape)
#assert(m.compute_log_likelihood() > linit)


# test prediction 0-8 一共9个模型
testlen = 9;
Xtest1 = m.X_mean.value[0:testlen]
Xtest2 = m.X_mean.value[testlen:testlen*2]
Xtest = (Xtest1+Xtest2)/2
Xtest = [Xtest, Xtest1]
# mu_f, var_f = m.predict_f(Xtest1)
mu_fFull, var_fFull = m.predict_f_full_cov(Xtest1)

# save the predicted model
output = np.reshape(mu_fFull, (testlen, D, 3))
output_dir = 'output' 
outName = []
for i in range(testlen):
    outName.append(str(i)+'_'+str(i+testlen))
ioUtil.output_point_cloud_ply(output, outName, output_dir, str(D))

## figure 
fig = plt.figure(figsize=(10,5))
plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
grid = gridspec.GridSpec(3, 5, wspace=0.05)

main_ax = fig.add_subplot(grid[:, 0:3])
f0, = main_ax.plot(m.X_mean.value[:,0], m.X_mean.value[:,1],'b.',markersize=5)
fig.canvas.mpl_connect('button_press_event', on_button_press)
main_ax.set_title('Bayesian GPLVM')
main_ax.axis('on')
# 骨架
f_axes = []
for i in range(3):
    side_ax = fig.add_subplot(grid[i, 3:5])
    side_ax.set_xticks([])  
    side_ax.set_yticks([]) 
    for sp in side_ax.spines.values():
        sp.set_color('c')
        sp.set_linewidth(5)
        sp.set_visible(False) 
 
    grid_side = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=grid[i,3:5], wspace=0.1, hspace=0.1)

    fj = [side_ax]
    f_axesj = [side_ax.axes]
    for j in range(2):        
        side_ax1 = fig.add_subplot(grid_side[0, j], projection='3d')

        side_ax1.view_init(elev=10, azim=25)
        side_ax1.axis('equal')
        side_ax1.set_alpha(1)
        side_ax1.set_facecolor('none')

        side_ax1.set_xticks([])  
        side_ax1.set_yticks([])    
        side_ax1.axis('off')  

        fi, = plot_skeleton(output[i,:],side_ax1)

        f_axesj.append(side_ax1)
   

    if f_axes == []:
        f_axes = [f_axesj]
    else:
        f_axes = np.vstack((f_axes, f_axesj))
     
ax_chosen = f_axes[0][0]
ax_skeleton = f_axes[0][1]
ax_horse = f_axes[0][2]
ax_chosen.spines['top'].set_visible(True)
ax_chosen.spines['bottom'].set_visible(True)

## deep learning things
modelPath = 'myNet/trained_models/'
mustSavePly = True

with tf.Graph().as_default():
    model = gen_horse.load_model()
    # check point: 二进制文件，它包含的权重变量，biases变量和其他变量
    sess = tf.Session()
        
    #metaPath = modelPath + 'epoch_200.ckpt.meta'
    ckptPath = tf.train.latest_checkpoint(modelPath)
    print('load checkpoint: ' + ckptPath)
    saver = tf.train.Saver( max_to_keep=5)
    #saver = tf.train.import_meta_graph(metaPath)
    saver.restore(sess, ckptPath )
 

plt.show()


print('finished')




