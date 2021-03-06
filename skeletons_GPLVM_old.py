#coding:utf-8
from __future__ import print_function
import gpflow
#from gpflow import kernels
from gpflow import ekernels
#import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from scipy.spatial.distance import cdist, squareform
from matplotlib.widgets import Button
from matplotlib import colors as mcolors

# %matplotlib inline
import pods
import argparse
import h5py
import threading

import gen_horse2horse
import ioUtil
import floyd
import bspline
import navigate
import thread_Generate

parser = argparse.ArgumentParser()
parser.add_argument("--hdf5", default='Skeletons/horse.hdf5' )
parser.add_argument("--name", default='skeleton')

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

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
    global preX
    global floyd_res
    global embeds
    global main_ax
    global bool_drawing
    global pointSet
    global Nav
    global key_list

    n_sample = 5
    if event.inaxes==None or event.inaxes==axgen or event.inaxes==axnav or event.inaxes==axloc:
        return 

    for i in range(1): 
        if  np.in1d(event.inaxes, f_axes[i]): 
            print('\n>>>>>>>>>>>> choose drawing window...')
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
    print('\t position: ', x, y)
    newX = np.array([x,y]).reshape([-1,2])
    
    ax = event.inaxes
    ax.plot([x], [y],'r.',markersize=5)
    event.canvas.draw()
    if bool_drawing:
        dist = floyd_res[0]
        parent = np.array(floyd_res[1], dtype=int)
        
        pStart, pEnd = floyd.cal_shortestpath(preX, newX, embeds, dist)
        print('\t pStart',pStart)
        print('\t pEnd',pEnd)
        path_idx = floyd.obtainPath(pStart, pEnd, parent)

        path_idx = np.concatenate(([pStart],path_idx))
        path = embeds[path_idx,:]      

        print('\t path index',path_idx)

        path=np.concatenate((preX,path),axis=0)
        path=np.concatenate((path,newX),axis=0)
        #print('>>>>>>>>>>>>>path',path)

        spline = bspline.bspline(path, n=50, degree=3)
        main_ax.plot(spline[:,0], spline[:,1], 'r.')
        main_ax.plot(path[:,0], path[:,1], 'grey') #floyd
        #key_idx = np.arange(0,5)*10
        key_idx = navigate.resample(spline,5)
        key_idx = np.append(key_idx,[49],axis=0)
        key_list = spline[key_idx,:]
        main_ax.scatter(key_list[:,0], key_list[:,1], s=50, c='red')
        event.canvas.draw()

        bool_drawing = False
    else:
        key_list = []
        preX = newX
        bool_drawing = True
        nav_init = []

    print('\t newX choosed', newX)

    event.canvas.draw()

    var = Nav.cal_variance(newX)
    var = np.array(var)
    print('\t var of the position: ', var[0,0,0])


class Index(object):
    ind = 0

    def generate(self, event):

        global key_list
        global preX
        global embeds
        global main_ax
        global bool_drawing
        global pointSet
        global Nav

        if Nav == None:
            print('\n>>>>>>>>>>>> prepare for navigating...')
            Nav = navigate.Navigator(embeds, m)


        if len(key_list) > 0:
            print('\n>>>>>>>>>>>> generating shapes on a path...')
            
            mu_fFull, var_fFull = m.predict_f_full_cov(key_list) 
            nametosave = []
            for xx in key_list:
                nametosave.append(str(xx[0])[:4]+'_'+str(xx[1])[:4])
            thread_Generate.generateModels(mu_fFull, key_list, embeds, pointSet, sess, model, nametosave)
            key_list = []

        else:
            print('\n>>>>>>>>>>>> generating a single shape...')
            newX = preX
            mu_fFull, var_fFull = m.predict_f_full_cov(newX)
            newxyz = np.reshape(mu_fFull, (1, D, 3)) #新的骨架
            # new shapes : get horses here
            nametosave = str(newX[:,0])[:4]+'_'+str(newX[:,1])[:4]
            neighbor_horse = thread_Generate.find_neighbor(newX, embeds, pointSet)
            neighbor_horse = np.reshape(neighbor_horse, [1, 2048, 3])
            horse_xyz = gen_horse2horse.gen_horses(sess, model, newxyz, neighbor_horse, mustSavePly, nametosave)

            ax_skeleton.clear()
            ax_horse.clear()
            ax_skeleton.axis('off') 
            ax_horse.axis('off') 

            plot_skeleton(newxyz[0], ax_skeleton)
            ax_horse.scatter(horse_xyz[:,0], horse_xyz[:,2], horse_xyz[:,1], s = 1)

        plt.draw()
        bool_drawing = False

    def navigate(self, event):
        global Nav
        global main_ax
        global key_list
        global nav_init

        n_nav = 1

        
        if Nav == None:
            print('\n>>>>>>>>>>>> prepare for navigating...')
            Nav = navigate.Navigator(embeds, m)

        if len(nav_init) == 0:
            print('\n>>>>>>>>>>>> set initial point...')
            nav_init = preX
            Nav.navigate_from(nav_init, n_nav)
            bool_drawing = False
        print('\n>>>>>>>>>>>> generating neighbors...')

        '''res = Nav.optimize(method = tf.train.GradientDescentOptimizer(0.001))
        p = res['x'].reshape([-1,2])
        fetches = res['fun']
        print('\t optimize result\n', res)'''
        pnew, W, Wdir = Nav.navigate()
        
        p = np.append(nav_init.reshape([-1,2]), pnew.reshape([-1,2]), axis = 0)
        print('\t new point:', p)
        main_ax.plot(p[:,0], p[:,1], 'r*-',markersize=5)
        dir1 = np.append(nav_init.reshape([-1,2]), (nav_init+W).reshape([-1,2]), axis = 0)
        dir2 = np.append(nav_init.reshape([-1,2]), (nav_init+Wdir).reshape([-1,2]), axis = 0)
        print('\t directions', W,Wdir)
        main_ax.plot(dir1[:,0], dir1[:,1], color=colors['pink'], markersize=5)
        main_ax.plot(dir2[:,0], dir2[:,1], color=colors['lightblue'], markersize=5)
        plt.draw()
        

        nav_init = pnew

    def locate(self, event):
        global embeds
        global skeletons
        global dist
        global m
        global main_ax
        global nav_init
        global ske

        print('\n>>>>>>>>>>>> locating skeletons...')

        locator = navigate.Locator(embeds, skeletons, dist, m)
        
        locator.locate_skeletons(ske)
        #final_X = Nav.optimize(method = tf.train.GradientDescentOptimizer(0.02))
        results = locator.optimize(method = tf.train.GradientDescentOptimizer(0.001))
        
        final_X = results['x']
        #xx_ske = embeds[[n1,n2],:]
        #print('\t two initial position', xx_ske)
        print('\t new position located', final_X)
        print('\t locate results:\n', results)
        #main_ax.scatter(xx_ske[:,0], xx_ske[:,1], s=50, alpha=0.3)
        main_ax.scatter(final_X[0], final_X[1], s=30, alpha=0.3)
        plt.draw()

        nav_init = np.array(final_X).reshape([-1,2])

    def clear(self, event):
        global embeds
        global skeletons
        global dist
        global m
        global main_ax
        global nav_init
        global ske

        print('\n>>>>>>>>>>>> locating skeletons...')

        locator = navigate.Locator(embeds, skeletons, dist, m)
        
        locator.locate_skeletons(ske)
        #final_X = Nav.optimize(method = tf.train.GradientDescentOptimizer(0.02))
        results = locator.optimize(method = tf.train.GradientDescentOptimizer(0.001))
        
        final_X = results['x']
        #xx_ske = embeds[[n1,n2],:]
        #print('\t two initial position', xx_ske)
        print('\t new position located', final_X)
        print('\t locate results:\n', results)
        #main_ax.scatter(xx_ske[:,0], xx_ske[:,1], s=50, alpha=0.3)
        main_ax.scatter(final_X[0], final_X[1], s=30, alpha=0.3)
        plt.draw()

        nav_init = np.array(final_X).reshape([-1,2])

   
        
        

np.random.seed(42)  #用于指定随机数生成时所用算法开始的整数值
gpflow.settings.numerics.quadrature = 'error'  # throw error if quadrature is used for kernel expectations

if __name__ == '__main__':
    data = ioUtil.load_examples(FLAGS.hdf5, 'names')
    skeletons = data.skeleton_in
    pointSet = data.pointSet_out
    print('Number of points * number of dimensions', skeletons.shape)

    # create model
    Q = 2 #pca降维的维度
    M = 10 #支撑点的数目
    N = skeletons.shape[0]
    D = skeletons.shape[1]
    Y = np.reshape(skeletons, (N, D*3))
    pointSet = np.reshape(pointSet, (N, 2048, 3))
    # PCA降维，提取前面5维向量作为基 100*5
    X_mean = gpflow.gplvm.PCA_reduce(Y, Q)

    # GPLVM
    # permutation:生成随机序列, 然后取前20, 20*5
    # 所谓inducing points可能就是一些假设存在于潜在空间中的点吧
    Z = np.random.permutation(X_mean.copy())[:M]
    k = ekernels.Add([ekernels.RBF(3, ARD=False, active_dims=[0,1,2]) ,
             ekernels.Linear(3, ARD=False, active_dims=[3, 4, 5])  ])
    X_var = 0.1*np.ones((N, Q))
    #k = ekernels.RBF(5, ARD=False, active_dims=[0,1,2,3,4]) 
    print('\n>>>>>>>>>>>> computing BayesianGPLVM...')
    m = gpflow.gplvm.BayesianGPLVM(X_mean=X_mean, X_var=X_var, Y=Y, kern=k, M=M, Z=Z)
    linit = m.compute_log_likelihood()
    #m.optimize(method = tf.train.GradientDescentOptimizer(0.02))
    m.optimize()

    embeds = np.array(m.X_mean.value)

    print('\n>>>>>>>>>>>> computing main direction by pca...')
    main_dir,W =  navigate.pca(embeds, 0.8)
    min_idx = np.argmin(main_dir)
    max_idx = np.argmax(main_dir)
    main_line = np.array([embeds[min_idx,:], embeds[max_idx,:]])

    #print('X_mean by GPLVM: ', m.X_mean.value.shape)
    #assert(m.compute_log_likelihood() > linit)



    print('\n>>>>>>>>>>>> computing shortest path...')
    floyd_res = floyd.calculate_floyd(embeds)
    dist = floyd_res[0]
    parent = np.array(floyd_res[1], dtype=int)
    edges = np.array(floyd_res[2])
    print('\t size of edges', edges.shape)


    ######################################################  figure  ##############
    print('\n>>>>>>>>>>>> prepare plot inits...')
    bool_drawing = False
    key_spl = []
    nav_init = []
    Nav = Nav = navigate.Navigator(embeds, m)

    print('\n>>>>>>>>>>>> ploting...')
    fig = plt.figure(facecolor='grey', figsize=(10,5))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    grid = gridspec.GridSpec(3, 5, wspace=0.05)

    ## 主图
    main_ax = fig.add_subplot(grid[:, 0:3])
    main_ax.patch.set_facecolor(colors['silver'])
    main_ax.xaxis.set_ticks_position('bottom') 
    main_ax.spines['bottom'].set_position(('data', 0)) 
    main_ax.yaxis.set_ticks_position('left') 
    main_ax.spines['left'].set_position(('data', 0))

    main_ax.scatter(embeds[:,0], embeds[:,1], s=5) #floyd

    '''for i in range(len(edges[0])):
        e = [edges[0][i], edges[1][i]]
        #print('>>>>>>>>>>>>> e', e)
        edge = embeds[e,:]      
        main_ax.plot(edge[:,0], edge[:,1], color='lightblue')'''

    fig.canvas.mpl_connect('button_press_event', on_button_press)
    main_ax.set_title('Bayesian GPLVM')
    main_ax.axis('on')


    callback = Index()
    axgen = plt.axes([0.6, 0.03, 0.09, 0.045])
    axloc = plt.axes([0.7, 0.03, 0.09, 0.045])
    axnav = plt.axes([0.8, 0.03, 0.09, 0.045])
    bgen = Button(axgen, 'generate')
    bgen.on_clicked(callback.generate)  
    bloc = Button(axloc, 'locate')
    bloc.on_clicked(callback.locate)
    bnav = Button(axnav, 'navigate')
    bnav.on_clicked(callback.navigate)

    ## 侧面的图
    f_axes = []
    for i in range(1):
        side_ax = fig.add_subplot(grid[i, 3:5])
        side_ax.set_xticks([])  
        side_ax.set_yticks([]) 
        for sp in side_ax.spines.values():
            sp.set_color(colors['dimgrey'])
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

            plot_skeleton(skeletons[i,...], side_ax1)

            f_axesj.append(side_ax1)

        horse_xyz = pointSet[i,...]
        f_axesj[2].clear()
        if i != 2:
            f_axesj[2].scatter(horse_xyz[:,0], horse_xyz[:,2], horse_xyz[:,1], s = 1)

        if f_axes == []:
            f_axes = [f_axesj]
        else:
            f_axes = np.vstack((f_axes, f_axesj))

    n1= 27
    n2 = 26
    ske = (skeletons[n1,:]+skeletons[n2,:])/2
    '''ax = f_axes[2][1]
    f_axes[2][2].clear()
    ax.clear()
    plot_skeleton(ske, ax)'''
         
    ax_chosen = f_axes[0][0]
    ax_skeleton = f_axes[0][1]
    ax_horse = f_axes[0][2]
    ax_chosen.spines['top'].set_visible(True)
    ax_chosen.spines['bottom'].set_visible(True)

    ## deep learning things
    modelPath = 'myNet/trained_models/horse2horse'
    mustSavePly = True

    with tf.Graph().as_default():
        model = gen_horse2horse.load_model()
        # check point: 二进制文件，它包含的权重变量，biases变量和其他变量
        config = tf.ConfigProto(allow_soft_placement = True)
        sess = tf.Session(config = config)
            
        #metaPath = modelPath + 'epoch_200.ckpt.meta'
        ckptPath = tf.train.latest_checkpoint(modelPath)
        #print('load checkpoint: ' + ckptPath)
        saver = tf.train.Saver( max_to_keep=5)
        #saver = tf.train.import_meta_graph(metaPath)
        saver.restore(sess, ckptPath )
     

    plt.show()


    print('finished')




