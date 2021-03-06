#coding:utf-8
from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy

from scipy.spatial.distance import cdist, squareform
import sys

import gpflow
from gpflow import kernels
from gpflow import ekernels
from gpflow.param import Parameterized, AutoFlow, DataHolder
from gpflow.param import Param
from gpflow.model import Model
from gpflow._settings import settings
from gpflow import session as session_mngr
from gpflow.mean_functions import Zero
from gpflow import likelihoods

float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64


class Locator(Model):
    def __init__(self, embeds, skeletons, dist, m, name='Model'):
        Model.__init__(self, name)
        self.skeletons = skeletons
        self.dist_embeds = dist
        self.nkpts = len(skeletons[0,:])
        self.npts = len(embeds)
        self.dist_skeletons = np.ones([self.npts,self.npts])*-1
        embeds = np.array(m.X_mean.value)
        self.X_mean = Param(embeds)
        self.Z = Param(np.array(m.Z.value))
        self.kern = deepcopy(m.kern)
        self.X_var = Param(np.array(m.X_var.value))
        self.Y = m.Y
        self.likelihood = likelihoods.Gaussian()
        self.mean_function = Zero()

        self.likelihood._check_targets(self.Y.value)
        self._session = None

        self.X_mean.fixed = True
        self.Z.fixed = True
        self.kern.fixed = True
        self.X_var.fixed = True
        self.likelihood.fixed = True


    def locate_skeletons(self, new_ske):
        embeds = self.X_mean.value
        skeletons = self.skeletons
        new_ske = new_ske.reshape([-1, self.nkpts, 3])       

        def cal_distSkeletons(shapesA, shapesB):
            dist = np.sqrt( ((shapesB-shapesA)**2).sum(axis=-1))
            dist = dist.sum(axis = -1)
            return dist
        dist = cal_distSkeletons(new_ske,skeletons) 
        dist = dist.reshape([-1])
        idx = np.argsort(dist)
        pStart = embeds[idx[0],:]
        pStart = pStart.reshape([1,2])

        self.Xnew = Param(pStart)
        self.new_ske = Param(new_ske)
        self.new_ske.fixed = True


    def _locate_loss(self, full_cov=True):
        num_inducing = tf.shape(self.Z)[0]
        psi1 = self.kern.eKxz(self.Z, self.X_mean, self.X_var)
        psi2 = tf.reduce_sum(self.kern.eKzxKxz(self.Z, self.X_mean, self.X_var), 0)
        Kuu = self.kern.K(self.Z) + tf.eye(num_inducing, dtype=float_type) * settings.numerics.jitter_level
        Kus = self.kern.K(self.Z, self.Xnew)
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)
        L = tf.cholesky(Kuu)

        A = tf.matrix_triangular_solve(L, tf.transpose(psi1), lower=True) / sigma
        tmp = tf.matrix_triangular_solve(L, psi2, lower=True)
        AAT = tf.matrix_triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=float_type)
        LB = tf.cholesky(B)
        c = tf.matrix_triangular_solve(LB, tf.matmul(A, self.Y), lower=True) / sigma
        tmp1 = tf.matrix_triangular_solve(L, Kus, lower=True)
        tmp2 = tf.matrix_triangular_solve(LB, tmp1, lower=True)
        mean = tf.matmul(tmp2, c, transpose_a=True)
        if full_cov:
            var = self.kern.K(self.Xnew) + tf.matmul(tmp2, tmp2, transpose_a=True) \
                  - tf.matmul(tmp1, tmp1, transpose_a=True)
            shape = tf.stack([1, 1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 2), shape)
        else:
            var = self.kern.Kdiag(self.Xnew) + tf.reduce_sum(tf.square(tmp2), 0) \
                  - tf.reduce_sum(tf.square(tmp1), 0)
            shape = tf.stack([1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 1), shape)
        tmp_ske = mean + self.mean_function(self.Xnew)
        tmp_ske = tf.reshape(tmp_ske, [1,-1,3])
        def cal_loss(new_ske, tmp_ske):  
            loss = tf.reduce_sum(tf.square(tmp_ske-new_ske), axis=-1)
            #loss = tf.sqrt(loss)
            return tf.reduce_sum(loss)

        loss = cal_loss(self.new_ske, tmp_ske)

        return loss

    def compile(self, session=None, graph=None, optimizer=None):

        g = tf.Graph()
        session = tf.Session(graph=g)
        with g.as_default():
            self._free_vars = tf.Variable(self.get_free_state())
            '''for p in self.sorted_params:
                print('>>>>>>>>>>>>>>> p', p)'''
            self.make_tf_array(self._free_vars)
            with self.tf_mode():
                f = self._locate_loss()
                g = tf.gradients(f, self._free_vars)[0]

            self._minusF = tf.identity(f, name='objective')
            self._minusG = tf.identity(g, name='grad_objective')

            # The optimiser needs to be part of the computational graph,
            # and needs to be initialised before tf.initialise_all_variables()
            # is called.
            if optimizer is None:
                opt_step = None
            else:
                opt_step = optimizer.minimize(
                    self._minusF, var_list=[self._free_vars])
            init = tf.global_variables_initializer()

        session.run(init)
        self._session = session

        # build tensorflow functions for computing the likelihood
        if settings.verbosity.tf_compile_verb:
            print("compiling tensorflow function...")
        sys.stdout.flush()

        self._feed_dict_keys = self.get_feed_dict_keys()

        def obj(x): #根据优化的结果计算函数值和梯度
            self.num_fevals += 1
            feed_dict = {self._free_vars: x}
            self.update_feed_dict(self._feed_dict_keys, feed_dict)
            f, g = self.session.run([self._minusF, self._minusG],
                                     feed_dict=feed_dict)
            return f.astype(np.float64), g.astype(np.float64)
            

        self._objective = obj
        if settings.verbosity.tf_compile_verb:
            print("done")
        sys.stdout.flush()
        self._needs_recompile = False

        return opt_step


class Navigator(Model):
    def __init__(self, embeds, m, name='Model'):
        Model.__init__(self, name)

        self.nkpts = len(m.Y.value[0])
        self.npts = len(embeds)
        embeds = np.array(m.X_mean.value)
        self.X_mean = Param(embeds)
        self.Z = Param(np.array(m.Z.value))
        self.kern = deepcopy(m.kern)
        self.X_var = Param(np.array(m.X_var.value))
        self.Y = m.Y
        self.likelihood = likelihoods.Gaussian()
        self.mean_function = Zero()

        self.likelihood._check_targets(self.Y.value)
        self._session = None

        self.X_mean.fixed = True
        self.Z.fixed = True
        self.kern.fixed = True
        self.X_var.fixed = True
        self.likelihood.fixed = True


    @AutoFlow((float_type, [None, None]))
    def cal_variance(self, Xnew):
        self.Pnew = Xnew

        full_cov=True
        #print('\t Pnew:', self.Pnew)
        #print('\t Z:', self.Z)

        num_inducing = tf.shape(self.Z)[0]
        psi1 = self.kern.eKxz(self.Z, self.X_mean, self.X_var)
        psi2 = tf.reduce_sum(self.kern.eKzxKxz(self.Z, self.X_mean, self.X_var), 0)
        Kuu = self.kern.K(self.Z) + tf.eye(num_inducing, dtype=float_type) * settings.numerics.jitter_level
        Kus = self.kern.K(self.Z, self.Pnew)
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)
        L = tf.cholesky(Kuu)

        A = tf.matrix_triangular_solve(L, tf.transpose(psi1), lower=True) / sigma
        tmp = tf.matrix_triangular_solve(L, psi2, lower=True)
        AAT = tf.matrix_triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=float_type)
        LB = tf.cholesky(B)
        c = tf.matrix_triangular_solve(LB, tf.matmul(A, self.Y), lower=True) / sigma
        tmp1 = tf.matrix_triangular_solve(L, Kus, lower=True)
        tmp2 = tf.matrix_triangular_solve(LB, tmp1, lower=True)
        mean = tf.matmul(tmp2, c, transpose_a=True)
        if full_cov:
            var = self.kern.K(self.Pnew) + tf.matmul(tmp2, tmp2, transpose_a=True) \
                  - tf.matmul(tmp1, tmp1, transpose_a=True)
            shape = tf.stack([1, 1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 2), shape)
        else:
            var = self.kern.Kdiag(self.Pnew) + tf.reduce_sum(tf.square(tmp2), 0) \
                  - tf.reduce_sum(tf.square(tmp1), 0)
            shape = tf.stack([1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 1), shape)

        return var

    def compile(self, session=None, graph=None, optimizer=None):

        g = tf.Graph()
        session = tf.Session(graph=g)
        with g.as_default():
            self._free_vars = tf.Variable(self.get_free_state())
            '''for p in self.sorted_params:
                print('>>>>>>>>>>>>>>> p', p)'''
            self.make_tf_array(self._free_vars)
            with self.tf_mode():
                f = self._navigate_loss()
                g = tf.gradients(f, self._free_vars)[0]

            self._minusF = tf.identity(f, name='objective')
            self._minusG = tf.identity(g, name='grad_objective')

            # The optimiser needs to be part of the computational graph,
            # and needs to be initialised before tf.initialise_all_variables()
            # is called.
            if optimizer is None:
                opt_step = None
            else:
                opt_step = optimizer.minimize(
                    self._minusF, var_list=[self._free_vars])
            init = tf.global_variables_initializer()

        session.run(init)
        self._session = session

        # build tensorflow functions for computing the likelihood
        if settings.verbosity.tf_compile_verb:
            print("compiling tensorflow function...")
        sys.stdout.flush()

        self._feed_dict_keys = self.get_feed_dict_keys()

        def obj(x): #根据优化的结果计算函数值和梯度
            self.num_fevals += 1
            feed_dict = {self._free_vars: x}
            self.update_feed_dict(self._feed_dict_keys, feed_dict)

            f, g = self.session.run([self._minusF, self._minusG],
                                     feed_dict=feed_dict)
            return f.astype(np.float64), g.astype(np.float64)

        self._objective = obj
        if settings.verbosity.tf_compile_verb:
            print("done")
        sys.stdout.flush()
        self._needs_recompile = False

        return opt_step


    def navigate_from(self, Pnew, n, r=0.1):
        self.r2 = r*r
        self.P_path = np.array([Pnew]).reshape([-1,2])
        Pnew = Pnew.reshape([1,2])
        self.Pnew = Param(np.tile(Pnew, (n,1)) )
        self.Pinit = Param(np.array(Pnew))
        self.Pinit.fixed = True  
        self.n = n 

    def navigate(self, delta=0.1):
        def _find_neighbors(P, n=8):
            embeds = self.X_mean.value
            dist = np.sum(np.square(P-embeds),axis=-1)
            dist = np.sqrt(dist)
            idx = np.argsort(dist)  
            nei = embeds[idx[:n+1], :]
            return nei, dist[idx[:n+1]]

        n = 8
        Pnew = self.Pnew.value
        X, dist = _find_neighbors(Pnew, n)
        print('\t dist', dist)
        A = 0.04
        B = 2
        dist = dist.reshape([-1,1])
        F = A/(np.power(dist, 1.2)) - B/np.power(dist,0.2)
        F = np.minimum(F, 0.1)
        pdir = Pnew-X
        pdir = pdir/dist
        t = delta
        xx = 0.5*t*t*F
        xx = np.sum(pdir*xx, axis=0)

        p, W = pca(X, n=1)
        W = W.reshape([2])
        W = W/np.sqrt(np.sum(W*W))
        if len(self.P_path) > 1  and W.dot(self.P_path[-1,:]-self.P_path[-2,:]) < 0:
            W = -1*W
        Pnew = self.Pnew.value + W*delta + xx
        self.Pnew = Param(Pnew)
        self.P_path = np.append(self.P_path, Pnew, axis=0)

        return Pnew, W*0.1, xx

       

    def _navigate_loss(self):
        alpha = 1
        beta = 0  
        gamma = 50

        full_cov=True  

        num_inducing = tf.shape(self.Z)[0]
        psi1 = self.kern.eKxz(self.Z, self.X_mean, self.X_var)
        psi2 = tf.reduce_sum(self.kern.eKzxKxz(self.Z, self.X_mean, self.X_var), 0)
        Kuu = self.kern.K(self.Z) + tf.eye(num_inducing, dtype=float_type) * settings.numerics.jitter_level
        Kus = self.kern.K(self.Z, self.Pnew)
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)
        L = tf.cholesky(Kuu)

        A = tf.matrix_triangular_solve(L, tf.transpose(psi1), lower=True) / sigma
        tmp = tf.matrix_triangular_solve(L, psi2, lower=True)
        AAT = tf.matrix_triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=float_type)
        LB = tf.cholesky(B)
        c = tf.matrix_triangular_solve(LB, tf.matmul(A, self.Y), lower=True) / sigma
        tmp1 = tf.matrix_triangular_solve(L, Kus, lower=True)
        tmp2 = tf.matrix_triangular_solve(LB, tmp1, lower=True)
        mean = tf.matmul(tmp2, c, transpose_a=True)
        if full_cov:
            var = self.kern.K(self.Pnew) + tf.matmul(tmp2, tmp2, transpose_a=True) \
                  - tf.matmul(tmp1, tmp1, transpose_a=True)
            shape = tf.stack([1, 1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 2), shape)
        else:
            var = self.kern.Kdiag(self.Pnew) + tf.reduce_sum(tf.square(tmp2), 0) \
                  - tf.reduce_sum(tf.square(tmp1), 0)
            shape = tf.stack([1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 1), shape)
    
        var_loss = tf.reduce_mean(var)

        dist2 = tf.reduce_sum(tf.square(self.Pnew-self.Pinit))
        dist_loss = tf.maximum(dist2-self.r2, tf.constant(0.0, dtype=tf.float64))

        def pairwise_l2_norm2_batch(x, y, scope=None):
            nump_x = tf.shape(x)[0]
            nump_y = tf.shape(y)[0]

            xx = tf.expand_dims(x, -1)
            xx = tf.tile(xx, tf.stack([1, 1, nump_y]))

            yy = tf.expand_dims(y, -1)
            yy = tf.tile(yy, tf.stack([1, 1, nump_x]))
            yy = tf.transpose(yy, perm=[2, 1, 0])

            diff = tf.subtract(xx, yy)
            square_diff = tf.square(diff)

            square_dist = tf.reduce_sum(square_diff, 1)

            return square_dist

        pts = tf.concat([self.Pnew, tf.reshape(self.Pinit,[1,2]) ], axis=0)
        square_dist = pairwise_l2_norm2_batch(pts, pts)
        
        nnk = tf.nn.top_k(tf.negative(square_dist), k=2)
        print(nnk.values)
        minRow = nnk.values[:,1]
        min_dist = self.r2 + minRow

        '''minRow = tf.reduce_mean(square_dist, axis = -1)
        min_dist = self.r2-minRow'''
        dist_nbs_loss = tf.reduce_mean(tf.maximum(min_dist, tf.constant(0.0, dtype=tf.float64)) )

        loss = gamma*dist_nbs_loss + alpha*var_loss + beta*dist_loss

        return loss

    def local_pca(self, delta=0.1):
        def _find_neighbors(P, n=8):
            embeds = self.X_mean.value
            dist = np.sum(np.square(P-embeds),axis=-1)
            idx = np.argsort(dist)[::-1]      
            nei = embeds[idx[:n+1], :]
            return nei

        X = _find_neighbors(self.Pnew.value, 8)
        p, W = pca(X)
        W = W/np.sqrt(np.sum(W*W))
        Pnew = self.Pnew.value + W*delta
        self.Pnew = Param(Pnew)
        return Pnew

def pca(X, r=1.0, n=0.0):
    evecs, evals = np.linalg.eigh(np.cov(X.T))
    print('\t evecs' ,evecs.shape)
    print('\t evals' ,evals.shape)
    i = np.argsort(evecs)[::-1]
    print('\t i' ,i)
    evecs = evecs[i]
    sum_evecs = np.sum(evecs)
    cumsum = np.cumsum(evecs)
    eve_sum = cumsum/sum_evecs
    print('\t evecs' ,evecs)
    print('\t eve_sum' ,eve_sum)
    if n > 0:
        W = evals[:,i]
        W = W[:,:n]
        return (X - X.mean(0)).dot(W), W
    idx = np.argmin(eve_sum-r)
    print('\t idx' ,idx)
    W = evals[:, i]
    W = W[:, :idx+1]
    return (X - X.mean(0)).dot(W), W


def resample(pos, n):
    dist = np.sum(np.square(pos[:-1,:]-pos[1:,:]), axis=1)
    dist = np.sqrt(dist)
    dist = np.concatenate(([0], dist), axis=0)
        
    cumsum_dist = np.cumsum(dist)
    length = cumsum_dist[-1]
    len_dist = length/(n+1)

    resample_len = np.arange(n+1)*len_dist
    resample_dist = cdist(resample_len.reshape([-1,1]), cumsum_dist.reshape([-1,1]))
    resample_idx = np.argmin(resample_dist, axis=1)
    resample_idx = np.append(resample_idx, len(pos)-1)
    return resample_idx    

if __name__ == '__main__':
    '''a = np.arange(22).reshape([11,2])
    idx = resample(a, 4)
    b = a[idx,:]'''
    ang = np.arange(36)/36.0*6.28
    print('>>>>>>>>>>> ang' ,ang.shape)
    a = np.array([np.cos(ang)*10, np.sin(ang)*10])
    a = a.T
    print('>>>>>>>>>>> a' ,a.shape)
    b = pca(a,0.6)
    print('>>>>>>>>>>> b' ,b.shape)
    plt.figure()
    plt.scatter(a[:,0], a[:,1])
    plt.scatter(b[:,0], b[:,1])
    plt.show()
    
    
