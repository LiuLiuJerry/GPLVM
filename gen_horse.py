#coding:utf-8
import tensorflow as tf
import numpy as np
import sys
import os
import argparse
import collections

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(   BASE_DIR + "/myNet")
sys.path.append(   BASE_DIR + "/myNet/pointnet_plusplus/utils")
sys.path.append(   BASE_DIR + "/myNet/pointnet_plusplus/tf_ops")
sys.path.append(   BASE_DIR + "/myNet/pointnet_plusplus/tf_ops/3d_interpolation")
sys.path.append(   BASE_DIR + "/myNet/pointnet_plusplus/tf_ops/grouping")
sys.path.append(   BASE_DIR + "/myNet/pointnet_plusplus/tf_ops/sampling")

import tensorflow as tf
import numpy as np # Nummeric python
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module, pointnet_deconv
from tf_sampling import farthest_point_sample, gather_point

import ioUtil

Model = collections.namedtuple("Model",
                   "pointSet_in_ph,  predictedSet" )

def encoder(input_points, FLAGS, is_training=False, bn_decay=None):

    l0_xyz = input_points
    l0_points = None
    l4_xyz, l4_points, l4_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=None, radius=None, nsample=None,
                    mlp=[32, 32, 64], mlp2=None, group_all=True,
                    is_training=is_training, bn_decay=bn_decay, scope='sa_layer4')
    l4_points = tf.squeeze(l4_points)
    return l4_points


def generate(features, FLAGS, is_training=False, bn_decay=None):

    gen_points = FLAGS.generate_num
    ### 将1×1024的特征转变为点云的坐标 全连接层，生成256个点， 每个点256维特征
    net = tf.reshape(features, [FLAGS.batch_size, 64])
    net = tf_util.fully_connected(net, 1024, scope='G_full_conn1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, gen_points*3, scope='G_full_conn2', bn_decay=bn_decay, activation_fn=None)
    net = tf.reshape(net, [FLAGS.batch_size, gen_points, 3])

    return net

def load_model():
    
    #with tf.Graph().as_default():
    # FLAGS
    FLAGS = collections.namedtuple("FLAGS", "gpu, batch_size, generate_num, point_num_in")
    FLAGS.gpu = 0
    FLAGS.batch_size = 1
    FLAGS.generate_num = 2048
    FLAGS.point_num_in = 19

    with tf.device('/gpu:'+str(FLAGS.gpu)):
                pointSet_in_ph = tf.placeholder( tf.float32, shape=(FLAGS.batch_size, FLAGS.point_num_in, 3) )
                features = encoder(pointSet_in_ph, FLAGS)
                genSet = generate(features, FLAGS)
                predictedSet = genSet

    
    return Model(pointSet_in_ph = pointSet_in_ph,      predictedSet = predictedSet)

# pointSet_in: shape=(FLAGS.point_num_in, 3) 
def gen_horses(sess, model, pointSet_in, mustSavePly, nametosave):
    
    #with tf.Graph().as_default():
    pointSet_in = np.array(pointSet_in, ndmin=3)
    # pointSet_in = tf.constant(pointSet_in, shape=[1, FLAGS.point_num_in, 3])
    print('type of pointSet_in : ', type(pointSet_in))
    print('shape of pointSet_in : ', pointSet_in.shape)
       # input data

    feed_dict = {
        model.pointSet_in_ph: pointSet_in,
    }
    
    fetches = {
        "predictedSet": model.predictedSet
    }
    # 计算结果
    results = sess.run(fetches, feed_dict = feed_dict)

    # write test results
    # 多算计次然后取个平均
    # save predicted point sets with 1 single feeding pass
    # Predicted_xyz1 = np.squeeze(np.array(results["predictedSet"]))
    Predicted_xyz1 = np.array(results["predictedSet"])
    print(Predicted_xyz1.shape)
    # save predicted point sets with 4 feeding passes
    for i in range(3):
        results = sess.run(fetches, feed_dict=feed_dict)
        Predicted_xyz__ = np.array(results["predictedSet"])
        Predicted_xyz4 = np.concatenate((Predicted_xyz1, Predicted_xyz__), axis=1)

    # save predicted point sets with 8 feeding passes
    for i in range(4):
        results = sess.run(fetches, feed_dict=feed_dict)
        Predicted_xyz__ = np.array(results["predictedSet"])
        Predicted_xyz8 = np.concatenate((Predicted_xyz4, Predicted_xyz__), axis=1)

    if mustSavePly:
        output_dir = 'myNet/output'
        nametosave = np.array(nametosave, ndmin=1)
        print('length of names : ', len(nametosave))
        print('names : ', nametosave[0])
        ioUtil.output_point_cloud_ply( Predicted_xyz1, nametosave, output_dir,
        'Ep' + '_predicted_' + 'X1')
        ioUtil.output_point_cloud_ply( Predicted_xyz4, nametosave, output_dir,
           'Ep' + '_predicted_' + 'X4')   
        ioUtil.output_point_cloud_ply( Predicted_xyz8, nametosave, output_dir,
           'Ep' + '_predicted_' + 'X8')    

    tf.Session.close
    Predicted_xyz = np.squeeze(Predicted_xyz1)
    return Predicted_xyz

# main function
if __name__ == '__main__':
    modelPath = 'myNet/trained_models/'
    data = ioUtil.load_skeletons('Skeletons/horse_kpts.hdf5', 'names')
    Y = data.pointSet_in
    N = Y.shape[0]
    D = Y.shape[1]
    pointSet_in = Y[0,:] # [19 * 3]
    nametosave = data.names[0]
    print(type(pointSet_in))
    mustSavePly = True

    with tf.Graph().as_default():
        model = load_model()
        # check point: 二进制文件，它包含的权重变量，biases变量和其他变量
        sess = tf.Session()
        
        #metaPath = modelPath + 'epoch_200.ckpt.meta'
        ckptPath = tf.train.latest_checkpoint(modelPath)
        print('load checkpoint: ' + ckptPath)
        saver = tf.train.Saver( max_to_keep=5)
        #saver = tf.train.import_meta_graph(metaPath)
        saver.restore(sess, ckptPath )
 
        gen_horses(sess, model, pointSet_in, mustSavePly, nametosave)

        tf.Session.close




