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
                   "skeleton_in_ph, pointSet_in_ph,  predictedSet" )

def encoder(input_points, FLAGS, is_training=False, bn_decay=None):

    l0_xyz = input_points
    l0_points = None
    l4_xyz, l4_points, l4_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=None, radius=None, nsample=None,
                    mlp=[32, 32, 64], mlp2=None, group_all=True,
                    is_training=is_training, bn_decay=bn_decay, scope='sa_layer4')
    l4_points = tf.squeeze(l4_points, 0)
    return l4_points


def get_displacements(input_points, ske_features, FLAGS, is_training = False, bn_decay=None):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """

    batch_size = FLAGS.batch_size
    num_points = FLAGS.point_num_out

    point_cloud = input_points

    l0_xyz = point_cloud
    l0_points = None

    # Set Abstraction layers 第一从次2048个点提取1024个点
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=1024, radius=0.1 * FLAGS.radiusScal, nsample=64,
                                                       mlp=[64, 64, 128], mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer1')  ### 最后一个变量scope相当于变量前缀
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=384, radius=0.2* FLAGS.radiusScal, nsample=64,
                                                       mlp=[128, 128, 256], mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=128, radius=0.4* FLAGS.radiusScal, nsample=64,
                                                       mlp=[256, 256, 512], mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer3')

    # PointNet
    l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=None, radius=None, nsample=None,
                                                       mlp=[512, 512, 1024], mlp2=None, group_all=True,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer4')

     ### Feature Propagation layers  #################  featrue maps are interpolated according to coordinate  ################     
    # 根据l4的特征值差值出l3
    l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [512, 512], is_training, bn_decay, scope='fa_layer1')
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [512, 256], is_training, bn_decay, scope='fa_layer2')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256, 128], is_training, bn_decay, scope='fa_layer3')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128, 128, 128], is_training, bn_decay, scope='fa_layer4')

    # 加入提取的skeleton特征 
    # ske_features : batch_size * featrues
    ske_features = tf.tile(tf.expand_dims(ske_features, 1), [1, num_points, 1])
    l0_points = tf.concat([l0_points, ske_features], axis=-1)
    # 特征转变成 displacement
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay )
    net = tf_util.conv1d(net, 64, 1, padding='VALID', bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.conv1d(net, 3, 1, padding='VALID', activation_fn=None, scope='fc3')

    displacements = tf.sigmoid(net) * FLAGS.range_max * 2 - FLAGS.range_max

    return displacements



def load_model():
    
    #with tf.Graph().as_default():
    # FLAGS
    FLAGS = collections.namedtuple("FLAGS", "gpu, batch_size, point_num_out, point_num_in, range_max, radiusScal")
    FLAGS.gpu = 0
    FLAGS.batch_size = 1
    FLAGS.point_num_out = 2048
    FLAGS.point_num_in = 19
    FLAGS.range_max = 1
    FLAGS.radiusScal = 1

    with tf.device('/gpu:'+str(FLAGS.gpu)):
                pointSet_in_ph = tf.placeholder( tf.float32, shape=(FLAGS.batch_size, FLAGS.point_num_out, 3) )
                skeleton_in_ph = tf.placeholder( tf.float32, shape=(FLAGS.batch_size, FLAGS.point_num_in, 3) )
                ske_features = encoder(skeleton_in_ph, FLAGS)
                displacements = get_displacements(pointSet_in_ph, ske_features, FLAGS, bn_decay=None)
                predictedSet = pointSet_in_ph + displacements

    
    return Model(pointSet_in_ph = pointSet_in_ph, skeleton_in_ph = skeleton_in_ph, predictedSet = predictedSet)

# pointSet_in: shape=(FLAGS.point_num_in, 3) 
def gen_horses(sess, model, skeleton_in, pointSet_in, mustSavePly, nametosave):
    
    pointSet_in = np.array(pointSet_in, ndmin=3)
    #print('type of pointSet_in : ', type(pointSet_in))
    #print('shape of pointSet_in : ', pointSet_in.shape)
       # input data

    feed_dict = {
        model.pointSet_in_ph: pointSet_in,
        model.skeleton_in_ph: skeleton_in
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
    #for i in range(3):
    #    results = sess.run(fetches, feed_dict=feed_dict)
    #    Predicted_xyz__ = np.array(results["predictedSet"])
    #    Predicted_xyz4 = np.concatenate((Predicted_xyz1, Predicted_xyz__), axis=1)

    # save predicted point sets with 8 feeding passes
    #for i in range(4):
    #    results = sess.run(fetches, feed_dict=feed_dict)
    #    Predicted_xyz__ = np.array(results["predictedSet"])
    #    Predicted_xyz8 = np.concatenate((Predicted_xyz4, Predicted_xyz__), axis=1)

    if mustSavePly:
        output_dir = 'myNet/output'
        nametosave = np.array(nametosave, ndmin=1)
        #print('length of names : ', len(nametosave))
        #print('names : ', nametosave[0])
        ioUtil.output_point_cloud_ply( Predicted_xyz1, nametosave, output_dir,
        'Ep' + '_predicted_' + 'X1')
        #ioUtil.output_point_cloud_ply( Predicted_xyz4, nametosave, output_dir,
        #   'Ep' + '_predicted_' + 'X4')   
        #ioUtil.output_point_cloud_ply( Predicted_xyz8, nametosave, output_dir,
        #   'Ep' + '_predicted_' + 'X8')    

    tf.Session.close
    Predicted_xyz = np.squeeze(Predicted_xyz1)
    return Predicted_xyz

# main function
if __name__ == '__main__':
    modelPath = 'myNet/trained_models/'
    data = ioUtil.load_examples('Skeletons/horse.hdf5', 'names')
    Y = data.skeleton_in
    N = Y.shape[0]
    D = Y.shape[1]
    pointSet_in = Y[0,:] # [19 * 3]
    nametosave = data.names[0]
    print(type(pointSet_in))
    mustSavePly = True

    with tf.Graph().as_default():
        model = load_model()
        # check point: 二进制文件，它包含的权重变量，biases变量和其他变量
        config = tf.ConfigProto(allow_soft_placement = True)
        sess = tf.Session(config = config)
        
        #metaPath = modelPath + 'epoch_200.ckpt.meta'
        ckptPath = tf.train.latest_checkpoint(modelPath)
        print('load checkpoint: ' + ckptPath)
        saver = tf.train.Saver( max_to_keep=5)
        #saver = tf.train.import_meta_graph(metaPath)
        saver.restore(sess, ckptPath )
 
        gen_horses(sess, model, pointSet_in, mustSavePly, nametosave)

        tf.Session.close




