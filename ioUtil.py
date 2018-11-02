import os
import sys
import numpy as np
import h5py
import collections
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


Examples = collections.namedtuple("Examples", "names, skeleton_in, pointSet_out")
Datas = collections.namedtuple("Datas", "names, skeleton_in, pointSet_in, pointSet_out")
def arrange_datas(examples):
    ## A->A, B->A, C->A, D->A, A->B, B->B, C->B, D->B
    skepts = examples.skeleton_in.shape[1]
    npts = examples.pointSet_out.shape[1]
    ############## train examples #########################
    skeleton_in = examples.skeleton_in
    pointSet_out = examples.pointSet_out # [n, npts, 3]
    names = examples.names
    EXAMPLE_NUM = examples.skeleton_in.shape[0]

    sess = tf.Session()
    name1 = sess.run(tf.tile(names, [EXAMPLE_NUM]))
    name2 = tf.expand_dims(names, 1)
    name2 = sess.run(tf.reshape(tf.tile(name2, [1, EXAMPLE_NUM]), [EXAMPLE_NUM*EXAMPLE_NUM]))

    newnames = []
    for i in range(EXAMPLE_NUM*EXAMPLE_NUM):
        #print(name1[i][0:-4] + '_'+name2[i])
        newnames.append(name1[i][0:-4] + '_'+name2[i])

    pointSet_in = pointSet_out
    pointSet_in = sess.run(tf.tile(pointSet_in, [EXAMPLE_NUM, 1, 1]))# ABCDABCDABCD

    skeleton_in = sess.run(tf.reshape(tf.tile(skeleton_in, [1, EXAMPLE_NUM, 1]), [EXAMPLE_NUM*EXAMPLE_NUM, skepts, 3]))
    ps_out = tf.tile(pointSet_out, [1, EXAMPLE_NUM, 1])
    pointSet_out = sess.run(tf.reshape(ps_out, [EXAMPLE_NUM*EXAMPLE_NUM, npts, 3]) )#AAABBBCCCDDD

    print('skeleton_in', skeleton_in.shape)
    print('pointSet_in', pointSet_in.shape)
    print('pointSet_out', pointSet_out.shape)

    return Datas(
                        skeleton_in = np.array(skeleton_in),
                        pointSet_in = np.array(pointSet_in),
                        pointSet_out = np.array(pointSet_out),
                        names = np.array(newnames),
    )

def shuffle_datas( data ):

    idx = np.arange(  data.names.shape[0] )
    np.random.shuffle(idx)

    return Datas(
        names=data.names[idx, ...],
        skeleton_in=data.skeleton_in[idx, ...],
        pointSet_in=data.pointSet_in[idx, ...],
        pointSet_out=data.pointSet_out[idx, ...],
    )

def shuffle_examples( data ):

    idx = np.arange(  data.names.shape[0] )
    np.random.shuffle(idx)

    return Examples(
        names=data.names[idx, ...],
        skeleton_in=data.skeleton_in[idx, ...],
        pointSet_out=data.pointSet_out[idx, ...],
    )


def load_examples(h5_filename,  fieldname_modelname ):
    f = h5py.File(h5_filename)
    # to be updated 
    fieldname_in = 'skeleton'
    fieldname_out = 'surface'
    skeleton_in = f[fieldname_in][:]
    pointSet_out = f[fieldname_out][:]
    names = f[fieldname_modelname][:]
    print('size of skeleton_in : ', skeleton_in.size)
    print('size of pointSet_out : ', pointSet_out.size)
    return Examples(
        names=names,
        skeleton_in=skeleton_in,
        pointSet_out=pointSet_out,
    )


def output_point_cloud_ply(xyzs, names, output_dir, foldername ):

    if not os.path.exists( output_dir ):
        os.mkdir(  output_dir  )

    plydir = output_dir + '/' + foldername

    if not os.path.exists( plydir ):
        os.mkdir( plydir )

    numFiles = len(names)

    for fid in range(numFiles):

        print('write: ' + plydir +'/'+names[fid]+'.ply')

        with open( plydir +'/'+names[fid]+'.ply', 'w') as f:
            pn = xyzs.shape[1]
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write('element vertex %d\n' % (pn) )
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('end_header\n')
            for i in range(pn):
                f.write('%f %f %f\n' % (xyzs[fid][i][0],  xyzs[fid][i][1],  xyzs[fid][i][2]) )
