import os
import sys
import numpy as np
import h5py
import collections

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


Examples = collections.namedtuple("Examples", "names, pointSet_in, pointSet_out")


def shuffle_examples( data ):

    idx = np.arange(  data.names.shape[0] )
    np.random.shuffle(idx)

    return Examples(
        names=data.names[idx, ...],
        pointSet_in=data.pointSet_in[idx, ...],
        pointSet_out=data.pointSet_out[idx, ...],
    )


def load_examples(h5_filename,  fieldname_modelname ):
    f = h5py.File(h5_filename)
    # to be updated 
    fieldname_in = 'skeleton'
    fieldname_out = 'surface'
    pointSet_in = f[fieldname_in][:]
    pointSet_out = f[fieldname_out][:]
    names = f[fieldname_modelname][:]
    print(pointSet_in.size)
    print(pointSet_out.size)
    return Examples(
        names=names,
        pointSet_in=pointSet_in,
        pointSet_out=pointSet_out,
    )

def load_skeletons(h5_filename,  fieldname_modelname):
    f = h5py.File(h5_filename)
    # to be updated 
    fieldname = 'skeleton'
    pointsets = f[fieldname][:]
    names = f[fieldname_modelname][:]
    return Examples(
        names=names,
        pointSet_in=pointsets,
        pointSet_out = []
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
