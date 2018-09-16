#coding:utf-8
from os import listdir
from os.path import isfile, join
import argparse
import h5py

from plyfile import PlyData

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default='Skeletons/horse_kpts' )
parser.add_argument("--num", default=19 )

FLAGS = parser.parse_args()

skeFolder = FLAGS.mode
pointNum = int(FLAGS.num)

# 一种简洁的构建List的方法，从for给定的List中选择出满足if条件的元素组成新的List，其中if是可以省略的
skefiles = [f for f in listdir(skeFolder) if isfile(join(skeFolder, f)) and f.endswith('.ply')  ]

numFiles = len(skefiles)

f = h5py.File( skeFolder + ".hdf5", "w")

string_dt = h5py.special_dtype(vlen=str)

names = f.create_dataset("names", (numFiles,), dtype=string_dt)
skeleton_points = f.create_dataset("skeleton", (numFiles, pointNum, 3), dtype='f')



for fid in range(numFiles):

    skename = skefiles[fid]
    basename = skename[0:-4]

    skel_path  = skeFolder + '/' + skename

    skelPlyData = PlyData.read( skel_path )


    names[ fid] = basename
    #print names[fid]
    print fid

    skeleton_points[ fid, :, 0 ] = skelPlyData['vertex']['x']
    skeleton_points[ fid, :, 1 ] = skelPlyData['vertex']['y']
    skeleton_points[ fid, :, 2 ] = skelPlyData['vertex']['z']

names.flush()
skeleton_points.flush()

