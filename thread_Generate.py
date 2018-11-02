# coding:utf-8
import numpy as np
import tensorflow as tf
import threading
import time

import gen_horse2horse

def action(ske, newX, embeds, pointSet, sess, model, nametosave):
    #global sess
    ske = np.reshape(ske, (1, -1, 3))
    # new shapes : get horses here
    neighbor_horse = find_neighbor(newX, embeds, pointSet)
    neighbor_horse = np.reshape(neighbor_horse, [1, 2048, 3])
    mustSavePly = True
    horse_xyz = gen_horse2horse.gen_horses(sess, model, ske, neighbor_horse, mustSavePly, nametosave)

def find_neighbor(pos, embeds, pointSet):
    n = embeds.shape[0]
    dist = np.sum(np.square(embeds-pos), axis=1)
    dist = np.sqrt(dist)
    min_pos = np.argmin(dist)
    return pointSet[min_pos, ...]

def generateModels(mu_fFull, newXs, embeds, pointSet, sess, model, nametosave):
    n = len(nametosave)
    ske_xyz = np.reshape(mu_fFull, (n, -1, 3))
    for i in range(len(ske_xyz)):
        ske = ske_xyz[i,:]
        newX = newXs[i]
        name = nametosave[i]
        t =threading.Thread(target=action,args=(ske, newX, embeds, pointSet, sess, model, name))
        t.start()
