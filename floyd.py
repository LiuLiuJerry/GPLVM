#coding:utf-8
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist, squareform
import matplotlib as mpl
import matplotlib.pyplot as plt

def calculate_floyd(Coords, r=0.5):
    n = Coords.shape[0]
    tri = Delaunay(Coords)
    tri_idx = tri.simplices.copy()
    #print('>>>>>>>>>Delaunay', tri_idx)
    M = np.ones([n, n], dtype=float)
    M = M * float('inf')

    Cdist = pdist(Coords)
    Cdist = np.array(squareform(Cdist))
    for x in tri_idx:
        a,b,c = x
        if Cdist[a,b] < r:
            M[a, b] = 1
            M[b, a] = 1
        if Cdist[a,c] < r:
            M[a, c] = 1
            M[c, a] = 1
        if Cdist[c,b] < r:
            M[c, b] = 1
            M[b, c] = 1

    #print('>>>>>>>>>M', M.shape)
    dist = M*Cdist
    #print('>>>>>>>>>shape of dist', dist.shape) 
    #print('>>>>>>>>>>>>>>>>>>dist', dist) 
    parent = np.ones([n,n])
    for i in range(n):
        parent[:,i] = i

    for k in range(n):
        for i in range(n):
            for j in range(i,n):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    dist[j][i] = dist[i][j]
                    parent[i][j] = parent[i][k]
                    parent[j][i] = parent[j][k]
    #print('>>>>>>>>>>>parent', parent)

    edges = np.where(dist<r)
    return [dist, parent, edges]

def obtainPath(i, j, parent):
    if j == i:
        return []
    if parent[i][j] == j:
        return [j]
    else:
        #return obtainPath(i, parent[i][j], parent) + [parent[i][j]] + obtainPath(parent[i][j], j, parent)
        return obtainPath(i, parent[i][j], parent) + obtainPath(parent[i][j], j, parent)

def cal_shortestpath(preX, newX, embeds, dist, r=0.5):
    n = len(dist)

    dist1 = np.sum(np.square(embeds-preX), axis=1)
    dist1 = np.sqrt(dist1)
    dist1 = dist1 * (dist1<r)
    dist2 = np.sum(np.square(embeds-newX), axis=1)
    dist2 = np.sqrt(dist2)   
    dist2 = dist2 * (dist2<r)
    len_path = 0.0
    pStart = 0
    pEnd = 0
    for i in range(n):
        if dist1[i] == 0:
            continue
        for j in range(n):
            if dist2[j] == 0:
                continue
            if len_path == 0.0 or dist1[i]+dist2[j]+dist[i][j] < len_path:
                len_path = dist1[i]+dist2[j]+dist[i][j]
                pStart = i
                pEnd = j

    return [pStart, pEnd]
        

if __name__ == '__main__':
    
    Coords = np.random.rand(50)
    Coords = Coords.reshape([25,2])
    #Coords[:,2] = 0
    res = calculate_floyd(Coords)
    
    dist = res[0]
    parent = np.array(res[1], dtype=int)
    delaunay = res[2]
    print('>>>>>>>>>>>>>dist',dist)
    print('>>>>>>>>>>>>>parent',parent)

    path_idx = obtainPath(0, 10, dist, parent)
    print('>>>>>>>>>>>>>path',path_idx)
    path_idx.append(10)
    path = Coords[path_idx,:]

    

    fig = plt.figure()
    plt.scatter(Coords[:,0], Coords[:,1])  
    plt.triplot(Coords[:,0], Coords[:,1], delaunay, linewidth=1.5)
    plt.plot(path[:,0], path[:,1], 'r')
    plt.show()

