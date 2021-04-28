#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   clustering.py    
@Contact :   
@License :   

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/8/2 0:52   linsher      1.0         None
'''

import time
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
# from Bio.Cluster import kcluster
# from Bio.Cluster import clustercentroids
# import

def kmeans_(vec_data, comms_num, map_id2node=None, use_set=False,n_init=10,init=None):
    """
    return clf and comms
    :param vec_data:
    :param comms_num:质心数
    :param map_id2node:
    :param use_set:
    :param n_init:运行n_init取最优
    :return:
    """

    if init is None :
        clf = KMeans(n_clusters=comms_num,n_init=n_init)
    else:
        clf = KMeans(n_clusters=comms_num, n_init=n_init,init=init)

    # start = time.time()
    # clf.fit_predict()
    clf.fit(vec_data)
    # end = time.time()
    # print("Kmeans running time: ", end - start)
    # print(clf.inertia_)

    cmus = defaultdict(list)
    if map_id2node is not None:
        for j in range(clf.labels_.shape[0]):
            cmus[clf.labels_[j]].append(map_id2node[j])
    else:
        for j in range(clf.labels_.shape[0]):
            cmus[clf.labels_[j]].append(j)

    comms = []
    for k, v in cmus.items():
        if use_set:
            comms.append(set(v))
        else:
            comms.append(v)


    # cs = clf.cluster_centers_
    # dists = np.zeros((vec_data.shape[0],cs.shape[0]))
    # col = 0
    # for c in cs:
    #     distances = np.sum(np.asarray(vec_data - c) ** 2,axis=1)
    #     # distances = np.sqrt(np.sum(np.asarray(vec_data - c) ** 2, axis=1))
    #     dists[:,col] = distances
    #     col += 1
    # mindst = np.min(dists,axis=1)
    # dstsum = np.sum(mindst)
    # print(dstsum)

    return comms ,clf


def kmeans_from_vec(vec_data, comms_num, map_id2node=None, use_set=False,n_init=10,init=None):

    # 调用kmeans类。
    if init is None :
        clf = KMeans(n_clusters=comms_num,n_init=n_init)
    else:
        clf = KMeans(n_clusters=comms_num, n_init=n_init,init=init)
    # 使用kmeans聚类。
    # start = time.time()
    # clf.fit_predict()
    clf.fit(vec_data)
    # end = time.time()
    # print("Kmeans running time: ", end - start)
    # print(clf.inertia_)

    cmus = defaultdict(list)
    if map_id2node is not None:
        for j in range(clf.labels_.shape[0]):
            cmus[clf.labels_[j]].append(map_id2node[j])
    else:
        for j in range(clf.labels_.shape[0]):
            cmus[clf.labels_[j]].append(j)

    comms = []
    for k, v in cmus.items():
        if use_set:
            comms.append(set(v))
        else:
            comms.append(v)

    return comms


def ward_from_vec(vec_data, comms_num, map_id2node, use_set=False):

    clf = AgglomerativeClustering(n_clusters=comms_num, linkage='ward')

    # clf = AgglomerativeClustering(n_clusters=comms_num, linkage='euclidean')
    # start = time.time()
    s = clf.fit(vec_data)
    # end = time.time()
    # print("Kmeans running time: ", end - start)

    cmus = defaultdict(list)
    for j in range(clf.labels_.shape[0]):
        cmus[clf.labels_[j]].append(map_id2node[j])

    comms = []
    for k, v in cmus.items():
        if use_set:
            comms.append(set(v))
        else:
            comms.append(v)

    return comms




def power_kmeans(vec_data, comms_num, map_id2node, use_set=False):

    clf = AgglomerativeClustering(n_clusters=comms_num, linkage='ward')
    clf.fit(vec_data)

    centers = np.zeros((comms_num,vec_data.shape[1]))
    nodes_num = [0]*comms_num

    for j in range(clf.labels_.shape[0]):
        label = clf.labels_[j]
        centers[label] += vec_data[j]
        nodes_num[label] += 1

    for la in range(comms_num):
        centers[la] /= nodes_num[la]


    clf2 = KMeans(n_clusters=comms_num,init=centers,n_init=1)
    clf2.fit(vec_data)
    print(clf2.inertia_)

    cmus = defaultdict(list)
    if map_id2node is not None:
        for j in range(clf.labels_.shape[0]):
            cmus[clf.labels_[j]].append(map_id2node[j])
    else:
        for j in range(clf.labels_.shape[0]):
            cmus[clf.labels_[j]].append(j)

    comms = []
    for k, v in cmus.items():
        if use_set:
            comms.append(set(v))
        else:
            comms.append(v)

    return comms
