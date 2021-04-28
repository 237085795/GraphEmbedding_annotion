import numpy as np

from ge.classify import read_node_label, Classifier
from ge import DeepWalk
from sklearn.linear_model import LogisticRegression
from ge import Node2Vec
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
import math
from cluster.clustering import kmeans_from_vec
from metric.nmi import calc as NMI
from sklearn.cluster import KMeans
import networkx.algorithms.community as nx_comm
# def NMI(A,B):
#     # 样本点数
#     total = len(A)
#     A_ids = set(A)
#     B_ids = set(B)
#     # 互信息计算
#     MI = 0
#     eps = 1.4e-45
#     for idA in A_ids:
#         for idB in B_ids:
#             idAOccur = np.where(A==idA)    # 输出满足条件的元素的下标
#             idBOccur = np.where(B==idB)
#             idABOccur = np.intersect1d(idAOccur,idBOccur)   # Find the intersection of two arrays.
#             px = 1.0*len(idAOccur[0])/total
#             py = 1.0*len(idBOccur[0])/total
#             pxy = 1.0*len(idABOccur)/total
#             MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
#     # 标准化互信息
#     Hx = 0
#     for idA in A_ids:
#         idAOccurCount = 1.0*len(np.where(A==idA)[0])
#         Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
#         Hy = 0
#     for idB in B_ids:
#         idBOccurCount = 1.0*len(np.where(B==idB)[0])
#         Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
#     MIhat = 2.0*MI/(Hx+Hy)
#     return MIhat

if __name__ == "__main__":
    G = nx.read_edgelist('../data/karate/karate_edgelist.txt',create_using=nx.Graph(), nodetype=None)

    model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)
    model.train(window_size=5, iter=3)

    # model = Node2Vec(G, walk_length=10, num_walks=80,
    #                  p=0.25, q=4, workers=1, use_rejection_sampling=0)
    # model.train(window_size = 5, iter = 3)


    embeddings = model.get_embeddings()
    X_train = [embeddings[x] for x in embeddings]
    num_coms = 2
    clusters = KMeans(n_clusters=num_coms).fit_predict(X_train)
    # print(clusters)

    communities = []
    for i in range(num_coms):
        communities.append(set())

    for i in range(len(clusters)):
        communities[clusters[i]].add(i)
    print(communities)

    # mod = nx_comm.modularity(G, communities)
    # print('The modularity of karate club graph is {:.4f}'.format(mod))

    #
    # predict=kmeans_from_vec(X_train,2)
    #
    real=[]

    file = open("../data/karate/real_karate.txt")
    while 1:
        line = file.readline()
        if not line:
            break
        else:
            real.append([int(i) for i in line.split()])
    file.close()
    #
    # # real = [map(int, l.split()) for l in open('../data/karate/real_karate.txt').readlines()]
    # # print(real)
    # print(predict,real)
    print(NMI(communities,real))
    # print(embeddings)