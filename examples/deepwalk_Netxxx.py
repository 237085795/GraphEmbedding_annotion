import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE

from cluster.clustering import kmeans_from_vec
from ge import DeepWalk,LINE,Node2Vec
from metric.modularity import cal_Q as Q
from metric.nmi import calc as NMI
from ge.classify import read_node_label

def NMI_Q(embeddings, num_coms):
    emb_list=[]
    for i in range(1,len(embeddings)+1):
        emb_list.append(embeddings[str(i)])
    # edgelist index start at 1  / emb_list index start at 0


    predict = kmeans_from_vec(emb_list, num_coms)  # index start by 0

    for i in range(len(predict)):
        predict[i] = [str(x) for x in predict[i]]
    # 数据处理

    real = []
    # file = open("../data/karate/real_karate.txt")
    # file = open("../data/dophlin/real_dolphin.txt")
    file = open("../data/football/real_football.txt")

    while 1:
        line = file.readline()
        if not line:
            break
        else:
            real.append([i for i in line.split()])
    file.close()


    mni = NMI(predict, real)

    predict_=predict
    for i in range(len(predict)):
        predict_[i]=[str(int(x)+1) for x in predict[i]]
    # predict index add 1


    q=Q(predict_, G)

    return mni,q

def plot(embeddings):
    emb_list=[]
    for i in range(1,len(embeddings)+1):
        # print(i)
        emb_list.append(embeddings[str(i)])

    emb_list = np.array(emb_list)
    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)
    # print(node_pos)

    X, Y = read_node_label('../data/Net300/community.dat')

    # colors = ['c', 'b', 'g', 'r', 'm', 'y', 'k']
    colors = ['r' ,'b','g','y']

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    print(color_idx)

    # plt.legend
    # plt.show()

if __name__ == "__main__":

    G = nx.read_edgelist('../data/Net300/network.dat', create_using=nx.Graph(), nodetype=None)

    # iter = 100
    # sum_mni = 0
    # sum_q = 0
    # for i in range(iter):
    #     # model = DeepWalk(G, embed_size=32,walk_length=20, num_walks=20, workers=1)
    #     # model.train(window_size=5, iter=5)
    #
    #     # model = Node2Vec(G, embed_size=32,walk_length=20, num_walks=20,
    #     #                  p=0.25, q=2, workers=1, use_rejection_sampling=0)
    #     # model.train(window_size = 5, iter = 3)
    #
    #     model = LINE(G, embedding_size=32, order='all')
    #     model.train(batch_size=1024, epochs=50, verbose=2)
    #
    #     embeddings = model.get_embeddings()
    #
    #     mertric = NMI_Q(embeddings, 12)
    #     sum_mni += mertric[0]
    #     sum_q += mertric[1]
    # print("mni:")
    # print(sum_mni / iter)
    # print("q:")
    # print(sum_q / iter)

    model = DeepWalk(G, embed_size=128,walk_length=25, num_walks=5, workers=1)
    model.train(window_size=5, iter=5)
    #
    # model = LINE(G, embedding_size=16, order='all')
    # model.train(batch_size=1024, epochs=50, verbose=2)

    # model = Node2Vec(G, embed_size=16,walk_length=10, num_walks=20,
    #                  p=0.25, q=2, workers=1, use_rejection_sampling=0)
    # model.train(window_size = 5, iter = 3)
    #
    embeddings = model.get_embeddings()
    plot(embeddings)
