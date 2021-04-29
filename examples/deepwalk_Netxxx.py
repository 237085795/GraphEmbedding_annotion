import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE

from cluster.clustering import kmeans_from_vec
from ge import DeepWalk
from metric.modularity import cal_Q as Q
from metric.nmi import calc as NMI
from ge.classify import read_node_label

def NMI_Q_plot(embeddings, num_coms):
    emb_list = [embeddings[x] for x in embeddings]

    # clusters = KMeans(n_clusters=num_coms).fit_predict(emb)
    # predict = []
    # for i in range(num_coms):
    #     predict.append(set())
    # for i in range(len(clusters)):
    #     predict[clusters[i]].add(str(i + 1))

    # predict = kmeans_from_vec(emb_list, num_coms)

    # for i in range(len(predict)):
    #     predict[i] = [str(x + 1) for x in predict[i]]
    # 数据处理

    X, Y = read_node_label('../data/Net500/community.dat')

    label_node = {}
    for i in range(len(X)):
        label_node.setdefault(Y[i][0], [])
        label_node[Y[i][0]].append(i)

    # print(label_node)
    emb_list = np.array(emb_list)
    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    plt.figure(dpi=300, figsize=(24, 12))

    for c, idx in label_node.items():
        # print(type(c),type(idx))
        print(c,idx)
        # print(node_pos[idx, 0],node_pos[idx, 1],node_pos[idx,2])
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
        # ax.scatter(node_pos[idx, 0], node_pos[idx, 1], node_pos[idx, 2], label=c)
    plt.legend()  # 图例
    plt.show()

    # real = []
    # # file = open("../data/karate/real_karate.txt")
    # # file = open("../data/dophlin/real_dolphin.txt")
    # file = open("../data/football/real_football.txt")
    #
    # while 1:
    #     line = file.readline()
    #     if not line:
    #         break
    #     else:
    #         real.append([i for i in line.split()])
    # file.close()
    #
    # # print(predict, real)
    # print("nmi:")
    # print(NMI(predict, real))
    # print("q:")
    # print(Q(predict, G))
    #

    # # print(node_pos)
    # # print(len(real), real)
    #
    # # colors = ['c', 'b', 'g', 'r', 'm', 'y', 'k', 'w','c', 'b', 'g', 'r', 'm', 'y', 'k', 'w']
    # # print(colors)
    #
    # for community in range(len(real)):
    #     for node_idx in real[community]:
    #         # print(community,node_idx)
    #         plt.scatter(node_pos[int(node_idx) - 1, 0], node_pos[int(node_idx) - 1, 1], label=str(community))
    # plt.legend
    # plt.show()

def plot_embeddings(embeddings):
    """将嵌入降维到2维并可视化

    :param embeddings:
    """
    X, Y = read_node_label('../data/Net500/community.dat')

    # print(Y)
    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    # model = TSNE(n_components=3)
    node_pos = model.fit_transform(emb_list)  # 返回矩阵
    # print(node_pos)


    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)
    # print(color_idx)
    # 生产标签_节点字典
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # 分辨率参数-dpi，画布大小参数-figsize
    plt.figure(dpi=300, figsize=(24, 12))
    for c, idx in color_idx.items():
        print(type(idx))
        print(c,idx)
        # print(node_pos[idx, 0],node_pos[idx, 1],node_pos[idx,2])
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
        # ax.scatter(node_pos[idx, 0], node_pos[idx, 1], node_pos[idx, 2], label=c)
    plt.legend()  # 图例
    plt.show()

if __name__ == "__main__":

    G = nx.read_edgelist('../data/Net500/network.dat', create_using=nx.Graph(), nodetype=None)

    model = DeepWalk(G, walk_length=30, num_walks=5, workers=1)
    model.train(window_size=5, iter=5)

    # model = Node2Vec(G, walk_length=10, num_walks=80,
    #                  p=0.25, q=4, workers=1, use_rejection_sampling=0)
    # model.train(window_size = 5, iter = 3)

    # model = LINE(G, embedding_size=128, order='second')
    # # model = LINE(G, embedding_size=128, order='first')
    # model = LINE(G, embedding_size=128, order='all')
    # model.train(batch_size=1024, epochs=50, verbose=2)

    embeddings = model.get_embeddings()
    print(embeddings)

    # NMI_Q_plot(embeddings, 17)
    # plot_embeddings(embeddings)
