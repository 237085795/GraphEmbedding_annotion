import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE

from cluster.clustering import kmeans_from_vec
from ge import DeepWalk
from metric.modularity import cal_Q as Q
from metric.nmi import calc as NMI


def NMI_Q_plot(embeddings, num_coms):
    emb_list = [embeddings[x] for x in embeddings]

    # clusters = KMeans(n_clusters=num_coms).fit_predict(emb)
    # predict = []
    # for i in range(num_coms):
    #     predict.append(set())
    # for i in range(len(clusters)):
    #     predict[clusters[i]].add(str(i + 1))
    predict = kmeans_from_vec(emb_list, num_coms)

    for i in range(len(predict)):
        predict[i] = [str(x+1) for x in predict[i]]
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

    for j in range(len(real)):
        real[i] = [str(int(y)+1) for y in real[i]]
    # print(predict)
    # print(real)
    print("nmi:")
    print(NMI(predict, real))
    print("q:")
    print(Q(real, G))

    # emb_list = np.array(emb_list)
    # model = TSNE(n_components=2)
    # node_pos = model.fit_transform(emb_list)
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


if __name__ == "__main__":
    # G = nx.read_edgelist('../data/karate/karate_edgelist.txt', create_using=nx.Graph(), nodetype=None)
    G = nx.read_edgelist('../data/football/football_edgelist.txt', create_using=nx.Graph(), nodetype=None)
    # G = nx.read_edgelist('../data/dophlin/dophlin_edgelist.txt', create_using=nx.Graph(), nodetype=None)

    model = DeepWalk(G, walk_length=20, num_walks=5, workers=1)
    model.train(window_size=5, iter=5)

    # model = Node2Vec(G, walk_length=10, num_walks=80,
    #                  p=0.25, q=4, workers=1, use_rejection_sampling=0)
    # model.train(window_size = 5, iter = 3)

    # model = LINE(G, embedding_size=128, order='second')
    # # model = LINE(G, embedding_size=128, order='first')
    # model = LINE(G, embedding_size=128, order='all')
    # model.train(batch_size=1024, epochs=50, verbose=2)

    embeddings = model.get_embeddings()

    NMI_Q_plot(embeddings, 12)
