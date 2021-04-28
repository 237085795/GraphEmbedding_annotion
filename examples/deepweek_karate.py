import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from ge import DeepWalk, Node2Vec, LINE
from metric.modularity import cal_Q as Q
from metric.nmi import calc as NMI

from cluster.clustering import kmeans_from_vec


def NMI_Q_plot(embeddings, num_coms=2):
    emb = [embeddings[x] for x in embeddings]

    # clusters = KMeans(n_clusters=num_coms).fit_predict(emb)
    # predict = []
    # for i in range(num_coms):
    #     predict.append(set())
    # for i in range(len(clusters)):
    #     predict[clusters[i]].add(str(i + 1))
    predict = kmeans_from_vec(emb,num_coms)
    for i in range(len(predict)):
        predict[i] = [str(x+1) for x in predict[i]]

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
    print(predict, real)
    print("nmi:")
    print(NMI(predict, real))
    print("q:")
    print(Q(predict, G))

    emb = np.array(emb)
    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb)
    # print(node_pos)
    print(len(real), real)

    colors = ['c', 'b', 'g', 'r', 'm', 'y', 'k', 'w','c', 'b', 'g', 'r', 'm', 'y', 'k', 'w']
    print(colors)
    for comu in range(len(real)):
        for idx in real[comu]:
            # print(comu,idx)
            plt.scatter(node_pos[int(idx) - 1, 0], node_pos[int(idx) - 1, 1], c=colors[comu])
    plt.legend
    plt.show()

if __name__ == "__main__":
    # G = nx.read_edgelist('../data/karate/karate_edgelist.txt', create_using=nx.Graph(), nodetype=None)
    G = nx.read_edgelist('../data/football/football_edgelist.txt', create_using=nx.Graph(), nodetype=None)
    # G = nx.read_edgelist('../data/dophlin/dophlin_edgelist.txt', create_using=nx.Graph(), nodetype=None)


    # model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)
    # model.train(window_size=5, iter=3)

    model = Node2Vec(G, walk_length=10, num_walks=80,
                     p=0.25, q=4, workers=1, use_rejection_sampling=0)
    model.train(window_size = 5, iter = 3)

    # model = LINE(G, embedding_size=128, order='second')
    # # model = LINE(G, embedding_size=128, order='first')
    # model = LINE(G, embedding_size=128, order='all')
    # model.train(batch_size=1024, epochs=50, verbose=2)

    embeddings = model.get_embeddings()

    NMI_Q_plot(embeddings,12)
