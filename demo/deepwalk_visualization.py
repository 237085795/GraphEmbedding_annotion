import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE

from ge import DeepWalk
from ge.classify import read_node_label


def plot(embeddings):
    emb_list = []
    for i in range(1, len(embeddings) + 1):
        emb_list.append(embeddings[str(i)])
    # 将嵌入向量字典转换成列表
    emb_list = np.array(emb_list)
    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)
    # 通过tsne转换成二维坐标
    X, Y = read_node_label('../data/Net300/community.dat', is_net=True)
    # ['1', '2', '3', '4'],[['4'], ['2'], ['4'], ['1']]
    commu_idx = {}
    for i in range(len(X)):
        commu_idx.setdefault(Y[i][0], [])
        commu_idx[Y[i][0]].append(i)

    for c, idx in commu_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)

    plt.legend()
    plt.show()

if __name__ == "__main__":
    G = nx.read_edgelist('../data/Net300/network.dat', create_using=nx.Graph(), nodetype=None)

    model = DeepWalk(G, embed_size=128, walk_length=25, num_walks=5, workers=1)
    model.train(window_size=5, iter=5)

    embeddings = model.get_embeddings()
    plot(embeddings)