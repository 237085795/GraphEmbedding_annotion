
import numpy as np

from ge.classify import read_node_label, Classifier
from ge import DeepWalk
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE

from mpl_toolkits.mplot3d import Axes3D


def evaluate_embeddings(embeddings):
    """分割训练评估，输出性能参数

    :param embeddings:
    """
    # X, Y = read_node_label('../data/wiki/wiki_labels.txt')
    # X, Y = read_node_label('../data/flight/labels-brazil-airports.txt', True)
    # X, Y = read_node_label('../data/flight/labels-europe-airports.txt', True)
    X, Y = read_node_label('../data/flight/labels-usa-airports.txt', True)
    tr_frac = 0.8  # 交叉验证百分比
    print("Training classifier using {:.2%} nodes...".format(tr_frac))
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)


def plot_embeddings(embeddings,):
    """将嵌入降维到2维并可视化

    :param embeddings:
    """
    # X, Y = read_node_label('../data/wiki/wiki_labels.txt')
    # X, Y = read_node_label('../data/flight/labels-brazil-airports.txt', True)
    # X, Y = read_node_label('../data/flight/labels-europe-airports.txt', True)
    X, Y = read_node_label('../data/flight/labels-usa-airports.txt', True)

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
    for c, idx in color_idx.items():
        # print(c,idx)
        # print(node_pos[idx, 0],node_pos[idx, 1],node_pos[idx,2])
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
        # ax.scatter(node_pos[idx, 0], node_pos[idx, 1], node_pos[idx, 2], label=c)
    plt.legend()  # 图例
    plt.show()

def plot_embeddings_3D(embeddings,):
    """将嵌入降维到3维并可视化

    :param embeddings:
    """
    X, Y = read_node_label('../data/wiki/wiki_labels.txt')
    # print(Y)
    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=3)
    node_pos = model.fit_transform(emb_list)  # 返回矩阵
    # print(node_pos)


    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)
    # print(color_idx)
    # 生产标签_节点字典
    fig = plt.figure()
    ax = Axes3D(fig)
    for c, idx in color_idx.items():
        # print(c,idx)
        # print(node_pos[idx, 0],node_pos[idx, 1],node_pos[idx,2])
        ax.scatter(node_pos[idx, 0], node_pos[idx, 1], node_pos[idx, 2], label=c)
    plt.legend()  # 图例
    plt.show()


if __name__ == "__main__":
    # G = nx.read_edgelist('../data/wiki/Wiki_edgelist.txt',
    #                      create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    # G = nx.read_edgelist('../data/flight/brazil-airports.edgelist',
    #                      create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    # G = nx.read_edgelist('../data/flight/europe-airports.edgelist',
    #                      create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    G = nx.read_edgelist('../data/flight/usa-airports.edgelist',
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)
    model.train(window_size=5, iter=3)
    embeddings = model.get_embeddings()

    evaluate_embeddings(embeddings)
    plot_embeddings(embeddings)
    # plot_embeddings_3D(embeddings)