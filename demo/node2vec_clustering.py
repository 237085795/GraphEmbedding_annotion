import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import networkx as nx
from cluster.clustering import kmeans_from_vec
from ge import Node2Vec
from ge.classify import read_node_label
from metric.modularity import cal_Q as Q
from metric.nmi import calc as NMI
from ge.utils import show

def NMI_Q(embeddings, num_coms):
    emb_list = []
    for i in range(1, len(embeddings) + 1):
        emb_list.append(embeddings[str(i)])
    # edgelist index start at 1  / emb_list index start at 0

    predict = kmeans_from_vec(emb_list, num_coms)  # index start by 0

    for i in range(len(predict)):
        predict[i] = [str(x) for x in predict[i]]
    # 数据处理

    X, Y = read_node_label('../data/Net_mu/Net0.1/community.dat', is_net=True)

    comu_idx = {}
    for i in range(len(X)):
        comu_idx.setdefault(Y[i][0], [])
        comu_idx[Y[i][0]].append(i)
    # print(comu_idx)
    real = []
    # print(real)
    for key in comu_idx:
        real.append(comu_idx[key])
    for i in range(len(real)):
        real[i] = [str(x) for x in real[i]]
    # print(predict)
    # print(real)
    nmi = NMI(predict, real)
    # print(mni)
    predict_ = predict
    for i in range(len(predict)):
        predict_[i] = [str(int(x) + 1) for x in predict[i]]
    # predict index add 1
    q = Q(predict_, G)
    # print(q)
    dict = {}
    dict['NMI'] = nmi
    dict['Q'] = q
    return dict


if __name__ == "__main__":
    G = nx.read_edgelist('../data/Net_mu/Net0.1/network.dat', create_using=nx.Graph(), nodetype=None)

    model = Node2Vec(G, embed_size=128, walk_length=25, num_walks=5,
                     p=0.25, q=2, workers=1, use_rejection_sampling=0)
    model.train(window_size=5, iter=3)

    embeddings = model.get_embeddings()
    metric = NMI_Q(embeddings, 5)
    show(metric)
