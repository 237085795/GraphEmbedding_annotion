import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE

from cluster.clustering import kmeans_from_vec
from ge import DeepWalk,LINE,Node2Vec
from metric.modularity import cal_Q as Q
from metric.nmi import calc as NMI


# [{'28', '29', '11', '56', '62', '31', '15', '52', '9', '8', '12', '50', '58', '16', '26', '30', '25', '27', '10', '51', '36'}
# {'49', '1', '34', '46', '20', '22', '48', '13', '60', '41', '45', '24', '3', '32', '57', '53', '61', '43', '17', '23', '37', '59', '7', '19', '5', '54', '18', '39', '6', '14', '42', '4', '38', '47', '40', '44', '35', '2', '33', '55', '21'}]

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
    # print(len(real), real)

    # colors = ['c', 'b', 'g', 'r', 'm', 'y', 'k']
    # print(colors)
    real = []
    file = open("../data/karate/real_karate.txt")
    # file = open("../data/dophlin/real_dolphin.txt")
    # file = open("../data/football/real_football.txt")

    while 1:
        line = file.readline()
        if not line:
            break
        else:
            real.append([i for i in line.split()])
    file.close()

    # real = [['1', '5', '6', '7', '9', '13', '17', '19', '22', '25', '26', '27', '31', '32', '41', '48', '54', '56', '57', '60'], ['0', '2
    colors = ['r' ,'b']

    for community in range(len(real)):
        for node_idx in real[community]:
            # print(community,node_idx)
            plt.scatter(node_pos[int(node_idx)-1, 0], node_pos[int(node_idx)-1, 1],c=colors[community])  # real从1开始，要-1

    plt.legend
    plt.show()

if __name__ == "__main__":
    # G = nx.read_edgelist('../data/karate/karate_edgelist.txt', create_using=nx.Graph(), nodetype=None)
    G = nx.read_edgelist('../data/football/football_edgelist.txt', create_using=nx.Graph(), nodetype=None)
    # G = nx.read_edgelist('../data/dophlin/dophlin_edgelist.txt', create_using=nx.Graph(), nodetype=None)


    iter=100
    sum_mni=0
    sum_q=0
    for i in range(iter):
        # model = DeepWalk(G, embed_size=32,walk_length=20, num_walks=20, workers=1)
        # model.train(window_size=5, iter=5)

        # model = Node2Vec(G, embed_size=32,walk_length=20, num_walks=20,
        #                  p=0.25, q=2, workers=1, use_rejection_sampling=0)
        # model.train(window_size = 5, iter = 3)


        model = LINE(G, embedding_size=32, order='all')
        model.train(batch_size=1024, epochs=50, verbose=2)

        embeddings = model.get_embeddings()

        mertric=NMI_Q(embeddings, 12)
        sum_mni+=mertric[0]
        sum_q+=mertric[1]
    print("mni:")
    print(sum_mni/iter)
    print("q:")
    print(sum_q/iter)

    # model = DeepWalk(G, embed_size=16,walk_length=10, num_walks=20, workers=1)
    # model.train(window_size=5, iter=5)
    #
    # model = LINE(G, embedding_size=16, order='all')
    # model.train(batch_size=1024, epochs=50, verbose=2)

    # model = Node2Vec(G, embed_size=16,walk_length=10, num_walks=20,
    #                  p=0.25, q=2, workers=1, use_rejection_sampling=0)
    # model.train(window_size = 5, iter = 3)
    #
    # embeddings = model.get_embeddings()
    # plot(embeddings)