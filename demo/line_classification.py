import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings("ignore")
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

import networkx as nx
from sklearn.linear_model import LogisticRegression
from ge import LINE
from ge.classify import read_node_label, Classifier

from ge.utils import show


def evaluate_embeddings(embeddings):
    X, Y = read_node_label('../data/flight/labels-usa-airports.txt', skip_head=True)
    tr_frac = 0.8
    print("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression(solver='liblinear'))
    return clf.split_train_evaluate(X, Y, tr_frac)

if __name__ == "__main__":
    G = nx.read_edgelist('../data/flight/usa-airports.edgelist', create_using=nx.Graph(), nodetype=None)

    # model = LINE(G, embedding_size=128, order='first')
    # model = LINE(G, embedding_size=128, order='second')
    model = LINE(G, embedding_size=128, order='all')
    model.train(batch_size=1024, epochs=10, verbose=2)

    embeddings = model.get_embeddings()
    metric=evaluate_embeddings(embeddings)
    print(metric)
    show(metric)