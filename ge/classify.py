from __future__ import print_function

import numpy
from sklearn.metrics import f1_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


class TopKRanker(OneVsRestClassifier):  # 继承OneVsRestClassifier
    def predict(self, X, top_k_list):
        probs = numpy.asarray(super().predict_proba(
            X))  # Python3.x 和 Python2.x 的一个区别是: Python 3 可以使用直接使用 super().xxx 代替 super(Class, self).xxx
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
        return numpy.asarray(all_labels)


class Classifier(object):

    def __init__(self, embeddings, clf):
        self.embeddings = embeddings
        self.clf = TopKRanker(clf)  # LogisticRegression()
        self.binarizer = MultiLabelBinarizer(sparse_output=True)

    def train(self, X, Y, Y_all):
        """预处理训练节点的嵌入表示和标签的稀疏矩阵表示作为OvR的输入

        :param X: 训练节点
        :param Y: 训练标签
        :param Y_all: 所有标签
        """
        self.binarizer.fit(Y_all)  # fit全部标签
        X_train = [self.embeddings[x] for x in X]
        Y = self.binarizer.transform(Y)  # transform Y
        # 预处理
        self.clf.fit(X_train, Y)

    def evaluate(self, X, Y):
        """

        :param X: 测试集节点
        :param Y: 测试集标签
        :return:
        """
        top_k_list = [len(l) for l in Y]
        Y_ = self.predict(X, top_k_list)
        Y = self.binarizer.transform(Y)
        averages = ["micro", "macro", "samples", "weighted"]
        results = {}
        for average in averages:
            results[average] = f1_score(Y, Y_, average=average)
        results['acc'] = accuracy_score(Y, Y_)
        print('-------------------')
        print(results)
        return results
        print('-------------------')

    def predict(self, X, top_k_list):
        """

        :param X:
        :param top_k_list:
        :return:
        """
        X_ = numpy.asarray([self.embeddings[x] for x in X])
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        return Y

    def split_train_evaluate(self, X, Y, train_percent, seed=0):
        """

        :param X: 节点
        :param Y: 标签
        :param train_percent: 采样率
        :param seed:初始随机种子
        :return:
        """
        state = numpy.random.get_state()

        training_size = int(train_percent * len(X))
        numpy.random.seed(seed)
        shuffle_indices = numpy.random.permutation(numpy.arange(len(X)))  # arange返回均匀间隔值的数组，start=0,stop,step=1,
        # permutation重排列数组
        X_train = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
        X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]
        # 交叉验证
        self.train(X_train, Y_train, Y)
        numpy.random.set_state(state)
        return self.evaluate(X_test, Y_test)


def read_node_label(filename, skip_head=False):
    """对两列文本返回左列X，右列Y

    :param filename:
    :param skip_head:
    :return:
    """
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        if skip_head:
            fin.readline()
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y
