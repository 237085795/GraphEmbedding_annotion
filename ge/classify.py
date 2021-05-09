from __future__ import print_function

import numpy
from sklearn.metrics import f1_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


class TopKRanker(OneVsRestClassifier):  # 继承OneVsRestClassifier  OneVsRestClassifier(LogisticRegression())
    def predict(self, X, top_k_list):
        """重写预测函数，将直接返回类别改为返回one_hot数组

        :param X: (481,128)嵌入数组
        :param top_k_list:[1, 1, 1, 1, 1, 1, 1,...]
        :return:
        """
        probs = numpy.asarray(super().predict_proba(X))
        # 函数传入测试集，predict_proba的返回值是一个矩阵，矩阵的index是对应第几个样本，columns对应第几个标签，矩阵中的数字则是第i个样本的标签为j的概率值。区别于predict直接返回标签值。
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
        #  将概率数组装换成one_hot数组
        return numpy.asarray(all_labels)


class Classifier(object):

    def __init__(self, embeddings, clf):
        self.embeddings = embeddings
        self.clf = TopKRanker(clf)  # LogisticRegression()
        self.binarizer = MultiLabelBinarizer(sparse_output=True)  # 数据预处理器

    def train(self, X, Y, Y_all):
        """预处理训练节点的嵌入表示和标签的稀疏矩阵表示作为OvR的输入，进行训练

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
        """得出性能参数

        :param X: 测试集节点
        :param Y: 测试集标签
        :return:性能参数列表
        """
        # print("Y:")
        # print(Y)
        # Out[1]:[['3'], ['16'], ['5'], ['11'], ['10'], ['1'], ['7']]
        top_k_list = [len(l) for l in Y]
        # print("list:")
        # print(top_k_list)
        # Out[2]:[1, 1, 1, 1, 1, 1, 1]
        Y_ = self.predict(X, top_k_list)  # 预测结果
        Y = self.binarizer.transform(Y)  # 实际标签
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
        """预测节点类别

        :param X:测试集节点
        :param top_k_list:[1, 1, 1, 1, 1, 1, 1]
        :return:预测结果的one_hot数组
        """
        X_ = numpy.asarray([self.embeddings[x] for x in X])  # 转成嵌入表示数组
        # print(X_.shape)
        # Out:(481, 128) 481个节点128维
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        # print(Y.shape)
        # Out:(481, 17) 481个节点17个类别 相当于one_hot
        return Y

    def split_train_evaluate(self, X, Y, train_percent, seed=0):
        """划分训练集测试集_训练_评估

        :param X: 节点
        :param Y: 标签
        :param train_percent: 训练集测试集比率
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
        numpy.random.set_state(state)  # 使随机生成器random保持相同的状态（state）
        return self.evaluate(X_test, Y_test)


def read_node_label(filename, skip_head=False,is_net=False):
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
        if is_net:
            vec = l.strip().split('\t')
        else:
            vec = l.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y
