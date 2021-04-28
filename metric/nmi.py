# coding=utf-8
# author: linsher
# date: 20201203
import numpy as np
import math
from collections import defaultdict

def calc(A, B, method='fast' ,ignore=True):
    """
    :param A: parties A
    :param B: parties B
    :param method:
    :param ignore:
    :return:
    """
    if method == 'fast' : return NMI_fast(A,B)
    A_label = []
    p_id = 0
    n_id = 0
    hash = defaultdict(int)
    for p in A:
        for i in p:
            if i in hash:
                print ('warning: node {} belong to multiple communities In party A, Please use ONMI rather NMI!'.format(i))
            hash[i] = n_id
            n_id += 1
            A_label.append(p_id)
        p_id += 1

    A_label = np.array(A_label)
    B_label = np.array(A_label)
    p_id = 0
    n2_id = 0
    B_hash = defaultdict(int)
    for p in B:
        for i in p:
            if i in B_hash:
                print ('warning: node {} belong to multiple communities In party B, Please use ONMI rather NMI!'.format(i))
            B_hash[i] = 1
            n2_id += 1
            B_label[hash[i]] = p_id
        p_id += 1

    if ignore:
        if n_id == len(A) or n2_id == len(B):
            return 0
    if method == 'matrix' : return NMI_byMatrix(A_label, B_label)
    elif method == 'iter' : return NMI_byIter(A_label,B_label)
    else : raise Exception('unknown method')

def NMI_byIter(A,B):
    """
    :param A: labels of nodes set A
    :param B: labels of nodes set B
    :return:
    """
    if len(A) != len( B):
        raise Exception('numbers of nodes in party A and B must be equal')

    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A==idA)
            idBOccur = np.where(B==idB)
            idABOccur = np.intersect1d(idAOccur,idBOccur)
            px = 1.0*len(idAOccur[0])/total
            py = 1.0*len(idBOccur[0])/total
            pxy = 1.0*len(idABOccur)/total
            MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0*len(np.where(A==idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0*len(np.where(B==idB)[0])
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
    MIhat = 2.0*MI/(Hx+Hy)
    return MIhat

def NMI_byMatrix(A,B):
    """
    :param A: labels of nodes set A
    :param B: labels of nodes set B
    :return:
    """
    if len(A) != len( B):
        raise Exception('numbers of nodes in party A and B must be equal')

    eps = 1.4e-45
    N = len(A)
    A = np.array(A).reshape(1,-1)
    B = np.array(B).reshape(1,-1)

    A_ids = np.unique(A).reshape(1, -1)
    B_ids = np.unique(B).reshape(1, -1)
    idAOccur = np.tile(A, (A_ids.shape[1] , 1) ) == np.tile( A_ids.T, (1, N))
    idBOccur = np.tile(B, (B_ids.shape[1] , 1) ) == np.tile( B_ids.T, (1, N))
    # calculate the number of nodes in different communities A and B, and in intersection AB respectively.
    idAOccur = idAOccur.astype(int)
    idBOccur = idBOccur.astype(int)
    idABOccur = idAOccur.dot(idBOccur.T)
    Px = idAOccur.T.sum(0) / N
    Py = idBOccur.T.sum(0) / N
    Pxy = idABOccur / N

    Px = Px.reshape(1,-1)
    Py = Py.reshape(1,-1)

    # calculate mutual information and entropy respectively.
    MImatrix = Pxy * np.log2(Pxy / (Px.T.dot(Py))+eps)
    MI = MImatrix.sum()
    Hx = - (Px * np.log2(Px + eps)).sum()
    Hy = - (Py * np.log2(Py + eps)).sum()
    # print(Hx, Hy, MI)
    MIhat = 2 * MI / (Hx + Hy)
    return MIhat

def NMI_byMemberMatrix(M1,M2):
    """
    calculate normalized mutual information based on member matrix.
    :param M1: member matrix 1.
    :param M2: member matrix 2.
    :return:
    """
    if M1.shape[0] != M2.shape[0]:
        raise Exception('numbers of nodes in M1 and M2 must be equal')

    eps = 1.4e-45
    N = M1.shape[0]

    Px = M1.sum(0) / N
    Py = M2.sum(0) / N
    Px = Px.reshape(1,-1)
    Py = Py.reshape(1,-1)

    Hx = - (Px * np.log2(Px + eps)).sum()
    Hy = - (Py * np.log2(Py + eps)).sum()

    Pxy = M1.T.dot(M2) / N
    MImatrix = Pxy * np.log2(Pxy / (Px.T.dot(Py))+eps)
    MI = MImatrix.sum()
    # print(Hx,Hy,MI)
    MIhat = 2 * MI / (Hx + Hy)
    return MIhat




def NMI_fast(A, B):

    all_nodes_in_A = [x for i in A for x in i]
    filter_A = np.unique(all_nodes_in_A)

    all_nodes_in_B = [x for i in B for x in i]
    filter_B = np.unique(all_nodes_in_B)

    n1 = len(filter_A)
    n2 = len(filter_B)

    if n1 < len(all_nodes_in_A) :
        print ('warning: some nodes in community are not exist in feature file or edge file in Party A')

    if n2 < len(all_nodes_in_B) :
        print ('warning: some nodes in community are not exist in feature file or edge file in Party B')

    if n1 > len(all_nodes_in_A) :
        print ('warning: nodes may belong to multiple communities in Party A, please use ONMI rather than NMI!')

    if n2 > len(all_nodes_in_B) :
        print('warning: nodes may belong to multiple communities in Party B, please use ONMI rather than NMI!')

    # n1 = sum([len(i) for i in A])
    # n2 = sum([len(i) for i in B])

    if n1 != n2:
        print ('warning: numbers of nodes in party A and B are not equal')

    eps = 1e-45
    H12 = 0
    for i in A:
        for j in B:
            ni = len(i)
            nj = len(j)
            nij = len(set(i).intersection(set(j)))
            H12 += nij*math.log(nij*n1/(ni*nj)+eps,2)

    H1 = 0
    for i in A:
        ni = len(i)
        H1 += ni*math.log(ni/n1+eps,2)

    H2 = 0
    for j in B:
        nj = len(j)
        H2 += nj*math.log(nj/n1+eps,2)

    # A_ = [len(i) for i in A]
    # ndA = np.array(A_).reshape(1,-1)
    # H1 = ndA * np.log2(ndA / n1 + eps)
    # H1 = H1.sum()

    # B_ = [len(i) for i in B]
    # ndB = np.array(B_).reshape(1,-1)
    # H2 = ndB * np.log2(ndB / n2 + eps)
    # H2 = H2.sum()

    MIhat = -2*H12/(H1+H2+eps)
    # MIhat = -  H12 / (min(H1,H2) + eps)
    return MIhat