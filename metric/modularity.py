import networkx as nx
import collections

# https://www.e-learn.cn/content/qita/2346706
# 展开式理解  模块度的大小定义为社区内部的总边数和网络中总边数的比例减去一个期望值，
# 该期望值是将网络设定为随机网络时同样的社区分配 所形成的社区内部的总边数和网络中总边数的比例的大小

# 对于有向图而言，Q = 1/M (遍历社区 ( Avw - kv_out * kw_in /M)) if (v,w) in the same community
# 也是算期望，可以理解为v有kv_out个入口，w有kw_in个出口，随机情况下从v到w存在边的概率为 kv_out * kw_in /M
# 不同于无向图 M = m(边数) Avw 也是表示 从v 到 w是否存在边 或者边权多少

# https://blog.csdn.net/marywbrown/article/details/62059231 新的参考，内部边记得乘2
def cal_Q2(partition,G):
    m = len(G.edges(None, False))
    a = []
    e = []
    if len(partition) == 1:
        return 0
    for community in partition:
        t = 0.0
        for node in community:
            t += len(list(G.neighbors(node)))
        a.append(t/(2*m))
        
    for community in partition:
        community = list(community)
        t = 0.0
        for i in range(len(list(community))):
            for j in range(len(list(community))):
                if(G.has_edge(community[i], community[j])):
                    t += 1.0
        e.append(t/(2*m))

    q = 0.0
    for ei,ai in zip(e,a):
        # 按展开式 e>0 一定存在不同v w,使得Avw=1  kvkw/2m一定小于1,因此Q一定为正
        # e = 0 时 说明是个孤立点社区，不参与计算
        if ei > 0:
            q += (ei - ai**2)
        
    
    return q


def cal_Q(partition, G):
    m = len(G.edges(None, False))
    a = []
    e = []

    if len(partition) == 1:
        return 0

    for community in partition:
        community = list(community)
        t1 = 0.0
        t2 = 0.0
        # Nc^2 => Nc*k
        community_nodes = set(community)
        for node in community:
            t1 += len(list(G.neighbors(node)))
            for nei in G.neighbors(node):
                if nei in community_nodes:
                    t2 += 1.0
        a.append(t1 / (2 * m)) # inner
        e.append(t2 / (2 * m)) # intra

    q = 0.0
    # print(partition)
    # print(e,a)
    for ei, ai in zip(e, a):
        # 按展开式 e>0 一定存在不同v w,使得Avw=1  kvkw/2m一定小于1,因此Q一定为正
        # e = 0 时 说明是个孤立点社区，不参与计算
        if ei > 0:
            q += (ei - ai ** 2)

    return q


def cal_EQ(partition, G):
    # kkm,rc = cal_KKM_RC(partition,G)
    # return 1000-rc
    m = G.number_of_edges()

    a = []
    e = []

    if len(partition) == 1:
        return 0

    node_label_count = collections.defaultdict(int)
    for community in partition:
        for node in community:
            node_label_count[node] +=1

    for community in partition:
        community = list(community)
        t1 = 0.0
        t2 = 0.0
        # Nc^2 => Nc*k
        community_nodes = set(community)
        for node in community:
            if node in G:
                t1 += len(G[node]) / node_label_count[node]
                for nei in G[node]:
                    if nei in community_nodes:
                        t2 += 1.0/(node_label_count[nei] * node_label_count[node])


        a.append(t1 / (2 * m))
        e.append(t2 / (2 * m))

    q = 0.0
    for ei, ai in zip(e, a):
        # 按展开式 e>0 一定存在不同v w,使得Avw=1  kvkw/2m一定小于1,因此Q一定为正
        # e = 0 时 说明是个孤立点社区，不参与计算
        # print(ei,ai)
        if ei > 0:
            q += (ei - ai ** 2)

    return q

def read_cmu(path):
    cmus = []
    # cnt =0
    with open(path, 'r') as f:
        for row in f.readlines():
            row = row.strip()
            r = row.split(" ",-1)
            cmus.append(r)
            # cnt += len(r)
    # print(cnt)
    return cmus


# qinze
def cal_Q3(communities, G):
    '''
        Reference: Community detection in graphs, equation (15)
        :param G: networkx graph object
        :return standard modularity
    '''
    Q = 0
    m = nx.number_of_edges(G)
    for com in communities:
        dc = 0
        lc = 0
        for i in com:
            dc += len(G[i])
            for j in G[i]:
                if j in com:
                    lc += 1
        lc /= 2.0  # for undirected graph
        Q += lc / m - (dc / (2.0 * m)) ** 2
    return Q


def cal_EQ_qinze(cover, G):
    vertex_community = collections.defaultdict(lambda: set())
    for i, c in enumerate(cover):
        for v in c:
            vertex_community[v].add(i)

    m = 0.0
    for v in G.nodes():
        neighbors = G.neighbors(v)
        for n in neighbors:
            if v > n:
                m += 1
    total = 0.0
    for c in cover:
        for i in c:
            o_i = len(vertex_community[i])
            k_i = len(G[i])
            for j in c:
                o_j = len(vertex_community[j])
                k_j = len(G[j])
                if i > j:
                    continue
                t = 0.0
                if j in G[i]:
                    t += 1.0 / (o_i * o_j)
                t -= k_i * k_j / (2 * m * o_i * o_j)
                if i == j:
                    total += t
                else:
                    total += 2 * t

    return round(total / (2 * m), 4)

def cal_EQ2(communities, G):
    '''
        Reference:Community detection in graphs, equation (39)
        :param G: networkx graph object
        :return overlapping modularity
    '''
    v_com = collections.defaultdict(lambda: set())
    for i, com in enumerate(communities):
        for v in com:
            v_com[v].add(i)
    EQ_1 = 0
    EQ_2 = 0
    m = nx.number_of_edges(G)
    for com in communities:
        s_2 = 0
        for i in com:
            k_i = len(G[i])
            o_i = len(v_com[i])
            s_i = 0
            for j in G[i]:
                if j in com:
                    o_j = len(v_com[j])
                    s_i += 1.0 / o_j
            EQ_1 += s_i / o_i
            s_2 += 1.0 * k_i / o_i
        EQ_2 += s_2 ** 2
    EQ = (EQ_1 - EQ_2 / (2.0 * m)) / (2.0 * m)
    return EQ



def cal_KKM_RC(partitions,G):

    comm_num = len(partitions)
    intra_score = 0
    inter_score = 0
    n = G.number_of_nodes()
    for partition in partitions:
        partition = set(partition)
        intra = 0
        inter = 0
        for node in partition:
            # if node not in G:
                # continue
            for nei in G.neighbors(node):
                if nei in partition:
                    intra += 1
                else:
                    inter += 1
        intra_score += intra / len(partition)
        inter_score += inter / len(partition)

    KKM = 2*(n - comm_num) - intra_score
    RC = inter_score
    # print(comm_num,n)
    return KKM, RC

def cal_EQ_KKM_RC(partitions,G):
    a = []
    e = []
    n = G.number_of_nodes()
    m = G.number_of_edges()
    intra_score = 0
    inter_score = 0
    comm_num = len(partitions)
    node_label_count = collections.defaultdict(int)
    for community in partitions:
        for node in community:
            node_label_count[node] +=1

    for community in partitions:
        t1 = 0.0
        t2 = 0.0
        community_nodes = set(community)
        intra = 0
        inter = 0
        for node in community_nodes:
            t1 += len(G[node]) / node_label_count[node]
            for nei in G[node]:
                if nei in community_nodes:
                    intra += 1
                    t2 += 1.0/(node_label_count[nei] * node_label_count[node])
                else:
                    inter += 1
        intra_score += intra / len(community_nodes)
        inter_score += inter / len(community_nodes)
        a.append(t1 / (2 * m))
        e.append(t2 / (2 * m))

    q = 0.0
    for ei, ai in zip(e, a):
        if ei > 0:
            q += (ei - ai ** 2)

    KKM = 2*(n - comm_num) - intra_score
    RC = inter_score

    return q,KKM,RC
    # return 1000-RC,KKM,RC


if __name__ == '__main__':
    cmus = read_cmu("./output/res.txt")
    graph = nx.read_edgelist("./data/network.dat")
    print(cal_Q(cmus,graph))