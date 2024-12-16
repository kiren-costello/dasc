import networkx as nx
# todo
# 二部图最大权匹配
def max_weight_matching(G, maxcardinality=False, weight='weight'):
    return nx.algorithms.matching.max_weight_matching(G, maxcardinality=maxcardinality, weight=weight)