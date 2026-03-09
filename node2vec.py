import numpy as np
import networkx as nx
from gensim.models import Word2Vec


class Node2Vec:
    """Node2Vec图嵌入"""

    def __init__(self, graph, dimensions=128, walk_length=80, num_walks=10,
                 p=1, q=1, workers=1):
        self.graph = graph
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.workers = workers
        self.walks = []

    def _preprocess_transition_probs(self):
        """预处理转移概率"""
        G = self.graph
        nodes = list(G.nodes())
        self.alias_nodes = {}
        self.alias_edges = {}

        for node in nodes:
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            self.alias_nodes[node] = self._alias_setup(normalized_probs)

            for idx, nbr in enumerate(sorted(G.neighbors(node))):
                alias_edge = self._get_alias_edge(node, nbr)
                self.alias_edges[(node, nbr)] = alias_edge

    def _alias_setup(self, probs):
        """Alias采样设置"""
        K = len(probs)
        q = np.zeros(K)
        J = np.zeros(K, dtype=int)

        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            q[kk] = K * prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()
            J[small] = large
            q[large] = q[large] - (1.0 - q[small])

            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        return J, q

    def _get_alias_edge(self, src, dst):
        """获取边的alias采样"""
        G = self.graph
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / q)

        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return self._alias_setup(normalized_probs)

    def _alias_draw(self, J, q):
        """Alias采样"""
        K = len(J)
        kk = int(np.floor(np.random.rand() * K))
        if np.random.rand() < q[kk]:
            return kk
        else:
            return J[kk]

    def _simulate_walks(self, nodes):
        """模拟随机游走"""
        G = self.graph
        walks = []

        for walk_iter in range(self.num_walks):
            for node in nodes:
                walk = [node]
                if len(list(G.neighbors(node))) == 0:
                    walks.append(walk)
                    continue

                while len(walk) < self.walk_length:
                    cur = walk[-1]
                    cur_nbrs = sorted(G.neighbors(cur))
                    if len(cur_nbrs) > 0:
                        if len(walk) == 1:
                            walk.append(cur_nbrs[self._alias_draw(
                                self.alias_nodes[cur][0], self.alias_nodes[cur][1])])
                        else:
                            prev = walk[-2]
                            next_node = cur_nbrs[self._alias_draw(
                                self.alias_edges[(prev, cur)][0],
                                self.alias_edges[(prev, cur)][1])]
                            walk.append(next_node)
                    else:
                        break

                walks.append(walk)

        return walks

    def fit(self):
        """训练Node2Vec"""
        G = self.graph
        self._preprocess_transition_probs()
        nodes = list(G.nodes())
        self.walks = self._simulate_walks(nodes)

        walks = [list(map(str, walk)) for walk in self.walks]
        model = Word2Vec(walks, vector_size=self.dimensions, window=10,
                         min_count=0, sg=1, workers=self.workers)
        self.embedding = np.array([model.wv[str(node)] for node in nodes])

    def get_embedding(self):
        """获取嵌入向量"""
        return self.embedding


def node2vec_embedding(adj_matrix, dimensions=64): 
    """对邻接矩阵进行Node2Vec嵌入，只返回 row 节点部分的嵌入"""
    import networkx as nx
    from node2vec import Node2Vec

    G = nx.Graph()

    n_row = adj_matrix.shape[0]
    n_col = adj_matrix.shape[1]

    # 添加节点
    G.add_nodes_from([(i, {'type': 'row'}) for i in range(n_row)])
    G.add_nodes_from([(j + n_row, {'type': 'col'}) for j in range(n_col)])

    # 添加边
    for i in range(n_row):
        for j in range(n_col):
            weight = adj_matrix[i, j]
            if weight > 0:
                G.add_weighted_edges_from([(i, j + n_row, weight)])

    # 运行Node2Vec
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=100,
                        num_walks=10, p=1, q=1, workers=2)
    node2vec.fit()
    embeddings = node2vec.get_embedding()

    return embeddings[:n_row]