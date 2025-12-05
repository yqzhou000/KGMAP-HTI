# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
from tqdm import tqdm


def node2vec_embedding(adj_matrix, embedding_dim=64, walk_length=80, 
                      num_walks=10, p=1, q=1, workers=4, window=10, 
                      min_count=0, batch_words=4):
    print("Generating Node2Vec embeddings...")
    
    if hasattr(adj_matrix, 'tocsr'):
        adj_matrix = adj_matrix.tocsr()
    
    num_nodes = adj_matrix.shape[0]
    G = nx.Graph()
    
    if hasattr(adj_matrix, 'nonzero'):
        rows, cols = adj_matrix.nonzero()
        for i, j in zip(rows, cols):
            if i < j:
                G.add_edge(i, j, weight=adj_matrix[i, j])
    else:
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if adj_matrix[i, j] > 0:
                    G.add_edge(i, j, weight=adj_matrix[i, j])
    
    print(f"Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    walks = []
    nodes = list(G.nodes())
    
    print("Generating random walks...")
    for _ in tqdm(range(num_walks), desc="Walk iteration"):
        np.random.shuffle(nodes)
        for node in nodes:
            walk = [node]
            while len(walk) < walk_length:
                cur = walk[-1]
                neighbors = list(G.neighbors(cur))
                if len(neighbors) > 0:
                    if len(walk) == 1:
                        walk.append(np.random.choice(neighbors))
                    else:
                        prev = walk[-2]
                        probs = []
                        for neighbor in neighbors:
                            if neighbor == prev:
                                probs.append(1.0 / p)
                            elif G.has_edge(neighbor, prev):
                                probs.append(1.0)
                            else:
                                probs.append(1.0 / q)
                        probs = np.array(probs)
                        probs = probs / probs.sum()
                        walk.append(np.random.choice(neighbors, p=probs))
                else:
                    break
            walks.append([str(n) for n in walk])
    
    print(f"Generated {len(walks)} walks")
    
    print("Training Word2Vec model...")
    model = Word2Vec(walks, vector_size=embedding_dim, window=window,
                    min_count=min_count, sg=1, workers=workers,
                    epochs=10, batch_words=batch_words)
    
    embeddings = np.zeros((num_nodes, embedding_dim))
    for i in range(num_nodes):
        if str(i) in model.wv:
            embeddings[i] = model.wv[str(i)]
        else:
            embeddings[i] = np.random.randn(embedding_dim) * 0.01
    
    print("Node2Vec embeddings generated successfully")
    
    return embeddings


def generate_node_features(num_nodes, feature_dim, method='random'):
    if method == 'random':
        features = np.random.randn(num_nodes, feature_dim).astype(np.float32)
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
    elif method == 'one_hot':
        features = np.eye(num_nodes, feature_dim, dtype=np.float32)
    elif method == 'identity':
        features = np.eye(num_nodes, dtype=np.float32)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return features
