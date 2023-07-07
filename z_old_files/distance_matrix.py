import pickle
import numpy as np
import pandas as pd
import networkx as nx
import torch
from multiprocessing import Pool

def initialize_graph():
    coordinates = pd.read_csv("vienna_dist.csv", header=None, sep=' ')
    coordinates.columns = ['coord1', 'coord2', 'dist']
    graph = nx.DiGraph()

    for _, row in coordinates.iterrows():
        graph.add_edge(row['coord1'], row['coord2'], weight=row['dist'])

    return graph

def compute_distance(graph, node_id1, node_ids2):
    distances = []
    for i, id1 in enumerate(node_id1):
        row = []
        for j, id2 in enumerate(node_ids2):
            shortest_path = nx.shortest_path(graph, id1, id2)
            shortest_path_distance = sum(graph.get_edge_data(u, v)['weight'] for u, v in zip(shortest_path, shortest_path[1:]))
            row.append(shortest_path_distance)
        distances.append(row)
        if id1%100 == 0:
            print('precomputed distance till id: {}'.format(id1))
    return distances

def precompute_distance_matrix(graph, node_ids, save_path):
    num_nodes = len(node_ids)

    # Use multiprocessing to parallelize distance computation
    with Pool() as pool:
        result = pool.starmap(compute_distance, [(graph, [id1], node_ids) for id1 in node_ids])

    # Convert result to numpy array
    print(np.array(result).shape)
    distance_matrix = np.array(result).reshape((num_nodes, num_nodes, 1))

    # Save distance matrix as a pickle file
    with open(save_path, 'wb') as f:
        pickle.dump(distance_matrix, f)

    return distance_matrix
