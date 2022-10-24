import gerrychain as gc
from gerrychain import Graph, Partition, Election
import lupartition
import networkx as nx 
from typing import Hashable
from lupartition import Mode, partition_all, partition, decision
import numpy as np
import pandoc
import random

#return a uniform spanning tree
def uniform_spanning_tree(graph: nx.Graph): 
    unvisited_vertices = list(graph.nodes())
    ust = nx.create_empty_copy(graph)
    first_vtx = random.sample(unvisited_vertices, 1)[0]
    unvisited_vertices.remove(first_vtx)
    while len(unvisited_vertices) > 0: 
        start_vtx = random.sample(unvisited_vertices, 1)[0]
        current_vtx = start_vtx
        path = [current_vtx]
        while current_vtx in unvisited_vertices: 
            next_vtx = random.sample(list(nx.neighbors(graph, current_vtx)), 1)[0]
            if next_vtx in path: 
                i = path.index(next_vtx) 
                path = path[:i]
            else:
                path.append(next_vtx)
            current_vtx = next_vtx
        for i in range(len(path)-1): 
            ust.add_edge(path[i], path[i+1]) 
            unvisited_vertices.remove(path[i])
    return ust

#count number of spanning trees of a graph
def num_spanning_trees(graph:nx.Graph): 
    return np.round(np.prod(nx.laplacian_spectrum(graph)[1:])/nx.number_of_nodes(graph))

#a function to generate ensembles for a given graph
def generate_ensemble(graph: nx.Graph, 
                      key: Hashable, 
                      parts: int, 
                      lower: float, 
                      upper: float, 
                      size: int):
    ensemble = []
    for i in range(size): 
        tree = random_spanning_tree(graph)
        partitions = partition_all(tree, key, parts, lower, upper)
        for part in partitions: 
            ensemble.append(Partition(graph, part))
    return ensemble
