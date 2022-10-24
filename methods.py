import gerrychain as gc
from gerrychain import Graph, Partition, Election
import lupartition
import networkx as nx 
from typing import Hashable
from lupartition import Mode, partition_all, partition, decision
import numpy as np
import pandoc
import random

#==========================================================
#Below are methods to generate ensembles

#a function to return a uniform spanning tree
def uniform_spanning_tree(graph: nx.Graph): 
    'Returns a uniform spanning tree of graph'
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
            path.append(next_vtx)
            current_vtx = next_vtx
        for i in range(len(path)-1): 
            ust.add_edge(path[i], path[i+1]) 
            unvisited_vertices.remove(path[i])
    return ust

#a function to generate ensembles for a given graph
def generate_ensemble(graph: nx.Graph, 
                      key: Hashable, 
                      parts: int, 
                      lower: float, 
                      upper: float, 
                      size: int):
    'Returns an ensemble of balanced partitions of graph. "key" is the weight key for the vertices, "parts" is the number of parts, "lower" and "upper" are the lower and upper bounds of the weight of each part, and "size" is the number of uniform spanning trees whose partitions are added to the ensemble. (Note that size is NOT the number of partitions in the ensemble).'
    ensemble = []
    for i in range(size): 
        tree = uniform_spanning_tree(graph)
        partitions = partition_all(tree, key, parts, lower, upper)
        for part in partitions: 
            ensemble.append(Partition(graph, part))
    return ensemble

#==========================================================
#Below are methods to analyze the ensembles generated 

#count number of spanning trees of a graph
def lognum_spanning_trees(graph:nx.Graph): 
    'Returns the log of the number of spanning trees of graph'
    return np.sum(np.log(nx.laplacian_spectrum(graph)[1:])) - np.log(nx.number_of_nodes(graph))

#find the quotient graph of a partition 
def quotient_graph(partition:gc.Partition): 
    'Returns the quotient graph of partition: the graph with vertex set partition.parts and edges between parts A and B if there exists an edge of the graph with one end in A and one end in B. The edges of the quotient graph are weighted by the number of edges of the graph with one end in A and one end in B.'
    quot = nx.MultiGraph()
    for edge in partition.graph.edges(): 
        u = edge[0]
        v = edge[1]
        quot.add_edge(partition.assignment[u], partition.assignment[v])
    return quot

#find the log of the probability of sampling a partition
def log_prob(partition:gc.Partition): 
    pr = 0
    for comp in partition.subgraphs:
        pr += lognum_spanning_trees(comp)
    quotient = quotient_graph(partition)
    pr += lognum_spanning_trees(quotient)
    return pr - lognum_spanning_trees(graph)
