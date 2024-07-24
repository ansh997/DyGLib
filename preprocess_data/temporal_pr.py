import os
import sys
# import pandas as pd
import numpy as np
import networkx as nx

# Set the working directory to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # this might cause issue
sys.path.append(project_root)

from TPR import temporal_pagerank


def build_graph(graph):
    # Extract nodes, edges, and timestamps
    edges = graph[['u', 'i', 'ts']].values
    
    G = nx.Graph()
    for edge in edges:
        source, target, timestamp = edge
        G.add_edge(source, target, timestamp=timestamp)
    return G


def temporal_page_rank(G, alpha=0.85, max_iter=100, tol=1e-6):
    nodes = G.nodes()
    num_nodes = G.number_of_nodes()
    
    # Initialize PageRank scores
    pr = {node: 1.0 / num_nodes for node in nodes}
    temp_pr = pr.copy()

    for _ in range(max_iter):
        change = 0
        for node in nodes:
            rank_sum = sum(pr[neighbor] / len(G[neighbor]) for neighbor in G.neighbors(node) if 'timestamp' in G[node][neighbor])
            temp_pr[node] = (1 - alpha) / num_nodes + alpha * rank_sum
        
        # Calculate change for convergence check
        change = sum(abs(temp_pr[node] - pr[node]) for node in nodes)
        
        if change < tol:
            break
        
        pr = temp_pr.copy()

    return pr


# Function to convert results to dictionary
def get_pagerank_scores(tpr_computer):
    scores = {}
    for i, node in enumerate(tpr_computer.active_mass[:, 0]):
        scores[node] = tpr_computer.temp_pr[i, 1]
    return scores


def temporal_pagerank_with_timestamps(graph, alpha=0.85, beta=0.1):
    
    # Extract nodes, edges, and timestamps
    edges = graph[['u', 'i', 'ts']].values
    nodes = np.unique(edges[:, :2])  # Get unique nodes from edges
    
    params = temporal_pagerank.TemporalPageRankParams(alpha, beta)
    
    # Initialize TemporalPageRankComputer
    tpr_computer = temporal_pagerank.TemporalPageRankComputer(nodes, [params])
    
    # Initialize a dictionary to store PageRank scores for each timestamp
    timestamp_pagerank = {}

    # Update PageRank Scores for each edge at each timestamp
    for edge in edges:
        src, trg, timestamp = edge
        tpr_computer.update((src, trg))
        
        # Store the PageRank scores for the current timestamp
        if timestamp not in timestamp_pagerank:
            timestamp_pagerank[timestamp] = get_pagerank_scores(tpr_computer)
        else:
            timestamp_pagerank[timestamp].update(get_pagerank_scores(tpr_computer))
    
    return timestamp_pagerank


def calc_timestamp_pagerank(graph):
    G = build_graph(graph)
    
    edges = graph[['u', 'i', 'ts']].values
    
    # Calculate Temporal PageRank for each timestamp
    timestamp_pagerank = {}
    for edge in edges:
        _, _, timestamp = edge
        if timestamp not in timestamp_pagerank:
            # Extract subgraph for the current timestamp
            subgraph_edges = [e for e in G.edges(data=True) if e[2]['timestamp'] == timestamp]
            subgraph = nx.Graph()
            subgraph.add_edges_from((e[0], e[1]) for e in subgraph_edges)
            
            # Calculate PageRank for the subgraph
            pr_scores = temporal_page_rank(subgraph)
            timestamp_pagerank[timestamp] = pr_scores
    return timestamp_pagerank
    


