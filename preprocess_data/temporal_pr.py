import warnings
warnings.filterwarnings("ignore")
import os
import sys
import heapq
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances

# Set the working directory to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # this might cause issue
sys.path.append(project_root)

from TPR import temporal_pagerank


# Function to convert results to dictionary
def get_pagerank_scores(tpr_computer):
    scores = {}
    for i, node in enumerate(tpr_computer.active_mass[:, 0]):
        scores[node] = tpr_computer.temp_pr[i, 1]
    return scores


def get_temporal_pagerank(_graph_df, alpha = 0.85, beta = 0.1):
    
    graph_df = _graph_df.copy(deep=True)
    
    # Extract nodes, edges, and timestamps
    edges = graph_df[['u', 'i', 'ts']].values
    nodes = np.unique(edges[:, :2])  # Get unique nodes from edges

    # Temporal PageRank Parameters
    params = temporal_pagerank.TemporalPageRankParams(alpha, beta)

    # Initialize TemporalPageRankComputer
    tpr_computer = temporal_pagerank.TemporalPageRankComputer(nodes, [params])
    
    # Update PageRank Scores
    for edge in edges:
        src, trg, timestamp = edge
        tpr_computer.update((src, trg))
    
    # Get PageRank scores
    page_rank_scores = get_pagerank_scores(tpr_computer)
    return page_rank_scores


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


def calc_inc_timestamp_pagerank_prev(graph):
    # Full vectorization is challenging due to the nature of PageRank computation, which requires iterative convergence checks.
    G = build_graph(graph)
    
    edges = graph[['u', 'i', 'ts']].values
    
    timestamp_pagerank = {}
    current_edges = []
    
    for edge in sorted(edges, key=lambda x: x[2]): # sorted edges by timestamps
        src, trg, ts = edge
        current_edges.append((src, trg))
        # G.add_edge(current_edges)
        
        if ts not in timestamp_pagerank:
            G.add_edges_from(current_edges)
            timestamp_pagerank[ts] = temporal_page_rank(G)
    return timestamp_pagerank


# def calc_inc_timestamp_pagerank(graph):
#     G = build_graph(graph)
    
#     edges = graph[['u', 'i', 'ts']].values
#     edges = sorted(edges, key=lambda x: x[2])  # Sort edges by timestamp
    
#     timestamp_pagerank = {}
#     current_edges = []

#     timestamps = np.unique(edges[:, 2])
#     for ts in timestamps:
#         # Filter edges up to the current timestamp
#         edges_up_to_ts = edges[edges[:, 2] <= ts]
#         current_edges.extend(edges_up_to_ts)

#         # Add new edges to the graph
#         G.add_edges_from((src, trg) for src, trg, _ in edges_up_to_ts)

#         # Compute PageRank for the current state of the graph
#         timestamp_pagerank[ts] = temporal_page_rank(G)
    
#     return timestamp_pagerank


def calc_inc_timestamp_pagerank_2(graph):
    G = build_graph(graph)
    
    # Initialize edges as a NumPy array
    edges = np.array(graph[['u', 'i', 'ts']].values)
    edges = sorted(edges, key=lambda x: x[2])  # Sort edges by timestamp
    
    timestamp_pagerank = {}
    current_edges = []

    # No need to check if edges is a list; it should remain a NumPy array
    timestamps = np.unique(graph['ts'].values)  # np.unique(edges[:, 2])
    for ts in timestamps:
        # Find indices where the timestamp column is less than or equal to ts
        indices = np.where(edges[:, 2] <= ts)[0]

        # Select rows from edges using these indices
        edges_up_to_ts = edges[indices]
        
        # Filter edges up to the current timestamp
        # edges_up_to_ts = edges[edges[:, 2] <= ts]
        
        current_edges.extend((src, trg) for src, trg, _ in edges_up_to_ts)

        # Add new edges to the graph
        G.add_edges_from((src, trg) for src, trg, _ in edges_up_to_ts)

        # Compute PageRank for the current state of the graph
        timestamp_pagerank[ts] = temporal_page_rank(G)
    
    return timestamp_pagerank

def calc_inc_timestamp_pagerank_3(df):
    G = build_graph(df)
    
    # Convert DataFrame columns to appropriate types if necessary
    df['u'] = df['u'].astype(int)  # Assuming 'u' needs to be integer
    df['i'] = df['i'].astype(int)  # Assuming 'i' needs to be integer
    # df['ts'] = pd.to_datetime(df['ts'])  # Assuming 'ts' is a datetime

    # Sort DataFrame by timestamp
    df.sort_values(by='ts', inplace=True)
    
    timestamp_pagerank = {}
    current_edges = []

    timestamps = df['ts'].unique()  # Get unique timestamps
    for ts in timestamps:
        # Filter DataFrame rows up to the current timestamp
        df_up_to_ts = df[df['ts'] <= ts]
        current_edges.extend(list(zip(df_up_to_ts['u'], df_up_to_ts['i'])))

        # Add new edges to the graph
        G.add_edges_from(current_edges)

        # Compute PageRank for the current state of the graph
        timestamp_pagerank[ts] = temporal_page_rank(G)
    
    return timestamp_pagerank



def calc_inc_timestamp_pagerank_slower(df):
    G = build_graph(df)
    
    # # Convert DataFrame columns to appropriate types if necessary
    # df['u'] = df['u'].astype(int)  # Assuming 'u' needs to be integer
    # df['i'] = df['i'].astype(int)  # Assuming 'i' needs to be integer
    # df['ts'] = pd.to_datetime(df['ts'])  # Assuming 'ts' is a datetime

    # Sort DataFrame by timestamp
    df.sort_values(by='ts', inplace=True)
    
    timestamp_pagerank = {}
    current_edges = []

    timestamps = df['ts'].unique()  # Get unique timestamps

    # Wrap the loop with tqdm for progress visualization
    for ts in tqdm(timestamps, desc="Calculating PageRank"):
        # Filter DataFrame rows up to the current timestamp
        df_up_to_ts = df[df['ts'] <= ts]
        current_edges.extend(list(zip(df_up_to_ts['u'], df_up_to_ts['i'])))

        # Add new edges to the graph
        G.add_edges_from(current_edges)

        # Compute PageRank for the current state of the graph
        timestamp_pagerank[ts] = temporal_page_rank(G)
    
    return timestamp_pagerank


def calc_inc_timestamp_pagerank(df):
    G = build_graph(df)
    
    # Convert DataFrame columns to appropriate types if necessary
    df['u'] = df['u'].astype(int)
    df['i'] = df['i'].astype(int)
    # df['ts'] = pd.to_datetime(df['ts'])

    # Sort DataFrame by timestamp
    df.sort_values(by='ts', inplace=True)
    
    timestamp_pagerank = {}
    current_edges = []

    timestamps = df['ts'].unique()

    # Preallocate memory for edge lists to avoid resizing in each iteration
    max_edges = len(df)
    current_edges = [[] for _ in range(max_edges)]

    # Wrap the loop with tqdm for progress visualization
    for i, ts in enumerate(tqdm(timestamps, desc="Calculating PageRank")):
        # Filter DataFrame rows up to the current timestamp
        df_up_to_ts = df[df['ts'] <= ts]
        # Update current_edges in place to avoid creating new lists
        current_edges[i] = list(zip(df_up_to_ts['u'], df_up_to_ts['i']))

        # Add new edges to the graph in batches
        G.add_edges_from(sum(current_edges[:i+1], []))

        # Compute PageRank for the current state of the graph
        timestamp_pagerank[ts] = temporal_page_rank(G)
    
    return timestamp_pagerank


def optimized_temporal_page_rank(G, alpha=0.85, max_iter=100, tol=1e-6):
    num_nodes = G.number_of_nodes()
    pr = np.full(num_nodes, 1.0 / num_nodes)
    adj_matrix = nx.adjacency_matrix(G).T.tocsr()  # Use nx.adjacency_matrix to get the adjacency matrix

    for _ in range(max_iter):
        pr_old = pr.copy()
        # Perform the dot product and scale by alpha
        scaled_dot_product = alpha * adj_matrix.dot(pr)
        # Add the teleportation factor
        pr = scaled_dot_product + ((1 - alpha) / num_nodes)
        change = np.sum(np.abs(pr - pr_old))

        if change < tol:
            break

    return pr

def optimized_calc_inc_timestamp_pagerank(graph):
    G = build_graph(graph)
    edges = np.array(graph[['u', 'i', 'ts']].values)
    # Ensure edges_sorted_by_ts remains a NumPy array after sorting
    edges_sorted_by_ts = np.array(sorted(edges, key=lambda x: x[2]))

    timestamp_pagerank = {}
    current_edges = []
    unique_timestamps = np.unique(edges_sorted_by_ts[:, 2])

    for ts in tqdm(unique_timestamps, desc="Calculating PageRank"):
        edges_up_to_ts = edges_sorted_by_ts[edges_sorted_by_ts[:, 2] <= ts]
        current_edges.extend((src, trg) for src, trg, _ in edges_up_to_ts)
        G.add_edges_from(current_edges)

        timestamp_pagerank[ts] = optimized_temporal_page_rank(G)
        current_edges.clear()  # Clear current_edges for the next timestamp

    return timestamp_pagerank

# def optimized_calc_inc_timestamp_pagerank(graph):
#     G = build_graph(graph)
#     edges = np.array(graph[['u', 'i', 'ts']].values)
#     # Ensure edges_sorted_by_ts remains a NumPy array after sorting
#     edges_sorted_by_ts = np.array(sorted(edges, key=lambda x: x[2]))

#     timestamp_pagerank = {}
#     unique_timestamps = np.unique(edges_sorted_by_ts[:, 2])

#     for ts in tqdm(unique_timestamps, desc="Calculating PageRank"):
#         # edges_up_to_ts = edges_sorted_by_ts[edges_sorted_by_ts[:, 2] == ts] # its upto that ts
#         edges_up_to_ts = edges_sorted_by_ts[edges_sorted_by_ts[:, 2] <= ts]
#         current_edges = [(src, trg) for src, trg, _ in edges_up_to_ts]
#         G.add_edges_from(current_edges)

#         timestamp_pagerank[ts] = optimized_temporal_page_rank(G)

#     return timestamp_pagerank


def temporal_pagerank_heap_np(E, beta, alpha, check_evolution=False):
    print('\t inside tpr heap method')
    # Convert edges to a NumPy array
    E = np.array(E, dtype=[('u', int), ('v', int), ('t', float)])
    
    # Get the maximum node index to size the r and s arrays appropriately
    max_node = max(E['u'].max(), E['v'].max())
    
    # Initialize r and s arrays
    r = np.zeros(max_node + 1)
    s = np.zeros(max_node + 1)
    
    ts_tpr = [] if check_evolution else None
    
    # Use a heap to efficiently process edges in time order
    heap = [(t, u, v) for u, v, t in E]
    heapq.heapify(heap)
    print('\t heapify successful')
    while heap:
        t, u, v = heapq.heappop(heap)
        
        # Update r and s values
        delta = 1 - alpha
        r[u] += delta
        s[u] += delta
        r[v] += s[u] * alpha
        
        if beta < 1:
            s_v_increment = s[u] * (1 - beta) * alpha
            s[v] += s_v_increment
            s[u] *= beta
        else:
            s[v] += s[u] * alpha
            s[u] = 0
        
        # Store evolution if required
        if check_evolution:
            # ts_tpr.append((t, r.copy()))  # Store r values at current timestamp
            # Normalize r before appending
            total_r = r.sum()
            if total_r > 0:
                ts_tpr.append((t, r.copy() / total_r))
    print('\t out of loop.')
    
    # Normalize r
    total_r = r.sum()
    if total_r > 0:
        r /= total_r
    
    if check_evolution:
        ts_tpr = np.array(ts_tpr, dtype=[('t', float), ('r', float, max_node + 1)])
    
    return r, ts_tpr


def mean_shift_removal(_graph_df, beta = 0.85, alpha = 0.15):
    print('in mss removal method....')
    graph_df = _graph_df.copy(deep=True)
    
    # Extract nodes, edges, and timestamps
    edges = graph_df[['u', 'i', 'ts']].values
    # nodes = np.unique(edges[:, :2])  # Get unique nodes from edges

    # Convert E to a more readable format if needed
    edges_new = [(int(u), int(v), float(t)) for u, v, t in edges]
    print('running temporal page rank method...')
    _, ts_tpr= temporal_pagerank_heap_np(edges_new, beta, alpha, True)
    print('Done.')
    
    print('sorting started....')
    sorted(ts_tpr, key=lambda x: x[0])
    print('sorting Completed.')
    

    # Extract timestamps and PageRank values
    timestamps, pagerank_arrays = zip(*ts_tpr)
    timestamps = np.array(timestamps)
    pagerank_arrays = np.array(pagerank_arrays)

    # Calculate mean shifts between consecutive timestamps
    mean_shifts = []
    print('Before calc')
    for i in tqdm(range(1, len(timestamps)), desc='running mean shift'):
        # if i % 100 ==0:
        # print('calculating mean shift....')
        prev_pagerank = pagerank_arrays[i - 1]
        curr_pagerank = pagerank_arrays[i]
        mean_shift = np.mean(np.abs(curr_pagerank - prev_pagerank))
        mean_shifts.append((timestamps[i], mean_shift))

    # Identify timesteps with highest mean shift
    print('Done.')
    mean_shifts.sort(key=lambda x: x[1], reverse=True)
    
    print('mean shift calc done....')
    
    return mean_shifts


def mean_shift_removal2(_graph_df, beta = 0.85, alpha = 0.15):
    print('in mss removal method....')
    graph_df = _graph_df.copy(deep=True)
    
    # Extract nodes, edges, and timestamps
    edges = graph_df[['u', 'i', 'ts']].values
    # nodes = np.unique(edges[:, :2])  # Get unique nodes from edges

    # Convert E to a more readable format if needed
    edges_new = [(int(u), int(v), float(t)) for u, v, t in edges]
    print('running temporal page rank method...')
    _, ts_tpr= temporal_pagerank_heap_np(edges_new, beta, alpha, True)
    print('Done.')
    
    print('sorting started....')
    sorted(ts_tpr, key=lambda x: x[0])
    print('sorting Completed.')
    

    # Extract timestamps and PageRank values
    timestamps, pagerank_arrays = zip(*ts_tpr)
    timestamps = np.array(timestamps)
    pagerank_arrays = np.array(pagerank_arrays)

    # Calculate mean shifts between consecutive timestamps
    mean_shifts = []
    print('Before calc')
    for i in tqdm(range(1, len(timestamps)), desc='running mean shift'):
        # if i % 100 ==0:
        # print('calculating mean shift....')
        prev_pagerank = np.mean(pagerank_arrays[i - 1])
        curr_pagerank = np.mean(pagerank_arrays[i])
        mean_shift = np.mean(np.abs(curr_pagerank - prev_pagerank))
        mean_shifts.append((timestamps[i], mean_shift))

    # Identify timesteps with highest mean shift
    print('Done.')
    mean_shifts.sort(key=lambda x: x[1], reverse=True)
    
    print('mean shift calc done....')
    
    return mean_shifts


def compute_mean_shifts_with_metrics(graph_df, beta=0.85, alpha=0.15, metric='mean_shift'):
    """
    Compute mean shifts and distance metrics between consecutive PageRank vectors 
    in a continuous time dynamic graph.
    
    Parameters:
        graph_df (pd.DataFrame): DataFrame containing the graph edges with columns ['u', 'i', 'ts'].
        beta (float): Damping factor for PageRank.
        alpha (float): Alpha factor for temporal PageRank.
        metric (str): Metric to use for mean shift ('mean_shift', 'euclidean', 'jaccard', 'cosine').
    
    Returns:
        list: List of tuples containing timestamp and the selected metric.
    """
    print('Starting mean shift and metrics computation...')
    
    # Extract edges from DataFrame
    edges = graph_df[['u', 'i', 'ts']].values
    edges_new = [(int(u), int(v), float(t)) for u, v, t in edges]
    
    print('Running temporal PageRank computation...')
    _, ts_tpr = temporal_pagerank_heap_np(edges_new, beta, alpha, check_evolution=True)
    print('Temporal PageRank computation completed.')
    
    print('Sorting timestamps...')
    sorted(ts_tpr, key=lambda x: x[0])    
    print('Sorting completed.')
    
    # Extract timestamps and PageRank values
    timestamps, pagerank_arrays = zip(*ts_tpr)
    timestamps = np.array(timestamps)
    pagerank_arrays = np.array(pagerank_arrays)
    
    # Calculate mean shifts and distance metrics between consecutive timestamps
    results = []
    print('Calculating mean shifts and distance metrics...')
    for i in tqdm(range(1, len(timestamps)), desc=f'Processing for {metric}...'):
        prev_pagerank = np.array(pagerank_arrays[i - 1])
        curr_pagerank = np.array(pagerank_arrays[i])
        
        if metric == 'mean_shift':
            # Mean Shift
            value = np.mean(np.abs(curr_pagerank - prev_pagerank))
        elif metric == 'euclidean':
            # Euclidean Distance
            value = np.linalg.norm(curr_pagerank - prev_pagerank)
        elif metric == 'jaccard':
            # Jaccard Distance
            # value = pairwise_distances([curr_pagerank], [prev_pagerank], metric='jaccard')[0][0]
            value = pairwise_distances(curr_pagerank.reshape(1, -1), prev_pagerank.reshape(1, -1), metric='jaccard')[0][0]
        elif metric == 'cosine':
            # Cosine Similarity
            value = np.dot(curr_pagerank, prev_pagerank) / (np.linalg.norm(curr_pagerank) * np.linalg.norm(prev_pagerank))
        else:
            raise ValueError(f"Unsupported metric: {metric} should be in mean_shift, euclidean, jaccard, cosine.")
        
        results.append((timestamps[i], value))
    
    print('Mean shifts and metrics calculation completed.')
    
    # Sort results by the selected metric in descending order if applicable
    if metric != 'cosine':  # Cosine similarity might not need descending order
        results.sort(key=lambda x: x[1], reverse=True)
    print('Sorting by the selected metric completed.')
    
    return results
    
    
    



