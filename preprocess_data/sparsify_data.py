import os
import sys
import getpass
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
# from pandas.testing import assert_frame_equal
from distutils.dir_util import copy_tree
from .preprocess_data import check_data
from .temporal_pr import temporal_pagerank_with_timestamps, calc_timestamp_pagerank,\
    calc_inc_timestamp_pagerank, optimized_calc_inc_timestamp_pagerank,\
    get_temporal_pagerank, mean_shift_removal, mean_shift_removal2, compute_mean_shifts_with_metrics, calculate_temporal_edge_rank,\
    calculate_combined_temporal_edgerank
import networkx as nx

# Set the working directory to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # this might cause issue
sys.path.append(project_root)

# scratch_location = r'/scratch/hmnshpl'
scratch_location = rf'/scratch/{getpass.getuser()}'

def EL_sparsify(graph, edge_raw_features, strategy='random', upto=0.7, dataset_name=''):
    """_summary_

    Args:
        graph (_type_): Original graph df
        edge_raw_features (_type_): Original edge raw features

    Returns:
        _type_: sparsified graph with edge features
    """
    if dataset_name == '':
        raise ValueError('Please pass a dataset name.')
    
    strategy = strategy.lower() # making it case insensitive
    tmp_graph = graph.copy(deep=True)
    tmp_graph = tmp_graph.sort_values(by=['u', 'i', 'ts'])
    
    # Exclude the first and last rows based on 'u' and 'i'
    # grouped = tmp_graph.groupby(['u', 'i'])
    modified_df = tmp_graph.copy(deep=True)  # grouped.apply(lambda x: x.iloc[1:-1]).reset_index(drop=True)
    
    sample_size = int(len(modified_df) * upto)
    
    # # Group by 'u' and 'i' and capture the first and last interactions
    # first_interactions = grouped.first().reset_index()
    # last_interactions = grouped.last().reset_index()
    
    # TODO: add random interactions --> 10% - 30%
    # we can do different selection strategy - right now random only - always keep strategy in small caps here
    match strategy:
        case 'random':
            # Randomly sample rows without replacement
            sampled_df = modified_df.sample(n=sample_size, random_state=42)
        case 'tpr_remove':
            # calculate page rank of a dataset
            # sample upto given percentage to be removed - since we are rejecting hence it should be different than selecting
            # use top % nodes and remove (1-upto) nodes
            # naive method
            # graph = build_graph(tmp_graph)
            # page_rank_scores = temporal_page_rank(graph)
            # Official method
            page_rank_scores = get_temporal_pagerank(tmp_graph)
            
            # Sort nodes by PageRank scores
            sorted_nodes = sorted(page_rank_scores.items(), key=lambda item: item[1], reverse=True)
            # Calculate the top upto% of nodes
            top_x_percent_count = int(len(sorted_nodes) * (1-upto))
            top_x_percent_nodes = sorted_nodes[:top_x_percent_count]
            # Extract the node IDs from the top 30 percent nodes
            top_x_percent_node_ids = {node for node, _ in top_x_percent_nodes}

            # Remove rows from graph_df where either source or target node is in the top 30% nodes
            sampled_df = modified_df[~modified_df['u'].isin(top_x_percent_node_ids) & ~modified_df['i'].isin(top_x_percent_node_ids)]
        case 'ts_tpr_remove_ss':
            # calculate timestamp level tpr
            # ts_level_tpr = temporal_pagerank_with_timestamps(tmp_graph)
            
            ts_level_tpr = calc_timestamp_pagerank(tmp_graph)  # snapshot implementation
            
            ts_aggregated_scores= {}
            for ts, scores in ts_level_tpr.items():
                agg_scores=sum(scores.values())
                ts_aggregated_scores[ts]=agg_scores
            
            # Sort timestamps by aggregated PageRank scores
            sorted_timestamps = sorted(ts_aggregated_scores.items(),
                                    key=lambda item: item[1], reverse=True)
            
            top_x_percent_count = int(len(sorted_timestamps) *(1-upto))
            
            top_x_percent_timestamps = [timestamp for timestamp, _ in sorted_timestamps[:top_x_percent_count]]
            
            sampled_df = modified_df[~modified_df['ts'].isin(top_x_percent_timestamps)]  # should we keep full training data - as we are already dropping duplicates
        case 'ts_tpr_remove_inc':
            # incremental 
            # ts_level_tpr = calc_inc_timestamp_pagerank(tmp_graph)  # incremental implementation
            ts_level_tpr = optimized_calc_inc_timestamp_pagerank(tmp_graph)  # incremental implementation
            
            ts_aggregated_scores= {}
            for ts, scores in ts_level_tpr.items():
                agg_scores=sum(scores.values()) # what different aggregation can I try here?
                ts_aggregated_scores[ts]=agg_scores
            
            # Sort timestamps by aggregated PageRank scores
            sorted_timestamps = sorted(ts_aggregated_scores.items(),
                                    key=lambda item: item[1], reverse=True)
            
            top_x_percent_count = int(len(sorted_timestamps) *(1-upto))
            
            top_x_percent_timestamps = [timestamp for timestamp, _ in sorted_timestamps[:top_x_percent_count]]
            
            sampled_df = modified_df[~modified_df['ts'].isin(top_x_percent_timestamps)]
        case 'ts_tpr_remove_mss':  # problem started from here
            # based in maximum mean shift strategy
            metric = 'mss'
            filename = f'{scratch_location}/sparsified_data/{dataset_name}_{metric}_sparsified_{upto}.csv'
            if os.path.exists(filename):
                print(f'reading {filename}...', end=' ')
                sampled_df=pd.read_csv(filename)
                sampled_df = sampled_df.loc[:, ~sampled_df.columns.str.contains('^Unnamed')]
                print(' done')
            else:
                mean_shifts = mean_shift_removal(tmp_graph)
            
                print('back to sparsify_data file....')
                
                threshold_index = int(len(mean_shifts) * (1-upto))
                top_mean_shifts = mean_shifts[:threshold_index]
                
                top_x_percent_timestamps = [ts for ts, _ in top_mean_shifts]
                
                sampled_df = modified_df[~modified_df['ts'].isin(top_x_percent_timestamps)]
                
                sampled_df.drop(['Unnamed: 0'], axis=1).to_csv(filename)
            print('data sampling successful.')
        case 'ts_tpr_remove_mss_2':
            # based in maximum mean shift strategy
            metric = 'mss2'
            filename = f'{scratch_location}/sparsified_data/{dataset_name}_{metric}_sparsified_{upto}.csv'
            if os.path.exists(filename):
                print(f'reading {filename}...', end=' ')
                sampled_df=pd.read_csv(filename)
                sampled_df = sampled_df.loc[:, ~sampled_df.columns.str.contains('^Unnamed')]
                print(' done')
            else:
                mean_shifts = mean_shift_removal2(tmp_graph)
            
                print('back to sparsify_data file....')
                
                threshold_index = int(len(mean_shifts) * (1-upto))
                top_mean_shifts = mean_shifts[:threshold_index]
                
                top_x_percent_timestamps = [ts for ts, _ in top_mean_shifts]
                
                sampled_df = modified_df[~modified_df['ts'].isin(top_x_percent_timestamps)]
                sampled_df.drop(['Unnamed: 0'], axis=1).to_csv(filename)
            print('data sampling successful.')
        case 'ts_tpr_remove_cosine':
            # based on maximum mean shift strategy
            metric = 'cosine'
            filename = f'{scratch_location}/sparsified_data/{dataset_name}_{metric}_sparsified_{upto}.csv'
            if os.path.exists(filename):
                print(f'reading {filename}...', end=' ')
                sampled_df=pd.read_csv(filename)
                sampled_df = sampled_df.loc[:, ~sampled_df.columns.str.contains('^Unnamed')]
                print(' done')
            else:
                mean_shifts = compute_mean_shifts_with_metrics(tmp_graph, metric='cosine')
                
                print('back to sparsify_data file....')
                
                threshold_index = int(len(mean_shifts) * (1-upto))
                top_mean_shifts = mean_shifts[:threshold_index]
                
                top_x_percent_timestamps = [ts for ts, _ in top_mean_shifts]
                
                sampled_df = modified_df[~modified_df['ts'].isin(top_x_percent_timestamps)]
                sampled_df.drop(['Unnamed: 0'], axis=1).to_csv(filename)
                print(filename, ' saved.')
            print('data sampling successful.')
        case 'ts_tpr_remove_euclidean':
            # based on maximum mean shift strategy
            metric = 'euclidean'
            filename = f'{scratch_location}/sparsified_data/{dataset_name}_{metric}_sparsified_{upto}.csv'
            if os.path.exists(filename):
                print(f'reading {filename}...', end=' ')
                sampled_df=pd.read_csv(filename)
                sampled_df = sampled_df.loc[:, ~sampled_df.columns.str.contains('^Unnamed')]
                print(' done')
            else:
                mean_shifts = compute_mean_shifts_with_metrics(tmp_graph, metric='euclidean')
                
                print('back to sparsify_data file....')
                
                threshold_index = int(len(mean_shifts) * (1-upto))
                top_mean_shifts = mean_shifts[:threshold_index]
                
                top_x_percent_timestamps = [ts for ts, _ in top_mean_shifts]
                
                sampled_df = modified_df[~modified_df['ts'].isin(top_x_percent_timestamps)]
                sampled_df.drop(['Unnamed: 0'], axis=1).to_csv(filename)
                print(filename, ' saved.')
            print('data sampling successful.')
        case 'ts_tpr_remove_jaccard':
            # based on maximum mean shift strategy
            metric = 'jaccard'
            filename = f'{scratch_location}/sparsified_data/{dataset_name}_{metric}_sparsified_{upto}.csv'
            if os.path.exists(filename):
                print(f'reading {filename}...', end=' ')
                sampled_df=pd.read_csv(filename)
                sampled_df = sampled_df.loc[:, ~sampled_df.columns.str.contains('^Unnamed')]
                print(' done')
            else:
                mean_shifts = compute_mean_shifts_with_metrics(tmp_graph, metric='jaccard')
            
                print('back to sparsify_data file....')
                
                threshold_index = int(len(mean_shifts) * (1-upto))
                top_mean_shifts = mean_shifts[:threshold_index]
                
                top_x_percent_timestamps = [ts for ts, _ in top_mean_shifts]
                
                sampled_df = modified_df[~modified_df['ts'].isin(top_x_percent_timestamps)]
                sampled_df.drop(['Unnamed: 0'], axis=1).to_csv(filename)
                
            print('data sampling successful.')
        case 'ts_tpr_remove_wasserstein':
            # based on maximum mean shift strategy
            metric = 'wasserstein'
            filename = f'{scratch_location}/sparsified_data/{dataset_name}_{metric}_sparsified_{upto}.csv'
            if os.path.exists(filename):
                print(f'reading {filename}...', end=' ')
                sampled_df=pd.read_csv(filename)
                sampled_df = sampled_df.loc[:, ~sampled_df.columns.str.contains('^Unnamed')]
                print(' done')
            else:
                mean_shifts = compute_mean_shifts_with_metrics(tmp_graph, metric='wasserstein')
            
                print('back to sparsify_data file....')
                
                threshold_index = int(len(mean_shifts) * (1-upto))
                top_mean_shifts = mean_shifts[:threshold_index]
                
                top_x_percent_timestamps = [ts for ts, _ in top_mean_shifts]
                
                sampled_df = modified_df[~modified_df['ts'].isin(top_x_percent_timestamps)]
                sampled_df.drop(['Unnamed: 0'], axis=1).to_csv(filename)
            print('data sampling successful.')
        case 'ts_tpr_remove_kl_divergence':
            # based on maximum mean shift strategy
            metric = 'kl_divergence'
            filename = f'{scratch_location}/sparsified_data/{dataset_name}_{metric}_sparsified_{upto}.csv'
            if os.path.exists(filename):
                print(f'reading {filename}...', end=' ')
                sampled_df=pd.read_csv(filename)
                sampled_df = sampled_df.loc[:, ~sampled_df.columns.str.contains('^Unnamed')]
                print(' done')
            else:
                mean_shifts = compute_mean_shifts_with_metrics(tmp_graph, metric=metric)
            
                print('back to sparsify_data file....')
                
                threshold_index = int(len(mean_shifts) * (1-upto))
                top_mean_shifts = mean_shifts[:threshold_index]
                
                top_x_percent_timestamps = [ts for ts, _ in top_mean_shifts]
                
                sampled_df = modified_df[~modified_df['ts'].isin(top_x_percent_timestamps)]
                sampled_df.drop(['Unnamed: 0'], axis=1).to_csv(filename)
            print('data sampling successful.')
        case 'ts_tpr_remove_jensen_shannon_divergence':
            # based on maximum mean shift strategy
            metric = 'jensen_shannon_divergence'
            filename = f'{scratch_location}/sparsified_data/{dataset_name}_{metric}_sparsified_{upto}.csv'
            if os.path.exists(filename):
                print(f'reading {filename}...', end=' ')
                sampled_df=pd.read_csv(filename)
                sampled_df = sampled_df.loc[:, ~sampled_df.columns.str.contains('^Unnamed')]
                print(' done')
            else:
                mean_shifts = compute_mean_shifts_with_metrics(tmp_graph, metric=metric)
            
                print('back to sparsify_data file....')
                
                threshold_index = int(len(mean_shifts) * (1-upto))
                top_mean_shifts = mean_shifts[:threshold_index]
                
                top_x_percent_timestamps = [ts for ts, _ in top_mean_shifts]
                
                sampled_df = modified_df[~modified_df['ts'].isin(top_x_percent_timestamps)]
                sampled_df.drop(['Unnamed: 0'], axis=1).to_csv(filename)
            print('data sampling successful.')
        case 'ts_tpr_remove_chebyshev':
            # based on maximum mean shift strategy
            metric = 'chebyshev'
            filename = f'{scratch_location}/sparsified_data/{dataset_name}_{metric}_sparsified_{upto}.csv'
            if os.path.exists(filename):
                print(f'reading {filename}...', end=' ')
                sampled_df=pd.read_csv(filename)
                sampled_df = sampled_df.loc[:, ~sampled_df.columns.str.contains('^Unnamed')]
                print(' done')
            else:
                mean_shifts = compute_mean_shifts_with_metrics(tmp_graph, metric=metric)
            
                print('back to sparsify_data file....')
                
                threshold_index = int(len(mean_shifts) * (1-upto))
                top_mean_shifts = mean_shifts[:threshold_index]
                
                top_x_percent_timestamps = [ts for ts, _ in top_mean_shifts]
                
                sampled_df = modified_df[~modified_df['ts'].isin(top_x_percent_timestamps)]
                sampled_df.drop(['Unnamed: 0'], axis=1).to_csv(filename)
            print('data sampling successful.')
        case 'ts_tpr_remove_ter':
            # based on maximum mean shift strategy
            metric = 'TER'
            filename = f'{scratch_location}/sparsified_data/{dataset_name}_{metric}_sparsified_{upto}.csv'
            if os.path.exists(filename):
                print(f'reading {filename}...', end=' ')
                sampled_df=pd.read_csv(filename)
                sampled_df = sampled_df.loc[:, ~sampled_df.columns.str.contains('^Unnamed')]
                print(' done')
            else:
                ter_dict = calculate_temporal_edge_rank(tmp_graph)
                
                sorted_ter_dict = dict(sorted(ter_dict.items(), key=lambda x: x[1], reverse=True))
                sorted_ter_dict = list(sorted_ter_dict.items())
            
                print('back to sparsify_data file....')

                threshold_index = int(len(sorted_ter_dict) * (1-upto))

                top_mean_shifts = sorted_ter_dict[:threshold_index]

                top_x_percent_timestamps = [ts for ts, _ in top_mean_shifts]

                sampled_df = tmp_graph[~tmp_graph['ts'].isin(top_x_percent_timestamps)]
                sampled_df.drop(['Unnamed: 0'], axis=1).to_csv(filename)
            print('data sampling successful.')
        case 'ts_tpr_remove_combined_ter':
            metric = 'Combined_TER'
            filename = f'{scratch_location}/sparsified_data/{dataset_name}_{metric}_sparsified_{upto}.csv'
            if os.path.exists(filename):
                print(f'reading {filename}...', end=' ')
                sampled_df=pd.read_csv(filename)
                sampled_df = sampled_df.loc[:, ~sampled_df.columns.str.contains('^Unnamed')]
                print(' done')
            else:
                ter_dict = calculate_combined_temporal_edgerank(tmp_graph)
                
                sorted_ter_dict = dict(sorted(ter_dict.items(), key=lambda x: x[1], reverse=True))
                sorted_ter_dict = list(sorted_ter_dict.items())
            
                print('back to sparsify_data file....')

                threshold_index = int(len(sorted_ter_dict) * (1-upto))

                top_mean_shifts = sorted_ter_dict[:threshold_index]

                top_x_percent_timestamps = [ts for ts, _ in top_mean_shifts]

                sampled_df = tmp_graph[~tmp_graph['ts'].isin(top_x_percent_timestamps)]
                sampled_df.drop(['Unnamed: 0'], axis=1).to_csv(filename)
            print('data sampling successful.')
        case _:
            raise ValueError(f'Unknown strategy {strategy}')
    # TODO: concat only for random sparsification strategy
    EL_graph = (pd.concat([
        # first_interactions,
        sampled_df,
        # last_interactions
        ])
        .drop_duplicates()
        .reset_index(drop=True)
        # .drop(['Unnamed: 0'], axis=1)
        .sort_values(['idx'], ascending=True) # this fixed it.
                )
    
    EL_edge_raw_features = edge_raw_features # edge_raw_features[sorted(EL_graph['idx'].values)]
    # EL_graph['idx'] = [i for i in range(1, len(EL_graph['idx'])+1)]
    # assert EL_graph.shape[0] == EL_edge_raw_features.shape[0]
    
    return EL_graph, EL_edge_raw_features


def EL_sparsify_data(dataset_name='wikipedia'):
    # Load data and train val test split
    graph = pd.read_csv('{}/processed_data/{}/ml_{}.csv'.format(scratch_location, dataset_name, dataset_name))
    edge_raw_features = np.load('{}/processed_data/{}/ml_{}.npy'.format(scratch_location, dataset_name, dataset_name))
    node_raw_features = np.load('{}/processed_data/{}/ml_{}_node.npy'.format(scratch_location, dataset_name, dataset_name))
    
    OUT_DF = '{}/sparsified_data/{}/ml_{}.csv'.format(scratch_location, dataset_name, dataset_name)
    OUT_FEAT = '{}/sparsified_data/{}/ml_{}.npy'.format(scratch_location, dataset_name, dataset_name)
    OUT_NODE_FEAT = '{}/sparsified_data/{}/ml_{}_node.npy'.format(scratch_location, dataset_name, dataset_name)
    
    EL_graph, EL_edge_raw_features = EL_sparsify(graph, edge_raw_features)
    
    EL_graph.to_csv(OUT_DF)  # edge-list
    np.save(OUT_FEAT, EL_edge_raw_features)  # edge features
    np.save(OUT_NODE_FEAT, node_raw_features)  # node features
    
    print(f'Sparsified {dataset_name}.')
    

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


def build_graph(graph):
    # Extract nodes, edges, and timestamps
    edges = graph[['u', 'i', 'ts']].values
    
    G = nx.Graph()
    for edge in edges:
        source, target, timestamp = edge
        G.add_edge(source, target, timestamp=timestamp)
    return G


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Interface for preprocessing datasets')
    parser.add_argument('--dataset_name', type=str,
                        choices=['wikipedia', 'reddit', 'mooc', 'lastfm', 'myket', 'enron', 'SocialEvo', 'uci',
                                'Flights', 'CanParl', 'USLegis', 'UNtrade', 'UNvote', 'Contacts'],
                        help='Dataset name', default='wikipedia')
    parser.add_argument('--node_feat_dim', type=int, default=172, help='Number of node raw features')

    args = parser.parse_args()

    print(f'Sparsifying dataset {args.dataset_name}...')
    if args.dataset_name in ['enron', 'SocialEvo', 'uci']:
        Path("{}/sparsified_data/{}/".format(scratch_location, args.dataset_name)).mkdir(parents=True, exist_ok=True)
        copy_tree("{}/DG_data/{}/".format(scratch_location, args.dataset_name), "{}/sparsified_data/{}/".format(scratch_location, args.dataset_name))
        print(f'Not implemented for enron, SocialEvo, uci graph yet.')
    else:
        Path("{}/sparsified_data/{}/".format(scratch_location, args.dataset_name)).mkdir(parents=True, exist_ok=True)
        copy_tree("{}/DG_data/{}/".format(scratch_location, args.dataset_name), "{}/sparsified_data/{}/".format(scratch_location, args.dataset_name))
        # bipartite dataset
        if args.dataset_name in ['wikipedia', 'reddit', 'mooc', 'lastfm', 'myket']:
            EL_sparsify_data(dataset_name=args.dataset_name)
        else:
            EL_sparsify_data(dataset_name=args.dataset_name)
        print(f'{args.dataset_name} is processed successfully.')

        if args.dataset_name not in ['myket']:
            check_data(args.dataset_name)
        print(f'{args.dataset_name} passes the checks successfully.')
