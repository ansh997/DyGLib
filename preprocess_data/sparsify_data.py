import os
import getpass
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from pandas.testing import assert_frame_equal
from distutils.dir_util import copy_tree
from preprocess_data import check_data


# scratch_location = r'/scratch/hmnshpl'
scratch_location = rf'/scratch/{getpass.getuser()}'

def EL_sparsify(graph_df, edge_raw_features):
    """_summary_

    Args:
        graph_df (_type_): Original graph df
        edge_raw_features (_type_): Original edge raw features

    Returns:
        _type_: sparsified graph with edge features
    """
    tmp_graph_df = graph_df.copy(deep=True)
    tmp_graph_df = tmp_graph_df.sort_values(by=['u', 'i', 'ts'])
    
    # Group by 'u' and 'i' and capture the first and last interactions
    first_interactions = tmp_graph_df.groupby(['u', 'i']).first().reset_index()
    last_interactions = tmp_graph_df.groupby(['u', 'i']).last().reset_index()
    
    EL_graph_df = pd.concat([first_interactions, last_interactions]).drop_duplicates().reset_index(drop=True).drop(['Unnamed: 0'], axis=1)
    
    EL_edge_raw_features = edge_raw_features # edge_raw_features[EL_graph_df['idx'].values]
    # assert EL_graph_df.shape[0] == EL_edge_raw_features.shape[0]
    
    return EL_graph_df, EL_edge_raw_features


def EL_sparsify_data(dataset_name='wikipedia'):
    # Load data and train val test split
    graph_df = pd.read_csv('{}/processed_data/{}/ml_{}.csv'.format(scratch_location, dataset_name, dataset_name))
    edge_raw_features = np.load('{}/processed_data/{}/ml_{}.npy'.format(scratch_location, dataset_name, dataset_name))
    node_raw_features = np.load('{}/processed_data/{}/ml_{}_node.npy'.format(scratch_location, dataset_name, dataset_name))
    
    OUT_DF = '{}/sparsified_data/{}/ml_{}.csv'.format(scratch_location, dataset_name, dataset_name)
    OUT_FEAT = '{}/sparsified_data/{}/ml_{}.npy'.format(scratch_location, dataset_name, dataset_name)
    OUT_NODE_FEAT = '{}/sparsified_data/{}/ml_{}_node.npy'.format(scratch_location, dataset_name, dataset_name)
    
    EL_graph_df, EL_edge_raw_features = EL_sparsify(graph_df, edge_raw_features)
    
    EL_graph_df.to_csv(OUT_DF)  # edge-list
    np.save(OUT_FEAT, EL_edge_raw_features)  # edge features
    np.save(OUT_NODE_FEAT, node_raw_features)  # node features
    
    print(f'Sparsified {dataset_name}.')
    


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

    
    

