#!/bin/bash

# Define the datasets, sparsification levels, and strategies
# datasets=("mooc" "wikipedia" "uci" "reddit")
# uci - sparsfication not implemented check why
# breaking for reddit

# Reddit
# Wiki [T-Spear]
# MOOC [T-Spear]
# UCI [T-Spear]
# Bitcoin [T-Spear]

datasets=("reddit")  # uci also left
uptos=(0.7 0.8 0.9)
strategies=(
    # "ts_tpr_remove_MSS"
    # "ts_tpr_remove_cosine"
    # "ts_tpr_remove_euclidean"
    # "ts_tpr_remove_jaccard"
    # "ts_tpr_remove_wasserstein"
    # "ts_tpr_remove_kl_divergence"
    # "ts_tpr_remove_chebyshev"
    # "ts_tpr_remove_jensen_shannon_divergence"
    # "ts_tpr_remove_TER"
    # "ts_tpr_remove_mss_2"
    "ts_tpr_remove_combined_ter"
)

# Iterate over each dataset
for dataset in "${datasets[@]}"; do
    # Iterate over each sparsification level
    for upto in "${uptos[@]}"; do
        # Iterate over each strategy
        for strategy in "${strategies[@]}"; do
            # Run the script with the specified arguments
            python preprocess_data/sparsify_data.py --dataset_name "$dataset" --upto "$upto" --strategy "$strategy"
        done
    done
done
