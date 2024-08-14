#!/bin/bash
#SBATCH -A research
#SBATCH -n 9
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/home2/hmnshpl/projects/results/DygLib/Link_Prediciton_ts_tpr_remove_MSS_0.8.txt
#SBATCH --nodelist gnode074
#SBATCH --time=96:00:00
#SBATCH --mail-user=himanshu.pal@research.iiit.ac.in
#SBATCH --mail-type=ALL

source ~/.bashrc

conda activate tg

python train_link_prediction.py --dataset_name wikipedia --model_name TGN --patch_size 2 --max_input_sequence_length 64 --num_runs 5 --gpu 0 --sparsify True --strategy ts_tpr_remove_MSS --sampling_upto 0.8
    