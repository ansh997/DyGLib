#!/bin/bash
#SBATCH -A research
#SBATCH -n 9
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/home2/hmnshpl/projects/results/DygLib/mooc_LP_model_save.txt
#SBATCH --nodelist gnode082
#SBATCH --time=96:00:00
#SBATCH --mail-user=himanshu.pal@research.iiit.ac.in
#SBATCH --mail-type=ALL


source ~/.bashrc

conda activate tg

python train_link_prediction.py --dataset_name mooc --model_name DyGFormer --load_best_configs --num_runs 5 --gpu 0
python train_link_prediction.py --dataset_name mooc --model_name TGN --load_best_configs --num_runs 5 --gpu 0
python train_link_prediction.py --dataset_name mooc --model_name JODIE --load_best_configs --num_runs 5 --gpu 0
python train_link_prediction.py --dataset_name mooc --model_name TGAT --load_best_configs --num_runs 5 --gpu 0