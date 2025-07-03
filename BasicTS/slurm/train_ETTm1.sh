#!/bin/bash
#SBATCH --job-name=Train
#  SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-vram-48gb
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=90G
#SBATCH --cpus-per-task=10

source ~/.bashrc  # Ensure Conda is properly initialized
#conda init
conda activate BasicTS
Dataset="ETTm1"

# Model="DLinear"
# python ~/Probabilistic_LTSF/BasicTS/experiments/train.py -c final_weights/${Dataset}/${Model}/quantile/${Dataset}_prob_quantile.py --gpus '0'
# python ~/Probabilistic_LTSF/BasicTS/experiments/train.py -c final_weights/${Dataset}/${Model}/i_quantile/${Dataset}_prob_quantile.py --gpus '0'
# python ~/Probabilistic_LTSF/BasicTS/experiments/train.py -c final_weights/${Dataset}/${Model}/univariate/${Dataset}_prob.py --gpus '0'
# python ~/Probabilistic_LTSF/BasicTS/experiments/train.py -c final_weights/${Dataset}/${Model}/multivariate/${Dataset}_prob.py --gpus '0'

Model="DeepAR"
# python ~/Probabilistic_LTSF/BasicTS/experiments/train.py -c final_weights/${Dataset}/${Model}/quantile/${Dataset}_prob_quantile.py --gpus '0'
python ~/Probabilistic_LTSF/BasicTS/experiments/train.py -c final_weights/${Dataset}/${Model}/i_quantile/${Dataset}_prob_quantile.py --gpus '0'
# python ~/Probabilistic_LTSF/BasicTS/experiments/train.py -c final_weights/${Dataset}/${Model}/univariate/${Dataset}_prob.py --gpus '0'

# Model="PatchTST"
# python ~/Probabilistic_LTSF/BasicTS/experiments/train.py -c final_weights/${Dataset}/${Model}/quantile/${Dataset}_prob_quantile.py --gpus '0'
# python ~/Probabilistic_LTSF/BasicTS/experiments/train.py -c final_weights/${Dataset}/${Model}/i_quantile/${Dataset}_prob_quantile.py --gpus '0'
# python ~/Probabilistic_LTSF/BasicTS/experiments/train.py -c final_weights/${Dataset}/${Model}/univariate/${Dataset}_prob.py --gpus '0'
# python ~/Probabilistic_LTSF/BasicTS/experiments/train.py -c final_weights/${Dataset}/${Model}/multivariate/${Dataset}_prob.py --gpus '0'