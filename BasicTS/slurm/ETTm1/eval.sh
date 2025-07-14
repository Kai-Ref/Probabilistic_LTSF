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
# python ~/Probabilistic_LTSF/notebooks/Final\ plots/run_eval.py
# python ~/Probabilistic_LTSF/notebooks/Final\ plots/run_eval_ETTm1.py --models DLinear --dists m
# python ~/Probabilistic_LTSF/notebooks/Final\ plots/run_eval_ETTm1.py --models PatchTST --dists m
python ~/Probabilistic_LTSF/notebooks/Final\ plots/run_eval_ETTm1.py --models DeepAR --dists iq --seeds 4
# python ~/Probabilistic_LTSF/notebooks/Final\ plots/run_eval_ETTm1.py --models DeepAR --dists iq
