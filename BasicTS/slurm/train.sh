#!/bin/bash
#SBATCH --job-name=myjob-01
#  SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-vram-32gb
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --mem=70G
#SBATCH --cpus-per-task=10

source ~/.bashrc  # Ensure Conda is properly initialized
#conda init
conda activate BasicTS

MODEL_NAME="PatchTST"
DATASET_NAME="ETTh1_prob"

python ~/Probabilistic_LTSF/BasicTS/experiments/train.py -c baselines/${MODEL_NAME}/${DATASET_NAME}.py --gpus '0'