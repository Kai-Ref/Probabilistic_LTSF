#!/bin/bash
#SBATCH --job-name=HPO
#  SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-vram-48gb
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=70G
#SBATCH --cpus-per-task=10

source ~/.bashrc  # Ensure Conda is properly initialized
#conda init
conda activate BasicTS

MODEL_NAME="PatchTST"
DATASET_NAME="ETTh1_prob"
# prediction length and input length
# OUTPUT_LEN=100


# python ~/Probabilistic_LTSF/BasicTS/experiments/hp_tuning.py -c baselines/${MODEL_NAME}/${DATASET_NAME}.py -s /home/kreffert/Probabilistic_LTSF/BasicTS/hp_tuning/${MODEL_NAME}.yaml --gpus '0'
python ~/Probabilistic_LTSF/BasicTS/experiments/hp_tuning.py -c baselines/${MODEL_NAME}/${DATASET_NAME}.py -s fwvzje2j --gpus '0'