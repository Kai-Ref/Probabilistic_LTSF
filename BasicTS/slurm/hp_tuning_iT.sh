#!/bin/bash
#SBATCH --job-name=HPO2
#  SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-vram-48gb
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=90G
#SBATCH --cpus-per-task=10

source ~/.bashrc  # Ensure Conda is properly initialized
#conda init
conda activate BasicTS

# prediction length and input length
# OUTPUT_LEN=100


# prediction length and input length
# OUTPUT_LEN=100

DATASET_NAME="ETTm1_prob"
# PatchTST
# MODEL_NAME="PatchTST"

# iTransformer
# MODEL_NAME="iTransformer"


# DeepAR
# MODEL_NAME="DeepAR"
# SweepID="dax8rkzl" #- multivariate 720

MODEL_NAME="DLinear"
SweepID="dsgresbt" #- multivariate 720

# python ~/Probabilistic_LTSF/BasicTS/experiments/hp_tuning.py -c baselines/${MODEL_NAME}/${DATASET_NAME}.py -s /home/kreffert/Probabilistic_LTSF/BasicTS/hp_tuning/${MODEL_NAME}.yaml --gpus '0'
python ~/Probabilistic_LTSF/BasicTS/experiments/hp_tuning.py -c baselines/${MODEL_NAME}/${DATASET_NAME}.py -s ${SweepID} --gpus '0'

# python ~/Probabilistic_LTSF/BasicTS/experiments/hp_tuning.py -c baselines/PatchTST/ETTh1_prob.py -s 'fd40u7me' --gpus '0, 1'
# python ~/Probabilistic_LTSF/BasicTS/experiments/train.py -c baselines/PatchTST/ETTh1_prob.py --gpus '0'
# python ~/Probabilistic_LTSF/BasicTS/experiments/train.py -c final_weights/PatchTST/univariate/ETTh1.py --gpus '0'