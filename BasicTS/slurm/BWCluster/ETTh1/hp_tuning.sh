#!/bin/bash
#SBATCH -p gpu_a100_il  # Use the dev_gpu_4_a100 partition with A100 GPUs dev_gpu_4
#SBATCH -n 1                   # Number of tasks (1 for single node)
#SBATCH -t 360           # Time limit (10 minutes for debugging purposes)
#SBATCH --mem=60000             # Memory request (adjust as needed)
#SBATCH --gres=gpu:1           # Request 1 GPU (adjust if you need more)
#SBATCH --cpus-per-task=16     # Number of CPUs per GPU (16 for A100)
#SBATCH --ntasks-per-node=1    # Number of tasks per node (1 in this case)

module load devel/miniforge

conda activate BasicTS

# prediction length and input length
# OUTPUT_LEN=100

DATASET_NAME="ETTh1_prob"
# PatchTST
# MODEL_NAME="PatchTST"

# iTransformer
# MODEL_NAME="iTransformer"


# DeepAR
MODEL_NAME="DeepAR"
SweepID="vgbb0kjf" #- univariate 720

# MODEL_NAME="DLinear"
# SweepID="6q22uza0" # multivariate 720

# python ~/Probabilistic_LTSF/BasicTS/experiments/hp_tuning.py -c baselines/${MODEL_NAME}/${DATASET_NAME}.py -s /home/ma/ma_ma/ma_kreffert/Probabilistic_LTSF/BasicTS/hp_tuning/${MODEL_NAME}.yaml --gpus '0'
python ~/Probabilistic_LTSF/BasicTS/experiments/hp_tuning.py -c baselines/${MODEL_NAME}/${DATASET_NAME}.py -s ${SweepID} --gpus '0'


# python ~/Probabilistic_LTSF/BasicTS/experiments/hp_tuning.py -c baselines/PatchTST/ETTh1_prob.py -s '6mxvo502' --gpus '0, 1'
# python ~/Probabilistic_LTSF/BasicTS/experiments/train.py -c baselines/PatchTST/ETTh1_prob.py --gpus '0'