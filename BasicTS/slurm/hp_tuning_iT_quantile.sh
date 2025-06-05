#!/bin/bash
#SBATCH --job-name=QU
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

DATASET_NAME="ETTm1_prob_quantile"
# PatchTST
# MODEL_NAME="PatchTST"

# iTransformer
# MODEL_NAME="iTransformer"
# DeepAR
MODEL_NAME="DeepAR"
SweepID="xjl3tqcr" #- quantile 720

# MODEL_NAME="DLinear"
# SweepID="cyk9i3vt" #- quantile 720



# python ~/Probabilistic_LTSF/BasicTS/experiments/hp_tuning.py -c baselines/${MODEL_NAME}/${DATASET_NAME}.py -s /home/kreffert/Probabilistic_LTSF/BasicTS/hp_tuning/${MODEL_NAME}_quantile.yaml --gpus '0'
python ~/Probabilistic_LTSF/BasicTS/experiments/hp_tuning.py -c baselines/${MODEL_NAME}/${DATASET_NAME}.py -s ${SweepID} --gpus '0'

# python ~/Probabilistic_LTSF/BasicTS/experiments/hp_tuning.py -c baselines/PatchTST/ETTh1_prob.py -s 'fd40u7me' --gpus '0, 1'
# python ~/Probabilistic_LTSF/BasicTS/experiments/train.py -c baselines/PatchTST/ETTh1_prob.py --gpus '0'
# python ~/Probabilistic_LTSF/BasicTS/experiments/train.py -c final_weights/PatchTST/univariate/ETTh1_prob_quantile.py --gpus '0'


# python ~/Probabilistic_LTSF/BasicTS/experiments/evaluate.py -cfg final_weights/PatchTST/univariate/ETTh1_prob.py --gpus '0' -ckpt final_weights/PatchTST/univariate/ETTh1_100_96_720/a8de06edad7530010e0b704422b431a2/PatchTST_best_val_NLL.py