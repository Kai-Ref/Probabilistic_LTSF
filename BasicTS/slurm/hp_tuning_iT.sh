#!/bin/bash
#SBATCH --job-name=HPO2
#  SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-vram-48gb
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=70G
#SBATCH --cpus-per-task=10

source ~/.bashrc  # Ensure Conda is properly initialized
#conda init
conda activate BasicTS

# prediction length and input length
# OUTPUT_LEN=100


DATASET_NAME="ETTh1_prob"
# PatchTST
# MODEL_NAME="PatchTST"
# regular settings - 96
# SweepID='4j6sm6dg' # -96 -> maybe have to do that again because Scaler/Filler params did not adjust?
# SweepID='6mxvo502' # - 192
# SweepID='' # - 336
# SweepID='' # - 720

# iTransformer
MODEL_NAME="iTransformer"
# regular settings - 96
# SweepID='' # - 96
SweepID='1o1erq16' # - 192
# SweepID='' # - 336
# SweepID='' # - 720




# python ~/Probabilistic_LTSF/BasicTS/experiments/hp_tuning.py -c baselines/${MODEL_NAME}/${DATASET_NAME}.py -s /home/kreffert/Probabilistic_LTSF/BasicTS/hp_tuning/${MODEL_NAME}.yaml --gpus '0'
python ~/Probabilistic_LTSF/BasicTS/experiments/hp_tuning.py -c baselines/${MODEL_NAME}/${DATASET_NAME}.py -s ${SweepID} --gpus '0'


# python ~/Probabilistic_LTSF/BasicTS/experiments/hp_tuning.py -c baselines/PatchTST/ETTh1_prob.py -s '1o1erq16' --gpus '0, 1'
# python ~/Probabilistic_LTSF/BasicTS/experiments/train.py -c baselines/PatchTST/ETTh1_prob.py --gpus '0'