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

DATASET_NAME="ETTh1_prob_quantile"
# PatchTST
MODEL_NAME="PatchTST"
# regular settings - 336
# SweepID='4j6sm6dg' # -96 -> maybe have to do that again because Scaler/Filler params did not adjust?
# SweepID='6mxvo502' # - 192
# SweepID='' # - 336
# SweepID='js2yngzt' # - 720
# -------------------------- IL 96
# SweepID='fd40u7me' # - 720 multivariate
# SweepID='7hg7ll8o' # - 720 univariate 
# SweepID='' # - 720 quantile 


# iTransformer
# MODEL_NAME="iTransformer"
# # regular settings - 336
# SweepID='s86t4dab' # - 96
# SweepID='1o1erq16' # - 192
# SweepID='' # - 336
# SweepID='jlsiwsgq' # - 720

# DeepAR
# MODEL_NAME="DeepAR"
# # regular settings - 96
# SweepID='1ggq3wmn' # - 96
# SweepID='' # - 192
# SweepID='' # - 336
# SweepID='sskubw3c' # - 720

python ~/Probabilistic_LTSF/BasicTS/experiments/hp_tuning.py -c baselines/${MODEL_NAME}/${DATASET_NAME}.py -s /home/kreffert/Probabilistic_LTSF/BasicTS/hp_tuning/${MODEL_NAME}_quantile.yaml --gpus '0'
# python ~/Probabilistic_LTSF/BasicTS/experiments/hp_tuning.py -c baselines/${MODEL_NAME}/${DATASET_NAME}.py -s ${SweepID} --gpus '0'

# python ~/Probabilistic_LTSF/BasicTS/experiments/hp_tuning.py -c baselines/PatchTST/ETTh1_prob.py -s 'fd40u7me' --gpus '0, 1'
# python ~/Probabilistic_LTSF/BasicTS/experiments/train.py -c baselines/PatchTST/ETTh1_prob.py --gpus '0'
# python ~/Probabilistic_LTSF/BasicTS/experiments/train.py -c final_weights/PatchTST/univariate/ETTh1_prob_quantile.py --gpus '0'