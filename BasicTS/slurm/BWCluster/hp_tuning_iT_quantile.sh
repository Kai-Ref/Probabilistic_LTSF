#!/bin/bash
#SBATCH -p gpu_a100_il  # Use the dev_gpu_4_a100 partition with A100 GPUs dev_gpu_4
#SBATCH -n 1                   # Number of tasks (1 for single node)
#SBATCH -t 240           # Time limit (10 minutes for debugging purposes)
#SBATCH --mem=60000             # Memory request (adjust as needed)
#SBATCH --gres=gpu:1           # Request 1 GPU (adjust if you need more)
#SBATCH --cpus-per-task=16     # Number of CPUs per GPU (16 for A100)
#SBATCH --ntasks-per-node=1    # Number of tasks per node (1 in this case)

module load devel/miniforge

conda activate BasicTS

# prediction length and input length
# OUTPUT_LEN=100


# prediction length and input length
# OUTPUT_LEN=100

DATASET_NAME="ETTh1_prob_quantile"
# PatchTST
# MODEL_NAME="PatchTST"
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
# SweepID='02lujanw' # - 720 quantile
# SweepID='' # - 720 iquantile

MODEL_NAME="DLinear"

python ~/Probabilistic_LTSF/BasicTS/experiments/hp_tuning.py -c baselines/${MODEL_NAME}/${DATASET_NAME}.py -s /home/kreffert/Probabilistic_LTSF/BasicTS/hp_tuning/${MODEL_NAME}_quantile.yaml --gpus '0'
# python ~/Probabilistic_LTSF/BasicTS/experiments/hp_tuning.py -c baselines/${MODEL_NAME}/${DATASET_NAME}.py -s ${SweepID} --gpus '0'

# python ~/Probabilistic_LTSF/BasicTS/experiments/hp_tuning.py -c baselines/PatchTST/ETTh1_prob.py -s 'fd40u7me' --gpus '0, 1'
# python ~/Probabilistic_LTSF/BasicTS/experiments/train.py -c baselines/PatchTST/ETTh1_prob.py --gpus '0'
# python ~/Probabilistic_LTSF/BasicTS/experiments/train.py -c final_weights/PatchTST/univariate/ETTh1_prob_quantile.py --gpus '0'