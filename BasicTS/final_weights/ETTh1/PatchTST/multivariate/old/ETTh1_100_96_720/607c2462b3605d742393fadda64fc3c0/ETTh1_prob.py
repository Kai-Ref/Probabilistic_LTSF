import os
import sys
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../..'))

from basicts.metrics import masked_mae, masked_mse, nll_loss, crps, Evaluator, quantile_loss, empirical_crps
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleProbTimeSeriesForecastingRunner, SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler, MinMaxScaler
from basicts.utils import get_regular_settings

from baselines.PatchTST.arch import PatchTST

############################## Hot Parameters ##############################
# Dataset & Metrics configuration
DATA_NAME = 'ETTh1'  # Dataset name
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = 96
OUTPUT_LEN = 720
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = False
RESCALE = True
NULL_VAL = regular_settings['NULL_VAL']

# Model architecture and parameters
MODEL_ARCH = PatchTST
NUM_NODES = 7
MODEL_PARAM = {
    "enc_in": NUM_NODES,
    "seq_len": INPUT_LEN,
    "pred_len": OUTPUT_LEN,
    "e_layers": 5,
    "n_heads": 2,
    "d_model": 64,
    "d_ff": 64,
    "dropout": 0.185360841476715,
    "fc_dropout": 0.36901599745387936,
    "head_dropout": 0.04836474431752078,
    "patch_len": 32,
    "stride": 128,
    "individual": 0,
    "padding_patch": "end",
    "revin": 1,
    "affine": 0,
    "subtract_last": 0,
    "decomposition": 1,
    "kernel_size": 13,
    "head_type": "probabilistic",
    "distribution_type": "m_lr_gaussian",
    "attn_dropout": 0.011447726128960237,
    "act": "gelu",
    "norm": "LayerNorm",
    "pre_norm": True,
    "learn_pe": True,
    "pe": "zeros",
    "prob_args": {
        "rank": 110,
    },
}
NUM_EPOCHS = 100

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = 'An Example Config'
CFG.GPU_NUM = 1
CFG.RUNNER = SimpleProbTimeSeriesForecastingRunner
CFG.USE_WANDB = False

############################## Environment Configuration ##############################

CFG.ENV = EasyDict() # Environment settings. Default: None

# GPU and random seed settings
CFG.ENV.TF32 = False # Whether to use TensorFloat-32 in GPU. Default: False. See https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere.
CFG.ENV.SEED = 47 # Random seed. Default: None
CFG.ENV.DETERMINISTIC = True # Whether to set the random seed to get deterministic results. Default: False
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True # Whether to enable cuDNN. Default: True
CFG.ENV.CUDNN.BENCHMARK = True# Whether to enable cuDNN benchmark. Default: True
CFG.ENV.CUDNN.DETERMINISTIC = False # Whether to set cuDNN to deterministic mode. Default: False

############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
})

############################## Scaler Configuration ##############################
CFG.SCALER = EasyDict()
CFG.SCALER.TYPE = MinMaxScaler
CFG.SCALER.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_ratio': TRAIN_VAL_TEST_RATIO[0],
    'norm_each_channel': NORM_EACH_CHANNEL,
    'rescale': RESCALE,
})

############################## Model Configuration ##############################
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0]
CFG.MODEL.TARGET_FEATURES = [0]

############################## Metrics Configuration ##############################
CFG.METRICS = EasyDict()
all_metrics = ["MSE", "abs_error", "abs_target_sum", "abs_target_mean",
               "MAPE", "sMAPE", "MASE", "RMSE", "NRMSE", "ND", "weighted_ND",
               "mean_absolute_QuantileLoss", "CRPS", "MAE_Coverage", "NLL"]
CFG.METRICS.FUNCS = EasyDict({
    'NLL': nll_loss,
    'CRPS': crps,
})
CFG.METRICS.TARGET = 'NLL'
CFG.METRICS.NULL_VAL = NULL_VAL

############################## Training Configuration ##############################
CFG.TRAIN = EasyDict()
CFG.TRAIN.RESUME_TRAINING = False
CFG.TRAIN.EARLY_STOPPING_PATIENCE = 5
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    # '/pfs/data6/home/ma/ma_ma/ma_kreffert/Probabilistic_LTSF/BasicTS/final_weights',
    '/home/kreffert/Probabilistic_LTSF/BasicTS/final_weights',
    f'{MODEL_ARCH.__name__}/multivariate',
    '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = nll_loss

CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.0033396746047616513,
    "weight_decay": 0.0002796603702438695,
}

CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [5, 25],
    "gamma": 0.24377762538007125,
}
CFG.TRAIN.CLIP_GRAD_PARAM = {
    'max_norm': 5.0
}
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 64
CFG.TRAIN.DATA.SHUFFLE = True

############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 64

############################## Test Configuration ##############################
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 64

############################## Evaluation Configuration ##############################
CFG.EVAL = EasyDict()
CFG.EVAL.USE_GPU = True