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
DATA_NAME = 'ETTh1'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = 96
OUTPUT_LEN = 720
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = True
RESCALE = True
NULL_VAL = regular_settings['NULL_VAL']

# Model architecture and parameters
MODEL_ARCH = PatchTST
NUM_NODES = 7
MODEL_PARAM = {
    "enc_in": NUM_NODES,
    "seq_len": INPUT_LEN,
    "pred_len": OUTPUT_LEN,
    "e_layers": 10,
    "n_heads": 2,
    "d_model": 8,
    "d_ff": 16,
    "dropout": 0.25129474147072045,
    "fc_dropout": 0.12148425120618409,
    "head_dropout": 0.09730336677004645,
    "patch_len": 8,
    "stride": 2,
    "individual": 0,
    "padding_patch": None,
    "revin": 0,
    "affine": 0,
    "subtract_last": 1,
    "decomposition": 1,
    "kernel_size": 49,
    "head_type": "probabilistic",
    "distribution_type": "quantile",
    "attn_dropout": 0.2055312479592259,
    "act": "relu",
    "norm": "LayerNorm",
    "pe": "sincos",
    "learn_pe": True,
    "pre_norm": True,
    "prob_args": {
        "quantiles": [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995],
        "num_layers": 7,
        "quantile_embed_dim": 128,
        "cos_embedding_dim": 128,
        "decoding": "concat",
        "fixed_qe": 96,
    },
}
NUM_EPOCHS = 100

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = 'An Example Config'
CFG.GPU_NUM = 1
CFG.RUNNER = SimpleProbTimeSeriesForecastingRunner
CFG.USE_WANDB = False

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
CFG.SCALER.TYPE = ZScoreScaler
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
    'QL': quantile_loss,
})
CFG.METRICS.TARGET = 'QL'
CFG.METRICS.NULL_VAL = NULL_VAL

############################## Training Configuration ##############################
CFG.TRAIN = EasyDict()
CFG.TRAIN.RESUME_TRAINING = False
CFG.TRAIN.EARLY_STOPPING_PATIENCE = 5
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    f'{MODEL_PARAM["distribution_type"]}_{MODEL_ARCH.__name__}',
    '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = quantile_loss

CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.012226217927984995,
    "weight_decay": 0.00022885715483695764,
}

CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [5, 25],
    "gamma": 0.364258083259319,
}
CFG.TRAIN.CLIP_GRAD_PARAM = {
    'max_norm': 5.0
}

CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 32
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