import os
import sys
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../..'))

from basicts.metrics import masked_mae, masked_mse, nll_loss, crps, Evaluator, quantile_loss, empirical_crps
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleProbTimeSeriesForecastingRunner, SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler, MinMaxScaler
from basicts.utils import get_regular_settings

from baselines.DLinear.arch import DLinear

############################## Hot Parameters ##############################
# Dataset & Metrics configuration
DATA_NAME = 'ETTh1'  # Dataset name
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = 96 # regular_settings['INPUT_LEN']
OUTPUT_LEN = 720  # Overwritten from new file
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = False  # Overwritten
RESCALE = True  # Overwritten
NULL_VAL = regular_settings['NULL_VAL']

# Model architecture and parameters
MODEL_ARCH = DLinear
MODEL_PARAM = {
    "seq_len": INPUT_LEN,
    "pred_len": OUTPUT_LEN,
    "individual": True,  # Overwritten
    "enc_in": 7,  # Overwritten
    "head_type": "probabilistic",  # Overwritten
    "distribution_type": "i_quantile",  # Overwritten
    "prob_individual": False,  # Overwritten
    "prob_args": {
        "quantiles": [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995],
        "num_layers": 5,  # Overwritten
        "quantile_embed_dim": 64,  # Overwritten
        "cos_embedding_dim": 64,  # Overwritten
        "decoding": "concat",  # Overwritten
        "fixed_qe": 48,  # Preserved
    },
}

NUM_EPOCHS = 100  # Confirmed from new file

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = 'An Example Config'
CFG.GPU_NUM = 1
CFG.RUNNER = SimpleProbTimeSeriesForecastingRunner
CFG.USE_WANDB = False

############################## Environment Configuration ##############################
CFG.ENV = EasyDict()
CFG.ENV.TF32 = False
CFG.ENV.SEED = 4
CFG.ENV.DETERMINISTIC = True
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True
CFG.ENV.CUDNN.BENCHMARK = True
CFG.ENV.CUDNN.DETERMINISTIC = True

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
CFG.SCALER.TYPE = MinMaxScaler  # Overwritten
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
CFG.TRAIN.RESUME_TRAINING = False  # Overwritten
CFG.TRAIN.EARLY_STOPPING_PATIENCE = 5  # Overwritten
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    '/home/kreffert/Probabilistic_LTSF/BasicTS/final_weights',
    f'{DATA_NAME}/{MODEL_ARCH.__name__}/i_quantile',
    '_'.join([str(CFG.ENV.SEED)])
)
CFG.TRAIN.LOSS = quantile_loss

CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"  # Overwritten
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.005013910381857603,  # Overwritten
    "weight_decay": 0.00002661411291285973,  # Overwritten
}

CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"  # Overwritten
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [5, 25],
    "gamma": 0.07686312551351807  # Overwritten
}
CFG.TRAIN.CLIP_GRAD_PARAM = {'max_norm': 5.0}

CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 32  # Overwritten
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