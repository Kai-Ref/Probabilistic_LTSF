import os
import sys
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../..'))
from basicts.metrics import masked_mae, masked_mse, nll_loss, crps, Evaluator, quantile_loss
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleProbTimeSeriesForecastingRunner, SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler, MinMaxScaler
from basicts.utils import get_regular_settings

from baselines.DeepAR.arch import DeepAR
# from .runner import DeepARRunner
# from .loss import gaussian_loss

############################## Hot Parameters ##############################
# Dataset & Metrics configuration
DATA_NAME = 'ETTh1'  # Dataset name
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = 96  # regular_settings['INPUT_LEN']  # Length of input sequence
OUTPUT_LEN = 720  # From new file
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']  # Train/Validation/Test split ratios
NORM_EACH_CHANNEL = True  # From new file
RESCALE = True  # From new file
NULL_VAL = regular_settings['NULL_VAL']  # Null value in the data

# Model architecture and parameters
MODEL_ARCH = DeepAR
MODEL_PARAM = {
    'cov_feat_size': 0,  # From new file
    'embedding_size': 64,  # From new file
    'hidden_size': 64,  # From new file
    'num_layers': 8,  # From new file
    'use_ts_id': False,  # From new file
    'id_feat_size': 16,  # From new file
    'num_nodes': 7,  # From new file
    "distribution_type": "i_quantile",  # From new file
    "prob_args": {
        "quantiles": [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995],
        "num_layers": 2,  # From new file
        "quantile_embed_dim": 128,  # From new file
        "cos_embedding_dim": 8,  # From new file
        "decoding": "concat",  # From new file
        "fixed_qe": 64,  # From new file
    },
}
NUM_EPOCHS = 100  # From new file

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = 'An Example Config'
CFG.GPU_NUM = 1
CFG.RUNNER = SimpleProbTimeSeriesForecastingRunner
CFG.USE_WANDB = False

############################## Environment Configuration ##############################
CFG.ENV = EasyDict()
CFG.ENV.TF32 = False
CFG.ENV.SEED = 1
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
CFG.SCALER.TYPE = ZScoreScaler  # From new file
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
CFG.TRAIN.RESUME_TRAINING = False  # From new file
CFG.TRAIN.EARLY_STOPPING_PATIENCE = 5  # From new file
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    '/home/kreffert/Probabilistic_LTSF/BasicTS/final_weights',
    f'{DATA_NAME}/{MODEL_ARCH.__name__}/i_quantile',
    # f'{MODEL_PARAM["distribution_type"]}_{MODEL_ARCH.__name__}',
    '_'.join([str(CFG.ENV.SEED)])
)
CFG.TRAIN.LOSS = quantile_loss
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"  # From new file
CFG.TRAIN.OPTIM.PARAM = {
    'lr': 0.0028482886405204752,  # From new file
    "weight_decay": 0.00017969527868614752,  # From new file
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"  # From new file
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [5, 25],
    "gamma": 0.1355877339691653,  # From new file
}
CFG.TRAIN.CLIP_GRAD_PARAM = {
    'max_norm': 5.0
}
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 128  # From new file
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
