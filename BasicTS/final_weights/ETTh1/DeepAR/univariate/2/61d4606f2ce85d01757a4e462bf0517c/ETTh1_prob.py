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
INPUT_LEN = 96 #regular_settings['INPUT_LEN']  # Length of input sequence
OUTPUT_LEN = 720 # Updated
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']  # Train/Validation/Test split ratios
NORM_EACH_CHANNEL = True # Updated
RESCALE = True # Updated
NULL_VAL = regular_settings['NULL_VAL'] # Null value in the data

# Model architecture and parameters
MODEL_ARCH = DeepAR
MODEL_PARAM = {
    'cov_feat_size': 0,  # Updated
    'embedding_size': 128,  # Updated
    'hidden_size': 32,  # Updated
    'num_layers': 8,  # Updated
    'use_ts_id': True,
    'id_feat_size': 8,  # Updated
    'num_nodes': 7,
    "distribution_type": "student_t",  # Updated
    "prob_args": {},
}
NUM_EPOCHS = 100  # Already correct

############################## General Configuration ##############################
CFG = EasyDict()
# General settings
CFG.DESCRIPTION = 'An Example Config'
CFG.GPU_NUM = 1
# Runner
CFG.RUNNER = SimpleProbTimeSeriesForecastingRunner

CFG.USE_WANDB = False

############################## Environment Configuration ##############################
CFG.ENV = EasyDict()
# GPU and random seed settings
CFG.ENV.TF32 = False # Whether to use TensorFloat-32 in GPU. Default: False. See https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere.
CFG.ENV.SEED = 2 # Random seed. Default: None
CFG.ENV.DETERMINISTIC = True # Whether to set the random seed to get deterministic results. Default: False
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True # Whether to enable cuDNN. Default: True
CFG.ENV.CUDNN.BENCHMARK = True# Whether to enable cuDNN benchmark. Default: True
CFG.ENV.CUDNN.DETERMINISTIC = True # Whether to set cuDNN to deterministic mode. Default: False

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
CFG.SCALER.TYPE = MinMaxScaler  # Updated
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
CFG.METRICS.FUNCS = EasyDict({'NLL': nll_loss,
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
    '/home/kreffert/Probabilistic_LTSF/BasicTS/final_weights',
    f'{DATA_NAME}/{MODEL_ARCH.__name__}/univariate',
    # f'{MODEL_PARAM["distribution_type"]}_{MODEL_ARCH.__name__}',
    '_'.join([str(CFG.ENV.SEED)])
)
CFG.TRAIN.LOSS = nll_loss

# Optimizer settings
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"  # Updated
CFG.TRAIN.OPTIM.PARAM = {
    'lr': 0.01120632065230968,  # Updated
    "weight_decay": 0.0006552434086191572,  # Updated
}

# Learning rate scheduler settings
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"  # Already matches
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [5, 25],
    "gamma": 0.6703581388363418  # Updated
}
CFG.TRAIN.CLIP_GRAD_PARAM = {
    'max_norm': 5.0
}

# Train data loader settings
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 128  # Updated
CFG.TRAIN.DATA.SHUFFLE = True

############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 128

############################## Test Configuration ##############################
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 128

############################## Evaluation Configuration ##############################
CFG.EVAL = EasyDict()
CFG.EVAL.USE_GPU = True