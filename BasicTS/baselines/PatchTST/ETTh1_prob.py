import os
import sys
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../..'))

from basicts.metrics import masked_mae, masked_mse, nll_loss, crps, Evaluator, quantile_loss, empirical_crps
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleProbTimeSeriesForecastingRunner, SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler, MinMaxScaler
from basicts.utils import get_regular_settings

from .arch import PatchTST

############################## Hot Parameters ##############################
# Dataset & Metrics configuration
DATA_NAME = 'ETTh1'  # Dataset name
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = 96 # regular_settings['INPUT_LEN']  # Length of input sequence
OUTPUT_LEN = 720 #regular_settings['OUTPUT_LEN']  # Length of output sequence
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']  # Train/Validation/Test split ratios
NORM_EACH_CHANNEL = True #regular_settings['NORM_EACH_CHANNEL'] # Whether to normalize each channel of the data
RESCALE = True #regular_settings['RESCALE'] # Whether to rescale the data
NULL_VAL = regular_settings['NULL_VAL'] # Null value in the data
# Model architecture and parameters
MODEL_ARCH = PatchTST
NUM_NODES = 7
MODEL_PARAM = {
    "enc_in": NUM_NODES,                        # num nodes
    "seq_len": INPUT_LEN,           # input sequence length
    "pred_len": OUTPUT_LEN,         # prediction sequence length
    "e_layers": 2,                              # num of encoder layers
    "n_heads": 2,
    "d_model": 32,
    "d_ff": 128,
    "dropout": 0.3,
    "fc_dropout": 0.3,
    "head_dropout": 0.1,
    "patch_len": 32,
    "stride": 64,
    "individual": 1,                            # individual head; True 1 False 0
    "padding_patch": "end",                     # None: None; end: padding on the end
    "revin": 1,                                 # RevIN; True 1 False 0
    "affine": 1,                                # RevIN-affine; True 1 False 0
    "subtract_last": 0,                         # 0: subtract mean; 1: subtract last
    "decomposition": 1,                         # decomposition; True 1 False 0
    "kernel_size": 25,                          # decomposition-kernel
    "head_type": "probabilistic",
    "distribution_type": "m_lr_gaussian",
    "prob_args":{#"quantiles": [],#[0.1, 0.25, 0.5, 0.75, 0.9],
                "rank":180,
                "base_distribution": "laplace",
                "base_prob_args": {"rank":7, "quantiles": [],},
                "n_flows": 2,
                "flow_hidden_dim": 16,
                "flow_type": "sigmoidal", # sigmoidal, rectified, affine
                }, 
}
NUM_EPOCHS = 100

############################## General Configuration ##############################
CFG = EasyDict()
# General settings
CFG.DESCRIPTION = 'An Example Config'
CFG.GPU_NUM = 1 # Number of GPUs to use (0 for CPU mode)
# Runner
CFG.RUNNER = SimpleProbTimeSeriesForecastingRunner

CFG.USE_WANDB = False

############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
# Dataset settings
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
    # 'mode' is automatically set by the runner
})

############################## Scaler Configuration ##############################
CFG.SCALER = EasyDict()
#Scaler settings
CFG.SCALER.TYPE = MinMaxScaler # Scaler class, None MinMaxScaler ZScoreScaler
CFG.SCALER.PARAM = EasyDict({
   'dataset_name': DATA_NAME,
   'train_ratio': TRAIN_VAL_TEST_RATIO[0],
   'norm_each_channel': NORM_EACH_CHANNEL,
    'rescale': RESCALE,
})

############################## Model Configuration ##############################
CFG.MODEL = EasyDict()
# Model settings
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0]
CFG.MODEL.TARGET_FEATURES = [0]

############################## Metrics Configuration ##############################

CFG.METRICS = EasyDict()
# Metrics settings
all_metrics = ["MSE", "abs_error", "abs_target_sum", "abs_target_mean",
                                "MAPE", "sMAPE", "MASE", "RMSE", "NRMSE", "ND", "weighted_ND",
                                "mean_absolute_QuantileLoss", "CRPS", "MAE_Coverage", "NLL", 
                                #"VS", "ES"
                                ]
CFG.METRICS.FUNCS = EasyDict({'NLL': nll_loss,
                            #'MAE': masked_mae,
                            #'MSE': masked_mse,
                            'CRPS': crps,
                            #'CRPS_E': empirical_crps,
                            #'QL': quantile_loss,
                            #'Evaluator': Evaluator(distribution_type=MODEL_PARAM['distribution_type'], 
                            #                        quantiles=MODEL_PARAM['quantiles']),
                            # 'Val_Evaluator': Evaluator(distribution_type=MODEL_PARAM['distribution_type'], metrics = all_metrics,
                            #                         quantiles=MODEL_PARAM['quantiles']),  # only use the evaluator during validation/testing iters
                            })
CFG.METRICS.TARGET = 'NLL'
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
CFG.TRAIN.LOSS = nll_loss
# Optimizer settings
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.0002,
    "weight_decay": 0.0001,
}
# Learning rate scheduler settings
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [5, 25],
    "gamma": 0.5
}
CFG.TRAIN.CLIP_GRAD_PARAM = {
    'max_norm': 5.0
}
# Train data loader settings
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 128
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

# Evaluation parameters
CFG.EVAL.USE_GPU = True # Whether to use GPU for evaluation. Default: True
