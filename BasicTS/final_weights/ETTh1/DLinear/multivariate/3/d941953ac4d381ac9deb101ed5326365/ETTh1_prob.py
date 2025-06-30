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
INPUT_LEN = 96# regular_settings['INPUT_LEN']  # Length of input sequence
OUTPUT_LEN = 720  # <-- Updated
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']  # Train/Validation/Test split ratios
NORM_EACH_CHANNEL = True  # <-- Updated
RESCALE = True  # <-- Updated
NULL_VAL = regular_settings['NULL_VAL'] # Null value in the data
# Model architecture and parameters
MODEL_ARCH = DLinear
MODEL_PARAM = {
    "seq_len": INPUT_LEN,
    "pred_len": OUTPUT_LEN,
    "individual": False,  # <-- Updated
    "enc_in": 7,  # <-- Confirmed
    "head_type": "probabilistic",  # <-- Confirmed
    "distribution_type": "m_lr_gaussian",  # <-- Confirmed
    "prob_individual": False,  # <-- Updated
    "prob_args": {'rank': 360},  # <-- Updated
}
NUM_EPOCHS = 100  # <-- Confirmed

############################## General Configuration ##############################
CFG = EasyDict()
# General settings
CFG.DESCRIPTION = 'An Example Config'
CFG.GPU_NUM = 1 # Number of GPUs to use (0 for CPU mode)
# Runner
CFG.RUNNER = SimpleProbTimeSeriesForecastingRunner

CFG.USE_WANDB = False

############################## Environment Configuration ##############################
CFG.ENV = EasyDict()
CFG.ENV.TF32 = False
CFG.ENV.SEED = 3
CFG.ENV.DETERMINISTIC = True
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True
CFG.ENV.CUDNN.BENCHMARK = True
CFG.ENV.CUDNN.DETERMINISTIC = True

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
# Scaler settings
CFG.SCALER.TYPE = ZScoreScaler  # <-- Confirmed
CFG.SCALER.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_ratio': TRAIN_VAL_TEST_RATIO[0],
    'norm_each_channel': NORM_EACH_CHANNEL,
    'rescale': RESCALE,
})

############################## Model Configuration ##############################
CFG.MODEL = EasyDict()
# Model settings
CFG.MODEL.NAME = "DLinear"  # <-- Confirmed
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
CFG.TRAIN.RESUME_TRAINING = False  # <-- Confirmed
CFG.TRAIN.EARLY_STOPPING_PATIENCE = 5
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    '/home/kreffert/Probabilistic_LTSF/BasicTS/final_weights',
    f'{DATA_NAME}/{MODEL_ARCH.__name__}/multivariate',
    '_'.join([str(CFG.ENV.SEED)])
)
CFG.TRAIN.LOSS = nll_loss
# Optimizer settings
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"  # <-- Confirmed
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.0004591411744501484,  # <-- Updated
    "weight_decay": 0.00030069416293540747  # <-- Updated
}
# Learning rate scheduler settings
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"  # <-- Confirmed
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [5, 25],
    "gamma": 0.32396501765349983  # <-- Updated
}
CFG.TRAIN.CLIP_GRAD_PARAM = {
    'max_norm': 5.0
}
# Train data loader settings
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 16  # <-- Updated
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
CFG.EVAL.USE_GPU = True # Whether to use GPU for evaluation. Default: True