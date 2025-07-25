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
OUTPUT_LEN = 720 #regular_settings['OUTPUT_LEN']  # Length of output sequence
TRAIN_VAL_TEST_RATIO = #regular_settings['TRAIN_VAL_TEST_RATIO']  # Train/Validation/Test split ratios
NORM_EACH_CHANNEL =  # Whether to normalize each channel of the data
RESCALE = True # Whether to rescale the data
NULL_VAL = regular_settings['NULL_VAL'] # Null value in the data
# Model architecture and parameters
MODEL_ARCH = DeepAR
MODEL_PARAM = {
    'cov_feat_size' : 2,
    'embedding_size' : 32,
    'hidden_size' : 48,
    'num_layers': 3,
    'use_ts_id'   : False,
    'id_feat_size': 32,
    'num_nodes': 7,
    # "head_type": "probabilistic", #-> for DeepAR there are only probabilistic head types!
    # "head_type": "probabilistic",
    "distribution_type": "i_quantile",
    "prob_args":{"quantiles": [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995], #[0.1, 0.25, 0.5, 0.75, 0.9],
                "num_layers": 2, 
                "quantile_embed_dim": 64, 
                "cos_embedding_dim": 128,
                "decoding": "hadamard",
                "fixed_qe": 48, # used for hadamard since it needs the right dimension
                # "rank":13,
                # "base_distribution": "laplace",
                # "base_prob_args": {"rank":7, "quantiles": [],},
                # "n_flows": 2,
                # "flow_hidden_dim": 16,
                # "flow_type": "sigmoidal", # sigmoidal, rectified, affine
                }, 
    }
NUM_EPOCHS = 100

############################## General Configuration ##############################
CFG = EasyDict()
# General settings
CFG.DESCRIPTION = 'An Example Config'
CFG.GPU_NUM = 1 # Number of GPUs to use (0 for CPU mode)
# Runner
CFG.RUNNER = SimpleProbTimeSeriesForecastingRunner #DeepARRunner

CFG.USE_WANDB = False

############################## Environment Configuration ##############################
CFG.ENV = EasyDict()
# GPU and random seed settings
CFG.ENV.TF32 = False # Whether to use TensorFloat-32 in GPU. Default: False. See https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere.
CFG.ENV.SEED = 0 # Random seed. Default: None
CFG.ENV.DETERMINISTIC = True # Whether to set the random seed to get deterministic results. Default: False
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True # Whether to enable cuDNN. Default: True
CFG.ENV.CUDNN.BENCHMARK = True# Whether to enable cuDNN benchmark. Default: True
CFG.ENV.CUDNN.DETERMINISTIC = True # Whether to set cuDNN to deterministic mode. Default: False

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
CFG.METRICS.FUNCS = EasyDict({#'NLL': nll_loss,
                            #'MAE': masked_mae,
                            #'MSE': masked_mse,
                            # 'CRPS': crps,
                            #'CRPS_E': empirical_crps,
                            'QL': quantile_loss,
                            #'Evaluator': Evaluator(distribution_type=MODEL_PARAM['distribution_type'], 
                            #                        quantiles=MODEL_PARAM['quantiles']),
                            # 'Val_Evaluator': Evaluator(distribution_type=MODEL_PARAM['distribution_type'], metrics = all_metrics,
                            #                         quantiles=MODEL_PARAM['quantiles']),  # only use the evaluator during validation/testing iters
                            })
CFG.METRICS.TARGET = 'QL'
CFG.METRICS.NULL_VAL = NULL_VAL

############################## Training Configuration ##############################
CFG.TRAIN = EasyDict()
CFG.TRAIN.RESUME_TRAINING = False
CFG.TRAIN.EARLY_STOPPING_PATIENCE = 5
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    '/home/kreffert/Probabilistic_LTSF/BasicTS/final_weights',
    f'{MODEL_ARCH.__name__}/quantile',
    # f'{MODEL_PARAM["distribution_type"]}_{MODEL_ARCH.__name__}',
    '_'.join([str(CFG.ENV.SEED)])
)
CFG.TRAIN.LOSS = quantile_loss
# Optimizer settings
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    'lr':0.003,
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

# Evaluation parameters
CFG.EVAL.USE_GPU = True # Whether to use GPU for evaluation. Default: True
