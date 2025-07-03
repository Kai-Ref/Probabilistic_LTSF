import matplotlib.pyplot as plt
import os
# os.chdir('/home/kreffert/Probabilistic_LTSF/BasicTS/')
os.chdir('/pfs/data6/home/ma/ma_ma/ma_kreffert/Probabilistic_LTSF/BasicTS/')
from basicts.data import TimeSeriesForecastingDataset
from basicts.utils import get_regular_settings
from basicts.scaler import ZScoreScaler, MinMaxScaler

def load_data():
    data_sets = {}
    for DATA_NAME in ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Traffic", "Weather", "ExchangeRate", "Illness"]:
        # regular_settings = get_regular_settings(DATA_NAME)
        TRAIN_VAL_TEST_RATIO = [0.6, 0.2, 0.2] #regular_settings['TRAIN_VAL_TEST_RATIO']
        NORM_EACH_CHANNEL = True
        RESCALE = True
        NULL_VAL = None #regular_settings['NULL_VAL']
        params = {'dataset_name': DATA_NAME,
                    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
                    'input_len': 96,
                    'output_len': 720,
                  'prefix': '/pfs/data6/home/ma/ma_ma/ma_kreffert/Probabilistic_LTSF/BasicTS/',
        }
        data_sets[DATA_NAME] = {}
        data_sets[DATA_NAME]['train'] = TimeSeriesForecastingDataset(mode='train', **params)
        data_sets[DATA_NAME]['val'] = TimeSeriesForecastingDataset(mode='valid', **params)
        data_sets[DATA_NAME]['test'] = TimeSeriesForecastingDataset(mode='test', **params)
    return data_sets

    
def create_scalers(data_sets, scaler_type='Z_score'):
    for DATA_NAME in data_sets.keys():
        params = {'dataset_name': DATA_NAME,
                'train_ratio': 1,
                'norm_each_channel': True,
                'rescale': True,
                 'prefix': '/pfs/data6/home/ma/ma_ma/ma_kreffert/Probabilistic_LTSF/BasicTS/',
                 }
        if scaler_type == 'Z_score':
            scaler = ZScoreScaler(**params)
        data_sets[DATA_NAME]['scaler'] = scaler
    return data_sets