import matplotlib.pyplot as plt
import os
os.chdir('/home/kreffert/Probabilistic_LTSF/BasicTS/')
from basicts.data import TimeSeriesForecastingDataset
from basicts.utils import get_regular_settings


def load_data():
    data_sets = {}
    for DATA_NAME in ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Traffic", "Weather", "ExchangeRate", "Illness"]:
        regular_settings = get_regular_settings(DATA_NAME)
        TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
        NORM_EACH_CHANNEL = True
        RESCALE = True
        NULL_VAL = regular_settings['NULL_VAL']
        params = {'dataset_name': DATA_NAME,
                    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
                    'input_len': 96,
                    'output_len': 720,
        }
        data_sets[DATA_NAME] = {}
        data_sets[DATA_NAME]['train'] = TimeSeriesForecastingDataset(mode='train', **params)
        data_sets[DATA_NAME]['val'] = TimeSeriesForecastingDataset(mode='valid', **params)
        data_sets[DATA_NAME]['test'] = TimeSeriesForecastingDataset(mode='test', **params)
    return data_sets