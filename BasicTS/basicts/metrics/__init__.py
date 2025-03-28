from .corr import masked_corr
from .mae import masked_mae
from .mape import masked_mape
from .mse import masked_mse
from .r_square import masked_r2
from .rmse import masked_rmse
from .smape import masked_smape
from .wape import masked_wape
from .prob_metrics import nll_loss, crps, quantile_loss, empirical_crps
from .probts import smape, Evaluator

ALL_METRICS = {
            'MAE': masked_mae,
            'MSE': masked_mse,
            'RMSE': masked_rmse,
            'MAPE': masked_mape,
            'WAPE': masked_wape,
            'SMAPE': masked_smape,
            'R2': masked_r2,
            'CORR': masked_corr,
            'NLL': nll_loss,
            'CRPS': crps,
            'CRPS_E': empirical_crps,
            'QL': quantile_loss,
            }

__all__ = [
    'masked_mae',
    'masked_mse',
    'masked_rmse',
    'masked_mape',
    'masked_wape',
    'masked_smape',
    'masked_r2',
    'masked_corr',
    'gaussian_nll_loss',
    'crps',
    'ALL_METRICS'
]