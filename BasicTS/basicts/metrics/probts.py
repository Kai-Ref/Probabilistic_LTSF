# ---------------------------------------------------------------------------------
# Portions of this file are derived from gluonts
# - Source: https://github.com/awslabs/gluonts
# - Paper: GluonTS: Probabilistic and Neural Time Series Modeling in Python
# - License: Apache-2.0
#
# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


from typing import Optional
import numpy as np
from gluonts.time_feature import get_seasonality
import torch.distributions as dist
import torch
from prob.prob_head import ProbabilisticHead # load that class for sampling


def mse_old(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    .. math::

        mse = mean((Y - \hat{Y})^2)
    """
    return np.mean(np.square(target - forecast))


def abs_error_old(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    .. math::

        abs\_error = sum(|Y - \hat{Y}|)
    """
    return np.sum(np.abs(target - forecast))


def abs_target_sum_old(target) -> float:
    r"""
    .. math::

        abs\_target\_sum = sum(|Y|)
    """
    return np.sum(np.abs(target))


def abs_target_mean_old(target) -> float:
    r"""
    .. math::

        abs\_target\_mean = mean(|Y|)
    """
    return np.mean(np.abs(target))


def mase_old(
    target: np.ndarray,
    forecast: np.ndarray,
    seasonal_error: np.ndarray,
) -> float:
    r"""
    .. math::

        mase = mean(|Y - \hat{Y}|) / seasonal\_error

    See [HA21]_ for more details.
    """
    diff = np.mean(np.abs(target - forecast), axis=1)
    mase = diff / seasonal_error
    # if seasonal_error is 0, set mase to 0
    mase = mase.filled(0)  
    return np.mean(mase)

def calculate_seasonal_error_old(
    past_data: np.ndarray,
    freq: Optional[str] = None,
):
    r"""
    .. math::

        seasonal\_error = mean(|Y[t] - Y[t-m]|)

    where m is the seasonal frequency. See [HA21]_ for more details.
    """
    seasonality = get_seasonality(freq)

    if seasonality < len(past_data):
        forecast_freq = seasonality
    else:
        # edge case: the seasonal freq is larger than the length of ts
        # revert to freq=1

        # logging.info('The seasonal frequency is larger than the length of the
        # time series. Reverting to freq=1.')
        forecast_freq = 1
        
    y_t = past_data[:, :-forecast_freq]
    y_tm = past_data[:, forecast_freq:]

    mean_diff = np.mean(np.abs(y_t - y_tm), axis=1)
    mean_diff = np.expand_dims(mean_diff, axis=1)

    return mean_diff



def mape_old(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    .. math::

        mape = mean(|Y - \hat{Y}| / |Y|))

    See [HA21]_ for more details.
    """
    return np.mean(np.abs(target - forecast) / np.abs(target))


def smape_old(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    .. math::

        smape = 2 * mean(|Y - \hat{Y}| / (|Y| + |\hat{Y}|))

    See [HA21]_ for more details.
    """
    return 2 * np.mean(
        np.abs(target - forecast) / (np.abs(target) + np.abs(forecast))
    )

def quantile_loss_old(target: np.ndarray, forecast: np.ndarray, q: float) -> float:
    r"""
    .. math::

        quantile\_loss = 2 * sum(|(Y - \hat{Y}) * ((Y <= \hat{Y}) - q)|)
    """
    return 2 * np.abs((forecast - target) * ((target <= forecast) - q))

def scaled_quantile_loss_old(target: np.ndarray, forecast: np.ndarray, q: float, seasonal_error) -> np.ndarray:
    return quantile_loss(target, forecast, q) / seasonal_error

def coverage_old(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    .. math::

        coverage = mean(Y < \hat{Y})
    """
    return np.mean(target < forecast)

def mape(target: torch.Tensor, forecast: torch.Tensor) -> float:
    return torch.mean(torch.abs(target - forecast) / torch.abs(target)).item()

def smape(target: torch.Tensor, forecast: torch.Tensor) -> float:
    return 2 * torch.mean(torch.abs(target - forecast) / (torch.abs(target) + torch.abs(forecast))).item()

# def quantile_loss(target: torch.Tensor, forecast: torch.Tensor, q: float) -> float:
#     return torch.sum(torch.abs((forecast - target) * ((target <= forecast).float() - q))).item()

# def quantile_loss(targets: torch.Tensor, forecasts: torch.Tensor, q: float) -> torch.Tensor:
#     """
#     Computes quantile loss.
#     """
#     error = targets - forecasts
#     return torch.maximum(q * error, (q - 1) * error)

def quantile_loss(targets: torch.Tensor, forecasts: torch.Tensor, q: float, dim=[1, 2]) -> torch.Tensor:
    """
    Computes quantile loss using the formula:
    quantile_loss = 2 * sum(|(Y - Ŷ) * ((Y <= Ŷ) - q)|)
    """
    error = targets - forecasts  # Compute the forecast error
    indicator = (targets <= forecasts).float()  # Get 1 where Y ≤ Ŷ, otherwise 0
    return 2 * torch.sum(torch.abs(error * (indicator - q)), dim=dim)

def scaled_quantile_loss(target: torch.Tensor, forecast: torch.Tensor, q: float, seasonal_error: torch.Tensor) -> torch.Tensor:
    return quantile_loss(target, forecast, q) / seasonal_error

def coverage(target: torch.Tensor, forecast: torch.Tensor) -> float:
    return (target < forecast).float()

def mse(target: torch.Tensor, forecast: torch.Tensor) -> float:
    return torch.mean((target - forecast) ** 2).item()

def abs_error(target: torch.Tensor, forecast: torch.Tensor) -> float:
    # print(torch.abs(target - forecast).shape)
    # print(torch.sum(torch.abs(target - forecast)).shape)
    #return torch.mean(torch.sum(torch.abs(target - forecast)), dim=0).item()
    return torch.sum(torch.abs(target - forecast), dim=[1, 2]) 

def abs_target_sum(target: torch.Tensor) -> float:
    return torch.mean(torch.sum(torch.abs(target), dim=0)).item()

def abs_target_mean(target: torch.Tensor) -> float:
    return torch.mean(torch.abs(target)).item()

def mase(target: torch.Tensor, forecast: torch.Tensor, seasonal_error: torch.Tensor) -> float:
    diff = torch.mean(torch.abs(target - forecast), dim=1)
    mase = diff / seasonal_error
    mase = torch.where(seasonal_error == 0, torch.zeros_like(mase), mase)
    return torch.mean(mase).item()

def calculate_seasonal_error(past_data: torch.Tensor, freq: Optional[str] = None) -> torch.Tensor:
    seasonality = get_seasonality(freq)
    if seasonality < past_data.size(1):
        forecast_freq = seasonality
    else:
        forecast_freq = 1

    y_t = past_data[:, :-forecast_freq]
    y_tm = past_data[:, forecast_freq:]

    mean_diff = torch.mean(torch.abs(y_t - y_tm), dim=1)
    mean_diff = mean_diff.unsqueeze(1)
    return mean_diff

import numpy as np
# from .metrics import *
import torch

class Evaluator:
    def __init__(self, metrics='all', quantiles=[], smooth=False, distribution_type=None):
        """
        Evaluator for probabilistic forecasting.
        Parameters
        ----------
        metrics -> List or str
            list of metrics to compute. If 'all', all available metrics are computed.
        quantiles -> List
            list of quantiles to compute. Only used if distribution_type is 'quantile'.
        smooth -> bool
            whether to smooth the forecasts before computing the metrics.
        distribution_type (same as in MODEL_PARAM) -> str
            type of distribution to use. If 'quantile', quantiles must be provided.
        """
        if len(quantiles) == 0:
            quantiles_num = 10
            self.quantiles = (1.0 * np.arange(quantiles_num) / quantiles_num)[1:]
        else:
            self.quantiles = quantiles
        self.ignore_invalid_values = True
        self.smooth = smooth
        self.distribution_type = distribution_type
        self.prob_head = ProbabilisticHead(1, 1, self.distribution_type)

        self.__all_metrics__ = ["MSE", "abs_error", "abs_target_sum", "abs_target_mean",
                                "MAPE", "sMAPE", "MASE", "RMSE", "NRMSE", "ND", "weighted_ND",
                                "mean_absolute_QuantileLoss", "CRPS", "MAE_Coverage"]
        if metrics == "all":
            self.metrics = self.__all_metrics__
        else:
            # check that the provided metrics are valid
            assert all([m in self.__all_metrics__ for m in self.metrics]) 
            self.metrics = metrics
        
        distributions = ["gaussian","laplace","student_t","lognormal","beta","gamma","weibull","poisson","negative_binomial","dirichlet"]

        # depending if the distribution is quantile or not, we need to set the sample flag
        if self.distribution_type in ["quantile", "i_quantile"]:
            self.sample = False
            self.loss_name = self.loss_name
            self.weighted_loss_name = self.weighted_loss_name
            self.coverage_name = self.coverage_name
        elif self.distribution_type in distributions:
            self.sample = True
        else: #TODO for point forecasts, compute the percentiles
            self.sample = False

    def loss_name(self, q):
        return f"QuantileLoss[{q}]"

    def weighted_loss_name(self, q):
        return f"wQuantileLoss[{q}]"

    def coverage_name(self, q):
        return f"Coverage[{q}]"

    def get_sequence_metrics(self, targets, forecasts, seasonal_error=None, samples_dim=1,loss_weights=None):
        mean_forecasts = forecasts.mean(axis=samples_dim)
        median_forecasts = np.quantile(forecasts, 0.5, axis=samples_dim)

        metrics = {
            "MSE": mse_old(targets, mean_forecasts),
            "abs_error": abs_error_old(targets, median_forecasts),
            "abs_target_sum": abs_target_sum_old(targets),
            "abs_target_mean": abs_target_mean_old(targets),
            "MAPE": mape_old(targets, median_forecasts),
            "sMAPE": smape_old(targets, median_forecasts),
        }
        
        if seasonal_error is not None:
            metrics["MASE"] = mase_old(targets, median_forecasts, seasonal_error)
        
        metrics["RMSE"] = np.sqrt(metrics["MSE"])
        metrics["NRMSE"] = metrics["RMSE"] / metrics["abs_target_mean"]
        metrics["ND"] = metrics["abs_error"] / metrics["abs_target_sum"]
        
        # calculate weighted loss
        if loss_weights is not None:
            nd = np.abs(targets - mean_forecasts) / np.sum(np.abs(targets), axis=(1, 2))
            loss_weights = loss_weights.detach().unsqueeze(0).unsqueeze(-1).numpy()
            weighted_ND = loss_weights * nd
            metrics['weighted_ND'] = np.sum(weighted_ND)
        else:
            metrics['weighted_ND'] = metrics["ND"]

        for q in self.quantiles:
            q_forecasts = np.quantile(forecasts, q, axis=samples_dim)
            metrics[self.loss_name(q)] = np.sum(quantile_loss_old(targets, q_forecasts, q))
            metrics[self.weighted_loss_name(q)] = \
                metrics[self.loss_name(q)] / metrics["abs_target_sum"]
            metrics[self.coverage_name(q)] = coverage_old(targets, q_forecasts)
        
        metrics["mean_absolute_QuantileLoss"] = np.mean(
            [metrics[self.loss_name(q)] for q in self.quantiles]
        )
        metrics["CRPS"] = np.mean(
            [metrics[self.weighted_loss_name(q)] for q in self.quantiles]
        )
        metrics["MAE_Coverage"] = np.mean(
            [
                np.abs(metrics[self.coverage_name(q)] - np.array([q]))
                for q in self.quantiles
            ]
        )
        return metrics

    def get_metrics_old(self, targets, forecasts, seasonal_error=None, samples_dim=1, loss_weights=None):
        metrics = {}
        seq_metrics = {}

        # TODO optimize below -> iterates over all 64 batches, maybe slow
        # Calculate metrics for each sequence
        for i in range(targets.shape[0]):
            single_seq_metrics = self.get_sequence_metrics(
                np.expand_dims(targets[i], axis=0),
                np.expand_dims(forecasts[i], axis=0),
                np.expand_dims(seasonal_error[i], axis=0) if seasonal_error is not None else None,
                samples_dim,
                loss_weights
            )
            for metric_name, metric_value in single_seq_metrics.items():
                if metric_name not in seq_metrics:
                    seq_metrics[metric_name] = []
                seq_metrics[metric_name].append(metric_value)
        
        for metric_name, metric_values in seq_metrics.items():
            metrics[metric_name] = np.mean(metric_values)
        return metrics

    def get_metrics(self, targets, forecasts, seasonal_error=None, samples_dim=1, loss_weights=None):
        metrics = {}
        # Convert targets and forecasts to PyTorch tensors
        targets = torch.tensor(targets).clone().detach()
        forecasts = torch.tensor(forecasts).clone().detach()
        # handle the different scenarios, e.g if forecasts are aggregated or not
        dim = [1, 2] if targets.dim() == 3 else [1]
        
        mean_forecasts = forecasts.mean(dim=samples_dim)
        median_forecasts = torch.quantile(forecasts, 0.5, dim=samples_dim)
        metrics = {
            "MSE": torch.mean((targets - mean_forecasts) ** 2, dim=dim),  # Per sequence MSE
            "abs_error": torch.sum(torch.abs(targets - median_forecasts), dim=dim),  # Per sequence abs error
            "abs_target_sum": torch.sum(torch.abs(targets), dim=dim),
            "abs_target_mean": torch.mean(torch.abs(targets), dim=dim),
            "MAPE": torch.mean(torch.abs((targets - median_forecasts) / targets), dim=dim),
            "sMAPE": torch.mean(2 * torch.abs(targets - median_forecasts) / (torch.abs(targets) + torch.abs(median_forecasts)), dim=dim),
        }
        
        if seasonal_error is not None:# TODO check if that is equal to 'normal' calculation
            metrics["MASE"] = mase(targets, median_forecasts, seasonal_error)

        metrics["RMSE"] = torch.sqrt(metrics["MSE"])
        metrics["NRMSE"] = metrics["RMSE"] / metrics["abs_target_mean"]
        metrics["ND"] = metrics["abs_error"] / metrics["abs_target_sum"]

        # Calculate weighted loss
        if loss_weights is not None:
            nd = torch.sum(torch.abs(targets - mean_forecasts), dim=dim) / metrics["abs_target_sum"]
            loss_weights = loss_weights.unsqueeze(0).unsqueeze(-1)  # Reshape for broadcasting
            metrics['weighted_ND'] = torch.sum(loss_weights * nd)
        else:
            metrics['weighted_ND'] = metrics["ND"]

        # Compute quantile-based metrics
        for q in self.quantiles:
            q_forecasts = torch.quantile(forecasts, q, dim=samples_dim)
            quant_loss = quantile_loss(targets, q_forecasts, q, dim=dim)

            metrics[self.loss_name(q)] = quant_loss
            metrics[self.weighted_loss_name(q)] = quant_loss / metrics["abs_target_sum"]
            metrics[self.coverage_name(q)] = coverage(targets, q_forecasts)

        metrics["mean_absolute_QuantileLoss"] = torch.mean(
            torch.stack([metrics[self.loss_name(q)] for q in self.quantiles]), dim=0
        )
        metrics["CRPS"] = torch.mean(
            torch.stack([metrics[self.weighted_loss_name(q)] for q in self.quantiles]), dim=0
        )

        # TODO: check the correct implementation and adjust it 
        # metrics["MAE_Coverage"] = torch.mean(
        #     torch.stack([torch.abs(metrics[self.coverage_name(q)] - q) for q in self.quantiles]), dim=0
        # )

        # Take the mean across all sequences
        return {key: torch.mean(value).item() for key, value in metrics.items()}

    @property
    def selected_metrics(self):
        return [ "ND",'weighted_ND', 'CRPS', "NRMSE", "MSE"] #, "MASE"]

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor, model, null_val: float = np.nan):
        #targets, forecasts, past_data, freq, loss_weights=None):
        """

        Parameters
        ----------
        targets
            groundtruth in (batch_size, prediction_length, target_dim)
        forecasts
            forecasts in (batch_size, num_samples, prediction_length, target_dim)
        Returns
        -------
        Dict[String, float]
            metrics
        """
        # if np.isnan(null_val):
        #     mask = ~torch.isnan(target)
        # else:
        #     eps = 5e-5
        #     mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.)
        # mask = mask.float()
        # mask /= torch.mean((mask))
        # mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

        # TODO other distribution samples
        # if self.distribution_type == "gaussian":
        #     # reshape targets from [bs x seq_len x nvars x 1] into [bs x seq_len x nvars]
        #     target = target.squeeze(-1) # [bs x seq_len x nvars]

        if self.sample:
            target = target.squeeze(-1)
            prediction = self.prob_head.sample(prediction, num_samples=100) # [samples x bs x seq_len x nvars]
            prediction = prediction.permute(1, 0, 2, 3)       # [bs x samples x seq_len x nvars]
        else:
            # reshape prediction from [bs x nvars x seq_len x num_params/quantiles] into [bs x num_params/quantiles x seq_len x nvars]
            prediction = prediction.permute(1, 0, 2, 3)     # [bs x num_quantiles x seq_len x nvars]
            prediction = prediction.squeeze(-1)
        
        # TODO: MASE and seasonal error calculation
        # past_data = process_tensor(past_data)
        # seasonal_error = calculate_seasonal_error(past_data, freq)
        seasonal_error = None
        loss_weights = None
        metrics = self.get_metrics(target, prediction, seasonal_error=seasonal_error, samples_dim=1, loss_weights=loss_weights)
        metrics_sum = self.get_metrics(target.sum(axis=-1), prediction.sum(axis=-1), samples_dim=1)
        
        # FOR DEBUGGING: THE OLD IMPLEMENTATION -> should produce the same results

        # target = process_tensor(target)
        # prediction = process_tensor(prediction)
        # if self.ignore_invalid_values:
        #     target = np.ma.masked_invalid(target)
        #     prediction = np.ma.masked_invalid(prediction)
        # old_metrics = self.get_metrics_old(target, prediction, seasonal_error=seasonal_error, samples_dim=1, loss_weights=loss_weights)
        # old_metrics_sum = self.get_metrics_old(target.sum(axis=-1), prediction.sum(axis=-1), samples_dim=1)
        # print(f'METRICS {[f"{m}-{(metrics[m], old_metrics[m])}"   for m in metrics.keys()]}')
        # print(f'METRICS SUM {[f"{m}-{(metrics_sum[m], old_metrics_sum[m])}"   for m in metrics_sum.keys()]}')

        # select output metrics
        output_metrics = dict()
        for k in metrics.keys():# self.selected_metrics:
            output_metrics[k] = metrics[k]
            if k in metrics_sum:
                output_metrics[f"{k}-Sum"] = metrics_sum[k]
        return output_metrics
    
def process_tensor(targets):
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().detach().numpy()
    elif isinstance(targets, np.ndarray):
        pass 
    else:
        raise TypeError("targets must be a torch.Tensor or a numpy.ndarray")
    return targets


class Evaluator_q:
    def __init__(self, smooth=False, quantile_levels=[0.01, 0.05, 0.1, 0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 0.95, 0.99]):
        self.smooth = smooth
        self.ignore_invalid_values = True
        self.quantile_levels = quantile_levels

    def loss_name(self, q):
        return f"QuantileLoss[{q}]"

    def weighted_loss_name(self, q):
        return f"wQuantileLoss[{q}]"

    def coverage_name(self, q):
        return f"Coverage[{q}]"


    def get_sequence_metrics(self, targets, forecasts, seasonal_error=None, samples_dim=1,loss_weights=None):
        
        if 0.5 in self.quantile_levels:
            # print(forecasts.shape)
            median_forecasts = forecasts[..., self.quantile_levels.index(0.5)] #np.quantile(forecasts, 0.5, axis=-1)
            mean_forecasts = median_forecasts
        else:
            median_forecasts = np.quantile(forecasts, 0.5, axis=-1)
            mean_forecasts = forecasts.mean(axis=-1)
        
        metrics = {
            "MSE": mse(targets, mean_forecasts),
            "abs_error": abs_error(targets, median_forecasts),
            "abs_target_sum": abs_target_sum(targets),
            "abs_target_mean": abs_target_mean(targets),
            "MAPE": mape(targets, median_forecasts),
            "sMAPE": smape(targets, median_forecasts),
        }
        
        if seasonal_error is not None:
            metrics["MASE"] = mase(targets, median_forecasts, seasonal_error)
        
        metrics["RMSE"] = np.sqrt(metrics["MSE"])
        metrics["NRMSE"] = metrics["RMSE"] / metrics["abs_target_mean"]
        metrics["ND"] = metrics["abs_error"] / metrics["abs_target_sum"]
        
        # calculate weighted loss
        if loss_weights is not None:
            nd = np.abs(targets - mean_forecasts) / np.sum(np.abs(targets), axis=(1, 2))
            loss_weights = loss_weights.detach().unsqueeze(0).unsqueeze(-1).numpy()
            weighted_ND = loss_weights * nd
            metrics['weighted_ND'] = np.sum(weighted_ND)
        else:
            metrics['weighted_ND'] = metrics["ND"]

        for q in self.quantile_levels:
            q_forecasts = forecasts[..., self.quantile_levels.index(q)]#np.quantile(forecasts, q, axis=samples_dim)
            metrics[self.loss_name(q)] = np.sum(quantile_loss(targets, q_forecasts, q))
            metrics[self.weighted_loss_name(q)] = \
                metrics[self.loss_name(q)] / metrics["abs_target_sum"]
            metrics[self.coverage_name(q)] = coverage(targets, q_forecasts)
        
        metrics["mean_absolute_QuantileLoss"] = np.mean(
            [metrics[self.loss_name(q)] for q in self.quantile_levels]
        )
        metrics["CRPS"] = np.mean(
            [metrics[self.weighted_loss_name(q)] for q in self.quantile_levels]
        )
        metrics["MAE_Coverage"] = np.mean(
            [
                np.abs(metrics[self.coverage_name(q)] - np.array([q]))
                for q in self.quantile_levels
            ]
        )
        return metrics

    def get_metrics(self, targets, forecasts, seasonal_error=None, samples_dim=1, loss_weights=None):
        metrics = {}
        seq_metrics = {}
        
        # Calculate metrics for each sequence
        for i in range(targets.shape[0]):
            single_seq_metrics = self.get_sequence_metrics(
                np.expand_dims(targets[i], axis=0),
                np.expand_dims(forecasts[i], axis=0),
                np.expand_dims(seasonal_error[i], axis=0) if seasonal_error is not None else None,
                samples_dim,
                loss_weights
            )
            for metric_name, metric_value in single_seq_metrics.items():
                if metric_name not in seq_metrics:
                    seq_metrics[metric_name] = []
                seq_metrics[metric_name].append(metric_value)
        
        for metric_name, metric_values in seq_metrics.items():
            metrics[metric_name] = np.mean(metric_values)
        return metrics

    def get_metri(self, targets, forecasts, loss_weights=None):
        """
        Parameters:
        targets: torch.Tensor or np.ndarray of shape (batch, prediction_length, time_series)
        forecasts: torch.Tensor or np.ndarray of shape (batch, prediction_length, time_series, quantiles)
        loss_weights: Optional weight tensor for loss computation
        """
        targets, forecasts = process_tensor(targets), process_tensor(forecasts)
        
        batch, pred_len, num_series, num_quantiles = forecasts.shape
        
        mean_forecasts = forecasts.mean(axis=-1)
        if 0.5 in self.quantiles:
            median_forecasts = forecasts[:, :, :, self.quantiles.index(0.5)] #np.quantile(forecasts, 0.5, axis=-1)
            mean_forecasts = median_forecasts
        
        metrics = {
            "MSE": mse(targets, mean_forecasts),
            "abs_error": abs_error(targets, median_forecasts),
            "abs_target_sum": abs_target_sum(targets),
            "abs_target_mean": abs_target_mean(targets),
            "MAPE": mape(targets, median_forecasts),
            "sMAPE": smape(targets, median_forecasts),
            "RMSE": np.sqrt(mse(targets, mean_forecasts)),
        }
        
        metrics["NRMSE"] = metrics["RMSE"] / metrics["abs_target_mean"]
        metrics["ND"] = metrics["abs_error"] / metrics["abs_target_sum"]
        
        if loss_weights is not None:
            loss_weights = process_tensor(loss_weights).reshape(batch, 1, num_series)
            nd = np.abs(targets - mean_forecasts) / np.sum(np.abs(targets), axis=(1, 2))
            metrics['weighted_ND'] = np.sum(loss_weights * nd)
        else:
            metrics['weighted_ND'] = metrics["ND"]
        
        # Compute quantile losses and coverages
        quantile_losses = []
        weighted_quantile_losses = []
        coverages = []
        
        for i, q in enumerate(self.quantiles):
            q_forecasts = forecasts[..., i]
            q_loss = np.sum(quantile_loss(targets, q_forecasts, q))
            weighted_q_loss = q_loss / metrics["abs_target_sum"]
            coverage_val = coverage(targets, q_forecasts)
            
            quantile_losses.append(q_loss)
            weighted_quantile_losses.append(weighted_q_loss)
            coverages.append(coverage_val)
        
        metrics["mean_absolute_QuantileLoss"] = np.mean(quantile_losses)
        metrics["CRPS"] = np.mean(weighted_quantile_losses)
        metrics["MAE_Coverage"] = np.mean(np.abs(np.array(coverages) - np.linspace(0.05, 0.95, num_quantiles)))
        
        return metrics

    @property
    def selected_metrics(self):
        return [ "ND",'weighted_ND', 'CRPS', "NRMSE", "MSE", "MASE"]
    
    def __call__(self, targets, forecasts, past_data, freq, loss_weights=None):
        """

        Parameters
        ----------
        targets
            groundtruth in (batch_size, prediction_length, target_dim)
        forecasts
            forecasts in (batch_size,  prediction_length, target_dim, quantile_level)
        Returns
        -------
        Dict[String, float]
            metrics
        """
        targets = process_tensor(targets)
        forecasts = process_tensor(forecasts)
        past_data = process_tensor(past_data)
        
        if self.ignore_invalid_values:
            targets = np.ma.masked_invalid(targets)
            forecasts = np.ma.masked_invalid(forecasts)
        
        seasonal_error = calculate_seasonal_error(past_data, freq)


        metrics = self.get_metrics(targets, forecasts, loss_weights)
        metrics = self.get_metrics(targets, forecasts, seasonal_error=seasonal_error, samples_dim=3, loss_weights=loss_weights)
        metrics_sum = self.get_metrics(targets.sum(axis=-1), forecasts.sum(axis=-2), samples_dim=1)
        
        # select output metrics
        output_metrics = dict()
        for k in self.selected_metrics:
            output_metrics[k] = metrics[k]
            if k in metrics_sum:
                output_metrics[f"{k}-Sum"] = metrics_sum[k]
        return output_metrics