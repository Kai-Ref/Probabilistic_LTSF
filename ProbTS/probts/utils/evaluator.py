import numpy as np
from .metrics import *
import torch

class Evaluator:
    def __init__(self, quantiles_num=10, smooth=False):
        self.quantiles = (1.0 * np.arange(quantiles_num) / quantiles_num)[1:]
        self.ignore_invalid_values = True
        self.smooth = smooth

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

        for q in self.quantiles:
            q_forecasts = np.quantile(forecasts, q, axis=samples_dim)
            metrics[self.loss_name(q)] = np.sum(quantile_loss(targets, q_forecasts, q))
            metrics[self.weighted_loss_name(q)] = \
                metrics[self.loss_name(q)] / metrics["abs_target_sum"]
            metrics[self.coverage_name(q)] = coverage(targets, q_forecasts)
        
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
            forecasts in (batch_size, num_samples, prediction_length, target_dim)
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

        metrics = self.get_metrics(targets, forecasts, seasonal_error=seasonal_error, samples_dim=1, loss_weights=loss_weights)
        metrics_sum = self.get_metrics(targets.sum(axis=-1), forecasts.sum(axis=-1), samples_dim=1)
        
        # select output metrics
        output_metrics = dict()
        for k in self.selected_metrics:
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