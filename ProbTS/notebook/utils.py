import numpy as np
import matplotlib.pyplot as plt
import torch
import os 
os.chdir("/home/kreffert/Probabilistic_LTSF/ProbTS")
from probts.data import ProbTSBatchData

def get_predictions_1(cli):
    test_dataloader = cli.datamodule.test_dataloader()
    model = cli.model
    model.eval()  # Ensure model is in eval mode
    device = 'cpu'
    model.to(device)  # Move to appropriate device
    mean_predictions = []
    std_predictions = []
    past_actuals = []
    future_actuals = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            batch = {key: val.to(device) for key, val in batch.items()}  # Move batch to device
            # Extract past actual values
            past_values = batch["past_target_cdf"].cpu()  # Shape: (batch_size, past_seq_len)
            print(past_values.shape)
            past_actuals.append(past_values)
    
            # Extract future actual values
            future_values = batch["future_target_cdf"].cpu()  # Shape: (batch_size, future_seq_len)
            future_actuals.append(future_values)
            print(future_values.shape)
    
            # Store predictions
            #samples_forecast = model.predict_step(batch, batch_idx)
            batch_data = ProbTSBatchData(batch, device)
            mean_forecast, std_forecast = cli.model.forecaster.forecast(batch_data, num_samples=None)
            print(mean_forecast.shape)
            mean_forecast = mean_forecast.squeeze(1)
            std_forecast = std_forecast.squeeze(1)
            mean_predictions.append(mean_forecast.cpu())
            std_predictions.append(std_forecast.cpu())
    
    # Convert to numpy if needed
    mean_predictions = torch.cat(mean_predictions, dim=0).numpy()
    std_predictions = torch.cat(std_predictions, dim=0).numpy()
    past_actuals = torch.cat(past_actuals, dim=0).numpy()
    future_actuals = torch.cat(future_actuals, dim=0).numpy()
    return mean_predictions, std_predictions, past_actuals, future_actuals

def get_predictions(cli):
    test_dataloader = cli.datamodule.test_dataloader()
    model = cli.model
    model.eval()  # Ensure model is in eval mode
    device = 'cpu'
    model.to(device)  # Move to appropriate device
    predictions = []
    past_actuals = []
    future_actuals = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            batch = {key: val.to(device) for key, val in batch.items()}  # Move batch to device
            # Extract past actual values
            past_values = batch["past_target_cdf"].cpu()  # Shape: (batch_size, past_seq_len)
            print(past_values.shape)
            past_actuals.append(past_values)
    
            # Extract future actual values
            future_values = batch["future_target_cdf"].cpu()  # Shape: (batch_size, future_seq_len)
            future_actuals.append(future_values)
            print(future_values.shape)
    
            # Store predictions
            #forecast = model.predict_step(batch, batch_idx)
            batch_data = ProbTSBatchData(batch, device)
            forecast = cli.model.forecaster.forecast(batch_data, num_samples=None)
            print(forecast.shape)
            forecast = forecast.squeeze(1)
            predictions.append(forecast.cpu())
            print(forecast.shape)
    
    # Convert to numpy if needed
    predictions = torch.cat(predictions, dim=0).numpy()
    past_actuals = torch.cat(past_actuals, dim=0).numpy()
    future_actuals = torch.cat(future_actuals, dim=0).numpy()
    return predictions, past_actuals, future_actuals




def plot_time_series(past_actuals, future_actuals, predictions, windows, series, ci=[5, 95], quantile_levels=[]):
    num_rows = len(series)  # One row per feature
    num_cols = len(windows)  # One column per time series index

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 3), sharex=True, sharey=False)

    if num_rows == 1 or num_cols == 1:
        axes = np.array(axes).reshape(num_rows, num_cols)  # Ensure 2D indexing

    for row, serie in enumerate(series):
        for col, window in enumerate(windows):
            ax = axes[row, col]  # Get the correct subplot

            # Plot past actuals
            ax.plot(range(len(past_actuals[window])), past_actuals[window, :, serie], label="Past", color='blue')

            # Plot future actuals
            ax.plot(range(len(past_actuals[window]), len(past_actuals[window]) + len(future_actuals[window])),
                    future_actuals[window, :, serie], label="Future", color='green')

            if ci != None:
                if len(quantile_levels)>1:# quantile forecast
                    ci = [i / 100 if i > 1 else i for i in ci]
                    assert ci[0] in quantile_levels
                    assert ci[1]in quantile_levels
                    assert 0.5 in quantile_levels 
                    lower_bound = predictions[window, :, serie, quantile_levels.index(ci[0])]
                    upper_bound = predictions[window, :, serie, quantile_levels.index(ci[1])]
                    median_pred = predictions[window, :, serie, quantile_levels.index(0.5)]
                else:# non quantile prob forecast -> need to calculate the quantiles first
                    lower_bound = np.percentile(predictions[window, :, :, serie], ci[0], axis=0)
                    upper_bound = np.percentile(predictions[window, :, :, serie], ci[1], axis=0)
                    median_pred = np.percentile(predictions[window, :, :, serie], 50, axis=0)
    
                pred_range = range(len(past_actuals[window]), len(past_actuals[window]) + len(median_pred))
                ax.fill_between(pred_range, lower_bound, upper_bound, color='red', alpha=0.3, label=f"{ci[0]}-{ci[1]}% CI")
                ax.plot(pred_range, median_pred, color='red', linestyle='dashed', label="Median Prediction")
            else:# point forecasts
                ax.plot(range(len(past_actuals[window]), len(past_actuals[window]) + len(predictions[window])),
                    predictions[window, :, serie], label="Prediction", color='red', linestyle='dashed')
            
            ax.axvline(x=len(past_actuals[window]), color='black', linestyle='dotted')  # Mark forecast start
            if row == 0:
                ax.set_title(f"Window {window}")
            if col == 0:
                ax.set_ylabel(f"Time Series {serie}")
            ax.legend(fontsize=6)
    plt.tight_layout()
    plt.show()


def plot_qq_coverage(future_actuals, predictions, windows, series, quantile_levels=[0.05, 0.5, 0.95], lower=False, quantile_regression=False):
    """
    Q-Q plot between the quantiles and count of coverage. 
    For example a single point represents the coverage of a certain quantile.
    -> for the 0.3 quantile prediction, ideally 30% of values should be below that predicted value.
    Parameters:
        future_actuals (ndarray): Ground truth future values (shape: [windows, time, series]).
        predictions (ndarray): Forecast samples (shape: [windows, samples, time, series]).
        windows (list): List of window indices to plot.
        series (list): List of series indices to plot.
        quantile_levels (list): List of quantiles to evaluate.
        lower (bool): If True, calculates coverage as [q_value>actuals], else [q_value>=actuals]
    """
    num_rows = len(series)  # One row per feature
    num_cols = len(windows)  # One column per time series index
    quantile_levels = sorted(quantile_levels)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 3), sharex=True, sharey=True)

    if num_rows == 1 or num_cols == 1:
        axes = np.array(axes).reshape(num_rows, num_cols)  # Ensure 2D indexing
    for row, serie in enumerate(series):
        for col, window in enumerate(windows):
            ax = axes[row, col]  # Get the correct subplot

            # Compute actual quantiles (empirical quantiles from the true future data)
            actual_quantiles = np.percentile(future_actuals[window, :, serie], quantile_levels*100)

            coverages = []
            for q in quantile_levels:
                if quantile_regression:
                    predicted_quantiles = predictions[window, :, serie, quantile_levels.index(q)]#TODO: will return wrong quantiles if not all quantiles are given
                else:
                    predicted_quantiles = np.percentile(predictions[window, :, :, serie], q*100, axis=0)#.mean(axis=1)
                
                if lower:
                    count = np.sum(predicted_quantiles>future_actuals[window, :, serie])
                else:
                    count = np.sum(predicted_quantiles>future_actuals[window, :, serie])
                coverage = count/len(predicted_quantiles)
                coverages.append(coverage)
                
            # Plot Q-Q cumulative scatter plot (flip the axes)
            ax.scatter(quantile_levels, coverages, color='blue', label="Q-Q points")

            # Plot the diagonal (ideal calibration line)
            ax.plot([0, 1],
                    [0, 1], 
                    linestyle="dashed", color="red", label="Ideal Line")

            # ax.legend(fontsize=6)
            if row == 0:
                ax.set_title(f"Window {window}")
            if col == 0:
                ax.set_ylabel(f"Coverage for Series {serie}")
            if row == len(series):
                ax.set_xlabel(f"Quantiles")

    plt.tight_layout()
    plt.show()
