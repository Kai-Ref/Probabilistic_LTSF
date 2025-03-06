import numpy as np
import matplotlib.pyplot as plt
import torch
import logging
from tqdm import tqdm
from scipy.stats import norm

def init(model, dataset, no_logging=False):
    if no_logging:
        logging.disable(logging.CRITICAL)  # Temporarily disable all logging
    #print("This will print, but logs are disabled")

    # Run a baseline model in BasicTS framework.
    # pylint: disable=wrong-import-position
    import os
    import sys
    from argparse import ArgumentParser
    
    __file__ = "/home/kreffert/Probabilistic_LTSF/BasicTS/experiments/train.py"
    sys.path.append(os.path.abspath(__file__ + '/../..'))
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    sys.argv = [__file__]
    
    import torch
    import basicts
    from easytorch.utils import set_visible_devices, get_logger
    from easytorch.config import init_cfg
    
    torch.set_num_threads(4) # aviod high cpu avg usage
    
    def parse_args():
        parser = ArgumentParser(description='Run time series forecasting model in BasicTS framework!')
        parser.add_argument('-c', '--cfg', default=f'baselines/{model}/{dataset}.py ', help='training config')
        parser.add_argument('-g', '--gpus', default='0', help='visible gpus')
        return parser.parse_args()
    
    args = parse_args()

    cfg = cfg = init_cfg(args.cfg, save=True)
    logger = get_logger('easytorch-launcher')
    logger.info('Initializing runner "{}"'.format(cfg['RUNNER']))
    runner = cfg['RUNNER'](cfg)
    runner.init_logger(logger_name='easytorch-training', log_file_name='training_log')


    runner.init_training(cfg)
    # Your function code here...
    if no_logging:
        logging.disable(logging.NOTSET)  # Re-enable logging
    return runner



def get_predictions(runner, data_loader="test"):
    if data_loader == "test":
        dataloader = runner.test_data_loader
    elif data_loader == "val":
        dataloader = runner.val_data_loader
    else:
        dataloader = runner.train_data_loader
    model = runner.model
    model.eval()  # Ensure model is in eval mode
    device = 'cuda:0'
    model.to(device)  # Move to appropriate device
    predictions = []
    past_actuals = []
    future_actuals = []
    
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader)):
            history = batch['inputs'][..., :1]     # B, L, N -> i.e. ignore the remaining features
            targets = batch['target'][..., :1]
            
            # TODO scaler
            model_return = runner.forward(batch, epoch=100, iter_num=100, train=False)
            #forecasts = model(batch['inputs'][..., :1].to(device), batch['target'].to(device), batch_seen=0, epoch=0, train=False)
            history = model_return['inputs']
            targets = model_return['target']
            forecasts = model_return['prediction']
            
            history, targets, forecasts = history.to('cpu'), targets.to('cpu'), forecasts.to('cpu') 
            future_actuals.append(targets)
            past_actuals.append(history)
            predictions.append(forecasts)
    
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
                    # Extract mean and std
                    median_pred = predictions[window, :, serie, 0]  # Shape: [num_time_steps]
                    std = predictions[window, :, serie, 1]   # Shape: [num_time_steps]
                    
                    # Create a grid of x values
                    #x = range(predictions.shape[1])
                    # Compute the PDFs for all time steps at once
                    #pdfs = norm.pdf(x, loc=median_pred, scale=std)  # Shape: [num_points, num_time_steps]
                    
                    #ax.plot(x, pdfs)#label=[f"Time Step {t+1} (μ={mean[t]}, σ={std[t]})" for t in range(predictions.shape[0])])
                    
                    lower_bound = median_pred + 2*std #np.percentile(predictions[window, :, :, serie], ci[0], axis=0)
                    upper_bound = median_pred - 2*std #np.percentile(predictions[window, :, :, serie], ci[1], axis=0)
                    #median_pred = np.percentile(predictions[window, :, :, serie], 50, axis=0)
    
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
