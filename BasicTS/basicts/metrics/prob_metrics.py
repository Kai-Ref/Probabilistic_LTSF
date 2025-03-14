# differentiate between distributional and quantile metrics 
# for quantile metrics -> iterate over the quantiles of the forecast (and percentiles?)

import numpy as np
import torch
from typing import Optional
from scipy.integrate import quad
from scipy.stats import norm




# The following is taken from properscoring
# See: https://github.com/properscoring/properscoring/blob/master/properscoring/_crps.py

import scipy.special as special
# Normalization constant for standard Gaussian PDF
_normconst = 1.0 / np.sqrt(2.0 * np.pi)

def _normpdf(x):
    """Standard normal probability density function (PDF)."""
    return _normconst * torch.exp(-(x * x) / 2.0)

# Standard normal cumulative distribution function (CDF)
def _normcdf(x):
    return torch.tensor(special.ndtr(x), dtype=x.dtype, device=x.device).clone().detach()

def crps(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """
    Compute the CRPS for a Gaussian distribution.

    Parameters:
    - prediction: Tensor of shape [batch_size, n_vars, seq_len, 2] containing mean & std.
    - target: Tensor of shape [batch_size, n_vars, seq_len] with observed values.
    - null_val: Value representing missing or invalid data. Default is NaN.

    Returns:
    - A scalar tensor representing the CRPS loss.
    """
    target = target.clone().squeeze(-1).detach().cpu()
    # Handle missing values
    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    # Extract mean and standard deviation from prediction
    mu = prediction[..., 0].detach().cpu()  # Mean
    sig = prediction[..., 1].detach().cpu()  # Standard deviation
    sig = torch.clamp(sig, min=1e-6)

    # Compute standardized target
    sx = (target - mu) / sig
    sx = sx.detach().cpu()
    pdf = _normpdf(sx)
    cdf = _normcdf(sx)
    pi_inv = 1.0 / np.sqrt(np.pi)
    crps = sig.detach().cpu() * (sx * (2 * cdf - 1) + 2 * pdf - pi_inv)

    # This would additionally returnt the grad wrt to mu and sig
    # if grad:
    #     dmu = 1 - 2 * cdf
    #     dsig = 2 * pdf - pi_inv
    #     return crps, np.array([dmu, dsig])
    crps = crps * mask
    return torch.mean(crps)


# TODO
def empirical_crps(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    return np.mean(y_pred_samples <= x)


def crps_old(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """
    Compute the Continuous Ranked Probability Score (CRPS) for a Gaussian distribution.

    Parameters:
    - prediction: Tensor of shape [batch_size, n_vars, seq_len, 2], where the last dimension contains
                  the mean (mus) and standard deviation (sigmas) of the predicted Gaussian distribution.
    - target: Tensor of shape [batch_size, n_vars, seq_len], containing the observed values.
    - null_val: Value representing missing or invalid data. Default is NaN.

    Returns:
    - A single CRPS value (scalar) for the entire batch.
    """
    # Handle missing values
    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    # Extract mean (mus) and standard deviation (sigmas)
    mus = prediction[..., 0]  # Shape: [batch_size, n_vars, seq_len]
    sigmas = prediction[..., 1]  # Shape: [batch_size, n_vars, seq_len]
    sigmas = torch.clamp(sigmas, min=1e-6)  # Ensure std is positive

    # Flatten tensors for vectorized computation
    mus_flat = mus.flatten().cpu().detach().numpy()  # Convert to NumPy for scipy integration
    sigmas_flat = sigmas.flatten().detach().cpu().numpy()
    target_flat = target.flatten().cpu().numpy()
    mask_flat = mask.flatten().cpu().numpy()

    # Vectorized CRPS computation
    def crps_integral(mu, sigma, y_true):
        """Compute CRPS for a single Gaussian distribution and observed value."""
        def integrand(x):
            return (norm.cdf(x, loc=mu, scale=sigma) - (x >= y_true)) ** 2
        crps_value, _ = quad(integrand, -np.inf, np.inf)
        return crps_value

    # Compute CRPS for all elements using vectorized operations
    crps_values = np.array([crps_integral(mu, sigma, y_true) for mu, sigma, y_true in zip(mus_flat, sigmas_flat, target_flat)])

    # Apply mask and compute the mean CRPS
    crps_values_masked = crps_values * mask_flat
    mean_crps = torch.mean(crps_values_masked)
    return mean_crps

def quantile_loss(prediction: torch.Tensor, target: torch.Tensor, quantiles, null_val: float = np.nan):
    """
    If quantiles are given as a list, the function assumes the predictions to encompass all quantile levels.
    If quantiles are given as a tensor, they are matched with the predictions shape, allowing for quantile level predictions that differ across batch elements.

    Parameters:
        y_pred (torch.Tensor): Predicted quantile values, shape [bs x n_vars x target_window x 1].
        y_true (torch.Tensor): Ground truth values, shape [bs x n_vars x target_window].
        quantiles (list or torch.Tensor): List of quantiles (e.g., [0.1, 0.5, 0.9]).

    Returns:
        torch.Tensor: Scalar loss value.
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    if isinstance(quantiles, list):
        quantiles = torch.tensor(quantiles) # Shape: [num_quantiles]

        # Dynamically get `num_quantiles` from list length
        num_quantiles = quantiles.shape[0]

        # Get dimensions from `prediction` or `target`
        batch_size, _, time_steps, _ = prediction.shape  # Extract dimensions dynamically

        # Reshape and expand quantiles without hardcoding
        quantiles = quantiles.view(1, 1, 1, num_quantiles).expand(batch_size, 1, time_steps, num_quantiles).to(prediction.device) # Shape: [64, 1, 7, num_quantiles]
    else:
        quantiles = quantiles.unsqueeze(1).unsqueeze(-1)  # Shape: [64, 1, 7, 1]
    errors = target - prediction  # Shape: [64, 96, 7, 1]
    loss = torch.max(quantiles * errors, (quantiles - 1) * errors)  # Shape: [64, 96, 7, 1]
    #loss = loss * mask  # Shape: [64, 96, 7, 1]
    return loss.mean()

def quantile_loss_old(prediction: torch.Tensor, target: torch.Tensor, quantiles, null_val: float = np.nan):
    """
    Computes the quantile loss for multiple quantiles.

    Parameters:
        y_pred (torch.Tensor): Predicted quantile values, shape [bs x n_vars x target_window x num_quantiles].
        y_true (torch.Tensor): Ground truth values, shape [bs x n_vars x target_window].
        quantiles (list): List of quantiles (e.g., [0.1, 0.5, 0.9]).

    Returns:
        torch.Tensor: Scalar loss value.
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    losses = []
    for i, q in enumerate(quantiles):
        errors = target[..., 0] - prediction[..., i]  # Compute residuals (true - predicted)
        loss = torch.max(q * errors, (q - 1) * errors)  # Pinball loss per quantile
        #loss_masked = loss * mask
        losses.append(loss.mean())  # Average over all samples
    return torch.mean(torch.stack(losses))  # Average across all quantiles

def gaussian_nll_loss(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan): #prediction: torch.Tensor, target: torch.Tensor, std: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    Calculate the Negative Log-Likelihood (NLL) loss for Gaussian distributions.

    Args:
        prediction (torch.Tensor): The predicted mean values as a tensor.
        target (torch.Tensor): The ground truth values as a tensor with the same shape as `prediction`.
        std (torch.Tensor): The predicted standard deviation values as a tensor with the same shape as `prediction`.
        reduction (str, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
            'mean': the sum of the output will be divided by the number of elements in the output.
            'sum': the output will be summed.
            'none': no reduction will be applied. Defaults to 'mean'.

    Returns:
        torch.Tensor: The NLL loss.

    Note:
        Taken from gaussian.py of DeepAR.
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    # Ensure std is positive
    mus = prediction[..., 0].unsqueeze(-1)              # [bs x nvars x seq_len x 1]
    sigmas = prediction[..., 1].unsqueeze(-1)           # [bs x nvars x seq_len x 1]
    sigmas = torch.clamp(sigmas, min=1e-6)


    distribution = torch.distributions.Normal(mus, sigmas)
    likelihood = distribution.log_prob(target)
    likelihood = likelihood * mask
    loss_g = -torch.mean(likelihood)
    return loss_g