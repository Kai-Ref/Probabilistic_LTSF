# differentiate between distributional and quantile metrics 
# for quantile metrics -> iterate over the quantiles of the forecast (and percentiles?)

import numpy as np
import torch
from typing import Optional
from scipy.integrate import quad
from scipy.stats import norm
import torch.distributions as dist



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

def quantile_loss_old(prediction: torch.Tensor, target: torch.Tensor, quantiles: list, null_val: float = np.nan):
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

def nll_loss(prediction: torch.Tensor, target: torch.Tensor, distribution_type: str, null_val: float = np.nan): #prediction: torch.Tensor, target: torch.Tensor, std: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    Generalized Negative Log-Likelihood (NLL) Loss for different probabilistic distributions.

    Args:
        prediction (torch.Tensor): The predicted distribution parameters.
        target (torch.Tensor): The ground truth values.
        distribution_type (str): The type of distribution (e.g., 'gaussian', 'laplace', 'poisson', etc.).
        null_val (float, optional): The value representing missing values in `target`. Defaults to NaN.

    Returns:
        torch.Tensor: The computed NLL loss.

    Note:
        Taken from gaussian.py of DeepAR.
    """
    target = target.squeeze(-1)
    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    # 2. Compute Negative Log-Likelihood (NLL) for each distribution
    if distribution_type == "gaussian":
        mu, sigma = prediction[..., 0], prediction[..., 1]
        sigma = torch.clamp(sigma, min=1e-6)  # Ensure std > 0
        distribution = dist.Normal(mu, sigma)
    elif distribution_type == "laplace":
        loc, scale = prediction[..., 0], prediction[..., 1]
        scale = torch.clamp(scale, min=1e-6)
        distribution = dist.Laplace(loc, scale)
    elif distribution_type == "student_t":
        mu, sigma, df = prediction[..., 0], prediction[..., 1], prediction[..., 2]
        sigma = torch.clamp(sigma, min=1e-6)
        df = torch.clamp(df, min=1.1)  # Ensure degrees of freedom > 1
        distribution = dist.StudentT(df, mu, sigma)
    elif distribution_type == "lognormal":
        mu, sigma = prediction[..., 0], prediction[..., 1]
        sigma = torch.clamp(sigma, min=1e-6)
        distribution = dist.LogNormal(mu, sigma)
    elif distribution_type == "beta":
        alpha, beta = prediction[..., 0], prediction[..., 1]
        alpha = torch.clamp(alpha, min=1e-6)
        beta = torch.clamp(beta, min=1e-6)
        distribution = dist.Beta(alpha, beta)
    elif distribution_type == "gamma":
        shape, rate = prediction[..., 0], prediction[..., 1]
        shape = torch.clamp(shape, min=1e-6)
        rate = torch.clamp(rate, min=1e-6)
        distribution = dist.Gamma(shape, rate)
    elif distribution_type == "weibull":
        scale, concentration = prediction[..., 0], prediction[..., 1]
        scale = torch.clamp(scale, min=1e-6)
        concentration = torch.clamp(concentration, min=1e-6)
        distribution = dist.Weibull(scale, concentration)
    elif distribution_type == "poisson":
        rate = prediction[..., 0]
        rate = torch.clamp(rate, min=1e-6)
        distribution = dist.Poisson(rate)
    elif distribution_type == "negative_binomial":
        rate, dispersion = prediction[..., 0], prediction[..., 1]
        rate = torch.clamp(rate, min=1e-6)
        dispersion = torch.clamp(dispersion, min=1e-6)
        probs = rate / (rate + dispersion)  # Convert to success probability
        distribution = dist.NegativeBinomial(total_count=dispersion, probs=probs)
    elif distribution_type == "dirichlet":
        concentration = prediction
        concentration = torch.clamp(concentration, min=1e-6)
        distribution = dist.Dirichlet(concentration)
    else:
        raise ValueError(f"Unsupported distribution type: {distribution_type}")
    log_likelihood = distribution.log_prob(target)
    # 3. Apply mask and return loss
    log_likelihood = log_likelihood * mask
    loss = -torch.mean(log_likelihood)
    
    return loss