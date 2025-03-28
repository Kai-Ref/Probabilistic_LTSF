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
    print(f"OLD {crps.shape}")
    print(torch.mean(crps))
    return torch.mean(crps)
