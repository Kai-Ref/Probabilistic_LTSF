import torch
import torch.nn as nn
import torch.distributions as dist
from .dist_heads import *
from .flow_heads import FlowHead

class ProbabilisticHead(nn.Module):
    """Main class that dynamically selects the distribution head."""
    DISTRIBUTIONS = {
        "gaussian": GaussianHead,  # Symmetric, bell-shaped distribution defined by mean (μ) and standard deviation (σ). Suitable for continuous data with normal errors.
        "laplace": LaplaceHead,  # Similar to Gaussian but with heavier tails. Defined by location (μ) and scale (b). Useful for modeling data with outliers.
        "student_t": StudentTHead,  # Like Gaussian but with heavier tails, controlled by degrees of freedom (ν). More robust to outliers.
        "lognormal": LogNormalHead,  # Distribution where the logarithm of the variable is normally distributed. Used for positive, skewed data (e.g., financial returns).
        "beta": BetaHead,  # Defined on the interval [0,1], controlled by two shape parameters (α, β). Used for modeling probabilities or proportions.
        "gamma": GammaHead,  # Defined for positive values, controlled by shape (k) and scale (θ). Used for modeling waiting times or rainfall amounts.
        "weibull": WeibullHead,  # Flexible distribution for modeling lifetimes and reliability analysis, defined by scale (λ) and shape (k).
        "poisson": PoissonHead,  # Discrete distribution modeling count data (e.g., number of arrivals per time period). Defined by rate (λ).
        "negative_binomial": NegativeBinomialHead,  # Models overdispersed count data, generalizing Poisson by allowing variance to exceed the mean.
        "dirichlet": DirichletHead,  # A distribution over probability vectors (e.g., multinomial proportions). Useful in Bayesian modeling and topic modeling.
        "quantile": QuantileHead, 
        "i_quantile": ImplicitQuantileHead,
        "m_gaussian": MultivariateGaussianHead,
        "m_lr_gaussian": LowRankMultivariateGaussianHead,
        "flow": FlowHead,
        "copula": CopulaHead,  # Copula-based multivariate head (see args for copula/marginal config)
        "gcopula": GaussianCopulaHead,  # Copula-based multivariate head (see args for copula/marginal config)
    }

    def __init__(self, input_dim, output_dim, distribution_type="gaussian", prob_args={}):
        super().__init__()
        if distribution_type not in self.DISTRIBUTIONS:
            raise ValueError(f"Unsupported distribution type: {distribution_type}")
        self.head = self.DISTRIBUTIONS[distribution_type](input_dim, output_dim, prob_args=prob_args)

    def forward(self, x, **kwargs):
        return self.head(x, **kwargs)
    
    def __get_dist__(self, prediction):
        return self.head.__get_dist__(prediction)

    def sample(self, head_output, num_samples=1, random_state=None):# TODO implement the functionality of sampling more than one sample -> consider desired shape of output
        return self.head.sample(head_output, num_samples, random_state=random_state)  


