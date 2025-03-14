import torch
import torch.nn as nn
import torch.distributions as dist

class BaseDistribution(nn.Module):
    """Abstract base class for different probabilistic heads."""
    def __init__(self, input_dim, output_dim, args):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward method.")

    def sample(self, head_output, num_samples=1):
        """Sample from the probabilistic head output."""
        raise NotImplementedError("Subclasses must implement sample method.")

class GaussianHead(BaseDistribution):
    """Gaussian distribution output: mean and variance."""
    def __init__(self, input_dim, output_dim, args):
        super().__init__(input_dim, output_dim, args)
        self.mean_layer = nn.Linear(input_dim, output_dim)
        self.std_layer = nn.Linear(input_dim, output_dim)
        self.activation = nn.Softplus()

    def forward(self, x):
        mu = self.mean_layer(x)
        sigma = self.activation(self.std_layer(x))
        return torch.stack([mu, sigma], dim=-1)

    def sample(self, head_output, num_samples=1):
        mu, sigma = head_output[..., 0], head_output[..., 1]
        return dist.Normal(mu, sigma).rsample((num_samples,))

class LaplaceHead(GaussianHead):
    """
    Laplace distribution has heavier than Gaussian. Consists of two parameters loc and scale, where scale is required to be >0.
    Uses the same structure as the Gaussian Head.
    Laplace distribution output: location and scale."""
    def sample(self, head_output, num_samples=1):
        loc, scale = head_output[..., 0], head_output[..., 1]
        return dist.Laplace(loc, scale).rsample()

class QuantileHead(BaseDistribution):
    """Quantile regression head."""
    def __init__(self, input_dim, output_dim, args):
        super().__init__(input_dim, output_dim, args)
        self.quantiles = args['quantiles']
        assert args['quantiles'] is not None, "quantiles must be specified for quantile regression"
        self.layers = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(len(self.quantiles))])

    def forward(self, x):
        return torch.stack([layer(x) for layer in self.layers], dim=-1)

    def sample(self, head_output, num_samples=1):
        """Uses the median quantile as the point estimate."""
        median_idx = self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2
        return head_output[..., median_idx]

class ImplicitQuantileHead(BaseDistribution):
    """Implicit Quantile Network (IQN) for quantile regression."""
    def __init__(self, input_dim, output_dim, args):
        super().__init__(input_dim, output_dim, args)
        self.num_layers = 1
        self.quantiles = args['quantiles']
        self.quantile_embed_dim = 64
        self.cos_embedding_dim = self.quantile_embed_dim
        self.quantile_embedding = nn.Linear(self.cos_embedding_dim, self.quantile_embed_dim)
        self.qr_head = nn.Linear(input_dim + self.quantile_embed_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        if self.training:
            u = torch.rand(batch_size, 1).to(x.device)  # # [batch_size, 1] - One random quantile level per element
            # IQN Cosine Embedding for Quantile Processing
            i = torch.arange(1, self.cos_embedding_dim + 1, device=x.device).float().unsqueeze(0)  # [1, embed_dim]
            cos_features = torch.cos(torch.pi * u * i)  # Cosine embedding
            phi_u = self.activation(self.quantile_embedding(cos_features))  # Learnable transformation
            if len(x.shape)<3:# this is the case when individual is used
                x = x.unsqueeze(dim=1) # shape: [64, 1, 60]
            phi_u = phi_u.unsqueeze(1).expand(-1, x.shape[1], -1)  # Shape: [64, x.shape[1], 60]
            # Concatenate quantile representation to input x
            x_augmented = torch.cat([x, phi_u], dim=-1).squeeze() # [batch_size, output_dim]
            predictions = self.qr_head(x_augmented)  # [batch_size, output_dim]
            if predictions.dim() != u.dim():
                u = u.unsqueeze(-1)
                batch_size, num_series, length = predictions.shape
                u = u.expand(batch_size, num_series, 1)
            return torch.cat([predictions, u], dim=-1).unsqueeze(-1)  # Append tau to output
        else:
            #u = self.quantiles.to(x.device).repeat(batch_size, 1)  # Fixed quantile levels for inference
            quantile_levels = torch.tensor(self.quantiles, device=x.device).float()  # Convert to tensor
            predictions = []
            with torch.no_grad():
                for u in quantile_levels:
                    u = torch.full((x.size(0), 1), u, device=x.device)  # Same tau for entire batch
                    # IQN Cosine Embedding for Quantile Processing
                    i = torch.arange(1, self.cos_embedding_dim + 1, device=x.device).float().unsqueeze(0)  # [1, embed_dim]
                    cos_features = torch.cos(torch.pi * u * i)  # Cosine embedding
                    phi_u = self.activation(self.quantile_embedding(cos_features))  # Learnable transformation
                    if len(x.shape)<3:# this is the case when individual is used
                        x = x.unsqueeze(dim=1) # shape: [64, 1, 60]
                    phi_u = phi_u.unsqueeze(1).expand(-1, x.shape[1], -1)  # Shape: [64, x.shape[1], 60]
                    # Concatenate quantile representation to input x
                    x_augmented = torch.cat([x, phi_u], dim=-1).squeeze()
                    pred = self.qr_head(x_augmented)  # [batch_size, num_quantiles, output_dim]
                    #return torch.cat([pred, u.unsqueeze(-1)], dim=-1)  # Append tau to output
                    predictions.append(pred)
            predictions = torch.stack(predictions, dim=1) # [batch_size, num_quantiles, (num_series), output_dim] -> num_series not always present, but for example when individual=False
            if predictions.dim() == 3:
                predictions = predictions.permute(0,2,1)  # [batch_size, output_dim, num_quantiles]
            else:
                predictions = predictions.permute(0, 2, 3, 1) # [batch_size, num_series, output_dim, num_quantiles]
            return predictions
    
    def sample(self, head_output, num_samples=1):
        """Select the 0.5 quantile (median) if available, else fallback to the midpoint."""
        # if self.training:
        #     return head_output[..., 0]  # Use the direct prediction during training
        # else:
        #     median_idx = len(self.quantiles) // 2 if self.quantiles else 0
        #     return head_output[..., median_idx]
        return NotImplementedError("Need to decide on a way to sample from IQN.")

class StudentTHead(GaussianHead):
    """Student's t-distribution for heavy-tailed data."""
    def __init__(self, input_dim, output_dim, args):
        super().__init__(input_dim, output_dim, args)
        self.df_layer = nn.Linear(input_dim, output_dim)
        self.df_activation = nn.Softplus()  # Ensure degrees of freedom > 0

    def forward(self, x):
        mu = self.mean_layer(x)
        sigma = self.activation(self.std_layer(x))
        df = self.df_activation(self.df_layer(x)) + 1  # Ensure df > 1
        return torch.stack([mu, sigma, df], dim=-1)

    def sample(self, head_output, num_samples=1):#TODO sampling throws an error
        mu, sigma, df = head_output[..., 0], head_output[..., 1], head_output[..., 2]
        return dist.StudentT(df, mu, sigma).rsample([num_samples])

class LogNormalHead(GaussianHead):
    """Log-normal distribution for positive-valued time series."""
    def sample(self, head_output, num_samples=1):
        mu, sigma = head_output[..., 0], head_output[..., 1]
        return dist.LogNormal(mu, sigma).rsample([num_samples])

class BetaHead(BaseDistribution):
    """Beta distribution for forecasting probabilities (0,1)."""
    def __init__(self, input_dim, output_dim, args):
        super().__init__(input_dim, output_dim, args)
        self.alpha_layer = nn.Linear(input_dim, output_dim)
        self.beta_layer = nn.Linear(input_dim, output_dim)
        self.activation = nn.Softplus()  # Ensure alpha, beta > 0

    def forward(self, x):
        alpha = self.activation(self.alpha_layer(x)) + 1e-6
        beta = self.activation(self.beta_layer(x)) + 1e-6
        return torch.stack([alpha, beta], dim=-1)

    def sample(self, head_output, num_samples=1):
        alpha, beta = head_output[..., 0], head_output[..., 1]
        return dist.Beta(alpha, beta).rsample([num_samples])

class GammaHead(BaseDistribution):
    """Gamma distribution for skewed positive-valued data."""
    def __init__(self, input_dim, output_dim, args):
        super().__init__(input_dim, output_dim, args)
        self.shape_layer = nn.Linear(input_dim, output_dim)
        self.rate_layer = nn.Linear(input_dim, output_dim)
        self.activation = nn.Softplus()

    def forward(self, x):
        shape = self.activation(self.shape_layer(x)) + 1e-6
        rate = self.activation(self.rate_layer(x)) + 1e-6
        return torch.stack([shape, rate], dim=-1)

    def sample(self, head_output, num_samples=1):
        shape, rate = head_output[..., 0], head_output[..., 1]
        return dist.Gamma(shape, rate).rsample([num_samples])

class WeibullHead(GammaHead):
    """Weibull distribution for event duration modeling."""
    def sample(self, head_output, num_samples=1):
        scale, concentration = head_output[..., 0], head_output[..., 1]
        return dist.Weibull(scale, concentration).rsample([num_samples])

class PoissonHead(BaseDistribution):
    """Poisson distribution for count data forecasting."""
    def __init__(self, input_dim, output_dim, args):
        super().__init__(input_dim, output_dim, args)
        self.rate_layer = nn.Linear(input_dim, output_dim)
        self.activation = nn.Softplus()  # Ensure rate > 0

    def forward(self, x):
        rate = self.activation(self.rate_layer(x)) + 1e-6
        return rate.unsqueeze(-1)

    def sample(self, head_output, num_samples=1):
        rate = head_output[..., 0]
        return dist.Poisson(rate).sample([num_samples])

class NegativeBinomialHead(PoissonHead):
    """Negative Binomial distribution for overdispersed count data."""
    def __init__(self, input_dim, output_dim, args):
        super().__init__(input_dim, output_dim, args)
        self.dispersion_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        rate = self.activation(self.rate_layer(x)) + 1e-6
        dispersion = self.activation(self.dispersion_layer(x)) + 1e-6
        return torch.stack([rate, dispersion], dim=-1)

    def sample(self, head_output, num_samples=1):
        rate, dispersion = head_output[..., 0], head_output[..., 1]
        return dist.NegativeBinomial(total_count=dispersion, probs=rate / (rate + dispersion)).sample([num_samples])

class DirichletHead(BaseDistribution):
    """Dirichlet distribution for multivariate probabilities."""
    def __init__(self, input_dim, output_dim, args):
        super().__init__(input_dim, output_dim, args)
        self.concentration_layer = nn.Linear(input_dim, output_dim)
        self.activation = nn.Softplus()

    def forward(self, x):
        concentration = self.activation(self.concentration_layer(x)) + 1e-6
        return concentration.unsqueeze(-1)

    def sample(self, head_output, num_samples=1):
        concentration = head_output[..., 0]
        return dist.Dirichlet(concentration).rsample([num_samples])

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
        "i_quantile": ImplicitQuantileHead
    }

    def __init__(self, input_dim, output_dim, distribution_type="gaussian", quantiles=[]):
        super().__init__()
        if distribution_type not in self.DISTRIBUTIONS:
            raise ValueError(f"Unsupported distribution type: {distribution_type}")
        self.head = self.DISTRIBUTIONS[distribution_type](input_dim, output_dim, {'quantiles':quantiles})

    def forward(self, x):
        return self.head(x)

    def sample(self, head_output, num_samples=1):# TODO implement the functionality of sampling more than one sample -> consider desired shape of output
        return self.head.sample(head_output, num_samples)  



# class ProbabilisticHead(nn.Module):
#     def __init__(self, input_dim, output_dim, distribution_type="gaussian", quantiles=[]):
#         super(ProbabilisticHead, self).__init__()
#         self.distribution_type = distribution_type
#         self.quantiles = quantiles

#         if self.distribution_type in ["gaussian", "laplace"]:
#             self.num_layers = 2
#             self.activation = nn.Softplus()
#             self.layers = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(self.num_layers)])
#         elif self.distribution_type == "quantile":
#             assert quantiles is not None, "num_quantiles must be specified for quantile regression"
#             self.num_layers = len(quantiles)
#             self.layers = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(self.num_layers)])
#         elif self.distribution_type == "i_quantile":
#             self.num_layers = 1
#             self.quantile_embed_dim = 60
#             self.cos_embedding_dim = self.quantile_embed_dim
#             self.quantile_embedding = nn.Linear(self.cos_embedding_dim, self.quantile_embed_dim)
#             self.qr_head = nn.Linear(input_dim + self.quantile_embed_dim, output_dim)
#             self.activation = nn.ReLU()
#         else:
#             raise ValueError(f"Unsupported distribution type: {distribution_type}")

#     def forward(self, x):
#         batch_size = x.size(0)

#         if self.distribution_type in ["gaussian", "laplace"]:
#             mu = self.layers[0](x)
#             sigma = self.activation(self.layers[1](x))
#             return torch.stack([mu, sigma], dim=-1)
#         elif self.distribution_type == "quantile":
#             return torch.stack([layer(x) for layer in self.layers], dim=-1)
#         elif self.distribution_type == "i_quantile":
#             if self.training:
#                 u = torch.rand(batch_size, 1).to(x.device)  # # [batch_size, 1] - One random quantile level per element
#                 # IQN Cosine Embedding for Quantile Processing
#                 i = torch.arange(1, self.cos_embedding_dim + 1, device=x.device).float().unsqueeze(0)  # [1, embed_dim]
#                 cos_features = torch.cos(torch.pi * u * i)  # Cosine embedding
#                 phi_u = self.activation(self.quantile_embedding(cos_features))  # Learnable transformation
#                 if len(x.shape)<3:# this is the case when individual is used
#                     x = x.unsqueeze(dim=1) # shape: [64, 1, 60]
#                 phi_u = phi_u.unsqueeze(1).expand(-1, x.shape[1], -1)  # Shape: [64, x.shape[1], 60]
#                 # Concatenate quantile representation to input x
#                 x_augmented = torch.cat([x, phi_u], dim=-1).squeeze() # [batch_size, output_dim]
#                 predictions = self.qr_head(x_augmented)  # [batch_size, output_dim]
#                 if predictions.dim() != u.dim():
#                     u = u.unsqueeze(-1)
#                     batch_size, num_series, length = predictions.shape
#                     u = u.expand(batch_size, num_series, 1)
#                 return torch.cat([predictions, u], dim=-1).unsqueeze(-1)  # Append tau to output
#             else:
#                 #u = self.quantiles.to(x.device).repeat(batch_size, 1)  # Fixed quantile levels for inference
#                 quantile_levels = torch.tensor(self.quantiles, device=x.device).float()  # Convert to tensor
#                 predictions = []
#                 with torch.no_grad():
#                     for u in quantile_levels:
#                         u = torch.full((x.size(0), 1), u, device=x.device)  # Same tau for entire batch
#                         # IQN Cosine Embedding for Quantile Processing
#                         i = torch.arange(1, self.cos_embedding_dim + 1, device=x.device).float().unsqueeze(0)  # [1, embed_dim]
#                         cos_features = torch.cos(torch.pi * u * i)  # Cosine embedding
#                         phi_u = self.activation(self.quantile_embedding(cos_features))  # Learnable transformation
#                         if len(x.shape)<3:# this is the case when individual is used
#                             x = x.unsqueeze(dim=1) # shape: [64, 1, 60]
#                         phi_u = phi_u.unsqueeze(1).expand(-1, x.shape[1], -1)  # Shape: [64, x.shape[1], 60]
#                         # Concatenate quantile representation to input x
#                         x_augmented = torch.cat([x, phi_u], dim=-1).squeeze()
#                         pred = self.qr_head(x_augmented)  # [batch_size, num_quantiles, output_dim]
#                         #return torch.cat([pred, u.unsqueeze(-1)], dim=-1)  # Append tau to output
#                         predictions.append(pred)
#                 predictions = torch.stack(predictions, dim=1) # [batch_size, num_quantiles, (num_series), output_dim] -> num_series not always present, but for example when individual=False
#                 if predictions.dim() == 3:
#                     predictions = predictions.permute(0,2,1)  # [batch_size, output_dim, num_quantiles]
#                 else:
#                     predictions = predictions.permute(0, 2, 3, 1) # [batch_size, num_series, output_dim, num_quantiles]
#                 return predictions

#     # def predict_with_quantiles(model, x, quantile_levels):
#     #     """
#     #     Generate predictions for specific quantile levels during inference.

#     #     Args:
#     #         model (nn.Module): The model containing the probabilistic head.
#     #         x (torch.Tensor): Input tensor of shape [batch_size, input_dim].
#     #         quantile_levels (list or torch.Tensor): List of quantile levels to predict.

#     #     Returns:
#     #         torch.Tensor: Predictions of shape [batch_size, num_quantiles, output_dim].
#     #     """
#     #     self.eval()  # Ensure the model is in evaluation mode
#     #     quantile_levels = torch.tensor(quantile_levels, device=x.device).float()  # Convert to tensor

#     #     predictions = []
#     #     with torch.no_grad():
#     #         for tau in quantile_levels:
#     #             tau_tensor = torch.full((x.size(0), 1), tau, device=x.device)  # Same tau for entire batch
#     #             pred = model.head(x, tau_tensor)  # Forward pass for this quantile
#     #             predictions.append(pred)

#     #     return torch.stack(predictions, dim=1)  # [batch_size, num_quantiles, output_dim]

# class ProbabilisticHead_1(nn.Module):
#     def __init__(self, input_dim, output_dim, distribution_type="gaussian", quantiles=None):
#         super(ProbabilisticHead, self).__init__()
#         self.distribution_type = distribution_type
#         self.layers = nn.ModuleList()
#         if self.distribution_type in ["gaussian", "laplace"]:
#             self.num_layers = 2
#             self.activation = nn.Softplus()  # Activation for standard deviation or scale
#         elif self.distribution_type == "quantile":
#             assert quantiles is not None, "num_quantiles must be specified for quantile regression"
#             self.num_layers = len(quantiles)
#         elif self.distribution_type == "i_quantile":
#             self.num_layers = 1
#             input_dim = input_dim + 1
#         else:
#             raise ValueError(f"Unsupported distribution type: {distribution_type}")
#         for _ in range(self.num_layers):
#             self.layers.append(nn.Linear(input_dim, output_dim))

#     def forward(self, x):
#         if self.distribution_type in ["gaussian", "laplace"]:
#             predictions = [self.layers[0](x)]
#             predictions.append(self.activation(self.layers[1](x)))
#         elif self.distribution_type == "quantile":
#             predictions = [layer(x) for layer in self.layers]
#         elif self.distribution_type == "i_quantile":
#             u = torch.rand(x.size(0), 1).to(x.device)  # quantile level for which to produce the prediction
#             x_with_quantile = torch.cat([x, u], dim=-1)  # Concatenate quantile to input
#             print(x_with_quantile.shape)
#             predictions = self.layers[0](x_with_quantile)
#             print(predictions.shape)
#             # Append the sampled quantile value to the predictions
#             predictions_with_quantile = torch.cat([predictions, u], dim=-1)
#             print(predictions_with_quantile.shape)
#             return predictions_with_quantile

#         predictions = torch.stack(predictions, dim=-1)
#         return predictions

