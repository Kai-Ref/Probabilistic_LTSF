import torch
import torch.nn as nn
import torch.distributions as dist
# from .flow_heads import FlowHead
import math

class BaseDistribution(nn.Module):
    """Abstract base class for different probabilistic heads."""
    def __init__(self, input_dim, output_dim, prob_args={}):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward method.")

    def __get_dist__(self, prediction):
        raise NotImplementedError("Subclasses must implement __get_dist__ method.")

    def sample(self, head_output, num_samples=1, random_state=None):
        """Sample from the probabilistic head output."""
        dist = self.__get_dist__(head_output)

        if random_state is not None:
            # Save RNG state to restore later
            rng_state = torch.get_rng_state()
            torch.manual_seed(random_state)

        samples = dist.rsample((num_samples,))

        if random_state is not None:
            # Restore the original RNG state
            torch.set_rng_state(rng_state)

        return samples

class MultivariateGaussianHead(BaseDistribution):
    def __init__(self, input_dim, output_dim, prob_args={}):
        """
        Predict a low-rank factorization of the covariance matrix
        
        Args:
        input_dim: Dimensionality of input features
        output_dim: Dimensionality of the output distribution
        rank: Rank of the low-rank factorization (default is 5)
        """
        super().__init__(input_dim, output_dim, prob_args=prob_args) 
        assert 'rank' in prob_args.keys()
        self.rank = prob_args['rank']
        
        # Mean prediction layer
        self.mean_layer = nn.Linear(input_dim, output_dim)
        
        # Low-rank factorization layers
        # V will be output_dim x rank matrix
        # S will be rank-sized vector of scaling factors
        self.V_layer = nn.Linear(input_dim, output_dim * self.rank)
        self.S_layer = nn.Linear(input_dim, self.rank)

    def forward(self, x):
        # The shape of x is [batch_size, nvars, features]
        batch_size, nvars, features = x.shape
        # Reshape x for linear layers: [batch_size*nvars, features]
        x_flat = x.reshape(-1, features)
        
        # Predict mean
        mean = self.mean_layer(x_flat)  # [batch_size*nvars, output_dim]
        
        # Predict low-rank factorization components
        V = self.V_layer(x_flat)  # [batch_size*nvars, output_dim*rank]
        S = torch.nn.functional.softplus(self.S_layer(x_flat))  # [batch_size*nvars, rank]
        
        # Reshape back to include nvars dimension
        mean = mean.view(batch_size, nvars, self.output_dim)  # [batch_size, nvars, output_dim]
        V = V.view(batch_size, nvars, self.output_dim, self.rank)  # [batch_size, nvars, output_dim, rank]
        S = S.view(batch_size, nvars, self.rank)  # [batch_size, nvars, rank]
        
        # Add dimension for concatenation
        mean = mean.unsqueeze(-1)  # [batch_size, nvars, output_dim, 1]
        S = S.unsqueeze(2)  # [batch_size, nvars, 1, rank]
        
        # Repeat S to match output_dim dimension of V
        S = S.repeat(1, 1, self.output_dim, 1)  # [batch_size, nvars, output_dim, rank]
        
        # Concatenate along the last dimension
        result = torch.cat([
            mean,  # [batch_size, nvars, output_dim, 1]
            V,     # [batch_size, nvars, output_dim, rank]
            S      # [batch_size, nvars, output_dim, rank]
        ], dim=-1)  # [batch_size, nvars, output_dim, 1 + 2*rank]
        
        return result

    def __get_dist__(self, prediction):
        distributions = []
        prediction = prediction.permute(0,2,1,3)
        # Extract components with correct dimensions
        mean = prediction[..., 0]  # [batch_size, nvars, output_dim]
        V_full = prediction[..., 1:1+self.rank]  # [batch_size, nvars, output_dim, rank]
        S_full = prediction[..., 1+self.rank:]  # [batch_size, nvars, output_dim, rank]
        
        # Ensure positive values for variance scaling factors
        S_full = torch.clamp(S_full, min=1e-6)
        
        # Process each variable separately
        for i in range(prediction.shape[1]):  # Loop over nvars
            var_mean = mean[:, i, :]  # [batch_size, output_dim]
            var_V = V_full[:, i, :, :]  # [batch_size, output_dim, rank]
            var_S = S_full[:, i, 0, :]  # [batch_size, rank] - using first output_dim slice
            
            # Create covariance matrix using robust method
            Q = ensure_positive_definite_matrix(var_V, var_S, method='robust_cov')
            
            # Create distribution
            distributions.append(dist.MultivariateNormal(var_mean, Q))
            
        return distributions

    def sample(self, head_output, num_samples=1):
        """Sample from the multivariate normal distributions."""
        batch_size, num_series, output_dim, _ = head_output.shape
        samples = torch.zeros(num_samples, batch_size, output_dim, num_series, device=head_output.device)
        distributions = self.__get_dist__(head_output)
        
        for i, distribution in enumerate(distributions):
            samples[:, :, :, i] = distribution.rsample((num_samples,))
            
        return samples

    # def forward(self, x):
    #     # Predict mean
    #     mean = self.mean_layer(x)
    #     # Predict low-rank factorization components
    #     # Reshape V to be (batch_size, output_dim * var_dim, rank)
    #     V = self.V_layer(x).view(x.shape[0], self.output_dim, self.rank)

    #     # Predict scaling factors with softplus to ensure positivity
    #     # Shape will be (batch_size, rank)
    #     S = torch.nn.functional.softplus(self.S_layer(x))
    #     S = S.unsqueeze(1).repeat(1, V.shape[1], 1)
        
    #     # Combine mean, V, and S into a single tensor
    #     # This allows for potential dropout or other operations
    #     return torch.cat([
    #         mean.unsqueeze(-1),  # First: mean tensor
    #         V,     # Second: V matrix
    #         S  # Last: scaling factors
    #     ], dim=-1)

    # def __get_dist__(self, prediction):
    #     distributions = []
    #     rank = self.rank
    #     mean, V_full, S_full = prediction[..., 0], prediction[..., 1:-rank], prediction[..., -rank:]
    #     S_full = S_full[:, 0, :, :] #torch.abs(S[:, 0, :, :])
    #     S_full = torch.clamp(S_full, min=1e-6)
    #     for i in range(prediction.shape[-2]):
    #         S = S_full[:, i, :]
    #         V = V_full[:, :, i, :]
    #         Q = ensure_positive_definite_matrix(V, S, method='robust_cov')
    #         distributions.append(dist.MultivariateNormal(mean[:, :, i], Q))
    #     return distributions

    # def sample(self, head_output, num_samples=1):
    #     """Since slightly different behavior overwrite the sampling function."""
    #     batch_size, output_dim, num_series = head_output[..., 0].shape
    #     samples = torch.zeros(num_samples, batch_size, output_dim, num_series, device=head_output.device)
    #     distributions = self.__get_dist__(head_output)
    #     for i, dist in enumerate(distributions):
    #         samples[:, :, :, i] = dist.rsample((num_samples,))
    #     return samples

class LowRankMultivariateGaussianHead(BaseDistribution):
    def __init__(self, input_dim, output_dim, prob_args={}):
        """
        Predict a low-rank factorization of the covariance matrix
       
        Args:
        input_dim: Dimensionality of input features
        output_dim: Dimensionality of the output distribution
        rank: Rank of the low-rank factorization (default is 5)
        """
        super().__init__(input_dim, output_dim, prob_args=prob_args)
        assert 'rank' in prob_args.keys()
        self.rank = prob_args['rank']

        self.output_dim = output_dim
        # Mean prediction layer
        self.mean_layer = nn.Linear(input_dim, output_dim)
       
        # Low-rank factorization layers
        # V will be output_dim x rank matrix
        # S will be rank-sized vector of scaling factors
        self.V_layer = nn.Linear(input_dim, output_dim * self.rank)
        self.S_layer = nn.Linear(input_dim, output_dim) #self.rank)
   
    def forward(self, x):
        if len(x.shape) <3:
            batch_size, features = x.shape
            nvars = 1
        else:
            # The shape of x is probably [batch_size, nvars, features]
            batch_size, nvars, features = x.shape
        # Reshape x for linear layers: [batch_size*nvars, features]
        x_flat = x.reshape(-1, features)
        # Predict mean
        mean = self.mean_layer(x_flat)  # [batch_size*nvars, output_dim]
        
        # Predict low-rank factorization components
        V = self.V_layer(x_flat)  # [batch_size*nvars, output_dim*rank]
        S = torch.nn.functional.softplus(self.S_layer(x_flat))  # [batch_size*nvars, output_dim]
        
        # Reshape back to include nvars dimension
        mean = mean.view(batch_size, nvars, -1)  # [batch_size, nvars, output_dim]
        V = V.view(batch_size, nvars, -1, self.rank)  # [batch_size, nvars, output_dim, rank]
        S = S.view(batch_size, nvars, self.output_dim)  # [batch_size, nvars, output_dim]
        
        # Add dimension for concatenation
        mean = mean.unsqueeze(-1)  # [batch_size, nvars, output_dim, 1]
        S = S.unsqueeze(-1)  # [batch_size, nvars, output_dim, 1]
        # Repeat S to match output_dim dimension of V
        # S = S.repeat(1, 1, V.shape[2], 1)  # [batch_size, nvars, output_dim, rank]

        # Now all tensors have shape [batch_size, nvars, output_dim, *]
        # Concatenate along the last dimension
        result = torch.cat([
            mean,  # [batch_size, nvars, output_dim, 1]
            V,     # [batch_size, nvars, output_dim, rank]
            S      # [batch_size, nvars, output_dim, 1]
        ], dim=-1)  # [batch_size, nvars, output_dim, 1 + rank + 1]
        return result

    def __get_dist__(self, prediction):
        distributions = []
        rank = self.rank
        prediction = prediction.permute(0,2,1,3)
        mean = prediction[..., 0]  # [batch_size, nvars, output_dim]
        V_full = prediction[..., 1:1+rank]  # [batch_size, nvars, output_dim, rank]
        S_full = prediction[..., 1+rank:]  # [batch_size, nvars, output_dim, 1]
        
        # Ensure positive values for variance scaling factors
        S_full = torch.clamp(S_full, min=1e-4)
        for i in range(prediction.shape[1]):  # Loop over nvars
            # Extract data for this variable
            var_mean = mean[:, i, :]  # [batch_size, output_dim]
            var_V = V_full[:, i, :, :]  # [batch_size, output_dim, rank]
            var_S = S_full[:, i, :, :].squeeze(-1)  # [batch_size, output_dim] - using first output_dim slice for S
            # var_S = torch.diag_embed(var_S)
            # Create distribution
            distributions.append(dist.LowRankMultivariateNormal(
                var_mean, 
                cov_factor=var_V, 
                cov_diag=var_S
            ))
            
        return distributions
        
    def sample(self, head_output, num_samples=1, random_state=None):
        """Since slightly different behavior overwrite the sampling function."""
        batch_size, output_dim, num_series, _ = head_output.shape
        samples = torch.zeros(num_samples, batch_size, output_dim, num_series, device=head_output.device)
        distributions = self.__get_dist__(head_output)
        if random_state is not None:
            # Save RNG state to restore later
            rng_state = torch.get_rng_state()
            torch.manual_seed(random_state)

        for i, distribution in enumerate(distributions):
            samples[:, :, :, i] = distribution.rsample((num_samples,))
        
        if random_state is not None:
            # Restore the original RNG state
            torch.set_rng_state(rng_state)

        return samples

class GaussianHead(BaseDistribution):
    """Gaussian distribution output: mean and variance."""
    def __init__(self, input_dim, output_dim, prob_args={}):
        super().__init__(input_dim, output_dim, prob_args=prob_args)
        self.mean_layer = nn.Linear(input_dim, output_dim)
        self.std_layer = nn.Linear(input_dim, output_dim)
        self.activation = nn.Softplus()

    def forward(self, x):
        mu = self.mean_layer(x)
        sigma = self.activation(self.std_layer(x))
        return torch.stack([mu, sigma], dim=-1)

    def __get_dist__(self, prediction):
        mu, sigma = prediction[..., 0], prediction[..., 1]
        sigma = torch.clamp(sigma, min=1e-6)  # Ensure std > 0, TODO maybe only do this during training?
        return dist.Normal(mu, sigma)

class LaplaceHead(GaussianHead):
    """
    Laplace distribution has heavier than Gaussian. Consists of two parameters loc and scale, where scale is required to be >0.
    Uses the same structure as the Gaussian Head.
    Laplace distribution output: location and scale."""
    def __get_dist__(self, prediction):
        loc, scale = prediction[..., 0], prediction[..., 1]
        scale = torch.clamp(scale, min=1e-6)
        return dist.Laplace(loc, scale)  # Ensure std > 0, TODO maybe only do this during training?

class QuantileHead(BaseDistribution):
    """Quantile regression head."""
    def __init__(self, input_dim, output_dim, prob_args={}):
        super().__init__(input_dim, output_dim, prob_args=prob_args)
        assert prob_args['quantiles'] is not None, "quantiles must be specified for quantile regression"
        self.quantiles = prob_args['quantiles']
        self.layers = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(len(self.quantiles))])

    def forward(self, x):
        return torch.stack([layer(x) for layer in self.layers], dim=-1)

    def sample(self, head_output, num_samples=1, random_state=None):
        """Uses the median quantile as the point estimate."""
        median_idx = self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2
        return head_output[..., median_idx]

class ImplicitQuantileHead(BaseDistribution):
    """Implicit Quantile Network (IQN) for quantile regression."""
    def __init__(self, input_dim, output_dim, prob_args={}):
        super().__init__(input_dim, output_dim, prob_args=prob_args)
        self.num_layers = prob_args['num_layers'] #1
        self.decoding = prob_args['decoding'] # either hadamard or concat
        self.cos_embedding_dim = prob_args['cos_embedding_dim']
        if self.decoding == "hadamard":
            self.quantile_embed_dim = prob_args['fixed_qe']
            self.qr_head = nn.Linear(self.quantile_embed_dim, output_dim)
        else:
            self.quantile_embed_dim = prob_args['quantile_embed_dim'] 
            self.qr_head = nn.Linear(input_dim + self.quantile_embed_dim, output_dim)
        self.quantile_embedding = nn.Linear(self.cos_embedding_dim, self.quantile_embed_dim)
        self.activation = nn.ReLU()
        self.quantiles = prob_args['quantiles'] # only needed in eval mode ->e.g. val/test quantile loss
        self.u = None

    def _make_pred(self, u, x):
        # IQN Cosine Embedding for Quantile Processing
        i = torch.arange(1, self.cos_embedding_dim + 1, device=x.device).float().unsqueeze(0)  # [1, embed_dim]
        cos_features = torch.cos(torch.pi * u * i)  # Cosine embedding
        phi_u = self.activation(self.quantile_embedding(cos_features))  # Learnable transformation
        if len(x.shape)<3:# this is the case when individual is used
            x = x.unsqueeze(dim=1) # shape: [64, 1, 60]
        phi_u = phi_u.unsqueeze(1).expand(-1, x.shape[1], -1)  # Shape: [64, x.shape[1], embedding dim]

        if self.decoding == "concat": # Concatenate quantile representation to input x
            x = torch.cat([x, phi_u], dim=-1) # [batch_size, output_dim]
        elif self.decoding == "hadamard": # Element-wise interaction
            x = x * (1 + phi_u)  # [batch_size, output_dim]

        predictions = self.qr_head(x).squeeze(1)  # [batch_size, output_dim]
        return predictions

    def forward(self, x, resample_u=False):
        batch_size = x.size(0)
        if self.training:
            if (self.output_dim == 1) and resample_u:
                if (self.u is None):
                    self.u = torch.rand(batch_size, 1).to(x.device)  # [batch_size, 1] - One random quantile level per element
                u = self.u[:batch_size]
            else:                    
                u = torch.rand(batch_size, 1).to(x.device)  # [batch_size, 1] - One random quantile level per element
            predictions = self._make_pred(u, x)
            if predictions.dim() != u.dim():
                u = u.unsqueeze(-1)
                batch_size, num_series, length = predictions.shape
                u = u.expand(batch_size, num_series, 1)
            # print(u.shape)
            return torch.cat([predictions, u], dim=-1).unsqueeze(-1)  # Append tau to output
        else:
            #u = self.quantiles.to(x.device).repeat(batch_size, 1)  # Fixed quantile levels for inference
            quantile_levels = torch.tensor(self.quantiles, device=x.device).float()  # Convert to tensor
            predictions = []
            with torch.no_grad():
                for tau in quantile_levels:
                    u = torch.full((x.size(0), 1), tau, device=x.device)  # Same tau for entire batch
                    pred = self._make_pred(u, x)
                    predictions.append(pred)
            predictions = torch.stack(predictions, dim=1) # [batch_size, num_quantiles, (num_series), output_dim] -> num_series not always present, for example when individual=False
            if predictions.dim() == 3:
                predictions = predictions.permute(0,2,1)  # [batch_size, output_dim, num_quantiles]
            else:
                predictions = predictions.permute(0, 2, 3, 1) # [batch_size, num_series, output_dim, num_quantiles]
            return predictions
    
    def sample(self, head_output, num_samples=1, random_state=None):
        """Select the 0.5 quantile (median) if available, else fallback to the midpoint."""
        if self.training:
            return head_output[..., 0]  # Use the direct prediction during training
        else:
            median_idx =  self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2
            return head_output[..., median_idx]

class StudentTHead(GaussianHead):
    """Student's t-distribution for heavy-tailed data."""
    def __init__(self, input_dim, output_dim, prob_args={}):
        super().__init__(input_dim, output_dim, prob_args=prob_args)
        self.df_layer = nn.Linear(input_dim, output_dim)
        self.df_activation = nn.Softplus()  # Ensure degrees of freedom > 0

    def forward(self, x):
        mu = self.mean_layer(x)
        sigma = self.activation(self.std_layer(x))
        df = self.df_activation(self.df_layer(x)) + 1  # Ensure df > 1
        return torch.stack([mu, sigma, df], dim=-1)

    def __get_dist__(self, prediction):#TODO sampling throws an error
        mu, sigma, df = prediction[..., 0], prediction[..., 1], prediction[..., 2]
        sigma = torch.clamp(sigma, min=1e-6)
        df = torch.clamp(df, min=1.1)  # Ensure degrees of freedom > 1
        return dist.StudentT(df, mu, sigma)

class LogNormalHead(GaussianHead):
    """Log-normal distribution for positive-valued time series."""

    def __get_dist__(self, prediction):
        mu, sigma = prediction[..., 0], prediction[..., 1]
        sigma = torch.clamp(sigma, min=1e-6)
        return dist.LogNormal(mu, sigma)

class BetaHead(BaseDistribution):
    """Beta distribution for forecasting probabilities (0,1)."""
    def __init__(self, input_dim, output_dim, prob_args={}):
        super().__init__(input_dim, output_dim, prob_args=prob_args)
        self.alpha_layer = nn.Linear(input_dim, output_dim)
        self.beta_layer = nn.Linear(input_dim, output_dim)
        self.activation = nn.Softplus()  # Ensure alpha, beta > 0

    def forward(self, x):
        alpha = self.activation(self.alpha_layer(x)) + 1e-6
        beta = self.activation(self.beta_layer(x)) + 1e-6
        return torch.stack([alpha, beta], dim=-1)

    def __get_dist__(self, prediction):
        alpha, beta = prediction[..., 0], prediction[..., 1]
        alpha = torch.clamp(alpha, min=1e-6)
        beta = torch.clamp(beta, min=1e-6)
        return dist.Beta(alpha, beta)

class GammaHead(BaseDistribution):
    """Gamma distribution for skewed positive-valued data."""
    def __init__(self, input_dim, output_dim, prob_args={}):
        super().__init__(input_dim, output_dim, prob_args=prob_args)
        self.shape_layer = nn.Linear(input_dim, output_dim)
        self.rate_layer = nn.Linear(input_dim, output_dim)
        self.activation = nn.Softplus()

    def forward(self, x):
        shape = self.activation(self.shape_layer(x)) + 1e-6
        rate = self.activation(self.rate_layer(x)) + 1e-6
        return torch.stack([shape, rate], dim=-1)

    def __get_dist__(self, prediction):
        shape, rate = prediction[..., 0], prediction[..., 1]
        shape = torch.clamp(shape, min=1e-6)
        rate = torch.clamp(rate, min=1e-6)
        return dist.Gamma(shape, rate)

class WeibullHead(GammaHead):
    """Weibull distribution for event duration modeling."""
    def sample(self, head_output, num_samples=1):
        scale, concentration = head_output[..., 0], head_output[..., 1]
        return dist.Weibull(scale, concentration).rsample([num_samples])

    def __get_dist__(self, prediction):
        scale, concentration = prediction[..., 0], prediction[..., 1]
        scale = torch.clamp(scale, min=1e-6)
        concentration = torch.clamp(concentration, min=1e-6)
        return dist.Weibull(scale, concentration)

class PoissonHead(BaseDistribution):
    """Poisson distribution for count data forecasting."""
    def __init__(self, input_dim, output_dim, prob_args={}):
        super().__init__(input_dim, output_dim, prob_args=prob_args)
        self.rate_layer = nn.Linear(input_dim, output_dim)
        self.activation = nn.Softplus()  # Ensure rate > 0

    def forward(self, x):
        rate = self.activation(self.rate_layer(x)) + 1e-6
        return rate.unsqueeze(-1)

    def __get_dist__(self, prediction):
        rate = prediction[..., 0]
        rate = torch.clamp(rate, min=1e-6)
        return dist.Poisson(rate)

class NegativeBinomialHead(PoissonHead):
    """Negative Binomial distribution for overdispersed count data."""
    def __init__(self, input_dim, output_dim, prob_args={}):
        super().__init__(input_dim, output_dim, prob_args=prob_args)
        self.dispersion_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        rate = self.activation(self.rate_layer(x)) + 1e-6
        dispersion = self.activation(self.dispersion_layer(x)) + 1e-6
        return torch.stack([rate, dispersion], dim=-1)

    def __get_dist__(self, prediction):
        rate, dispersion = prediction[..., 0], prediction[..., 1]
        rate = torch.clamp(rate, min=1e-6)
        dispersion = torch.clamp(dispersion, min=1e-6)
        probs = rate / (rate + dispersion)  # Convert to success probability
        return dist.NegativeBinomial(total_count=dispersion, probs=probs)

class DirichletHead(BaseDistribution):
    """Dirichlet distribution for multivariate probabilities."""
    def __init__(self, input_dim, output_dim, prob_args={}):
        super().__init__(input_dim, output_dim, prob_args=prob_args)
        self.concentration_layer = nn.Linear(input_dim, output_dim)
        self.activation = nn.Softplus()

    def forward(self, x):
        concentration = self.activation(self.concentration_layer(x)) + 1e-6
        return concentration.unsqueeze(-1)

    def __get_dist__(self, prediction):
        concentration = prediction[..., 0]
        concentration = torch.clamp(concentration, min=1e-6)
        return dist.Dirichlet(concentration)

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
            # "quantile": QuantileHead, 
            # "i_quantile": ImplicitQuantileHead,
            "m_gaussian": MultivariateGaussianHead,
            "m_lr_gaussian": LowRankMultivariateGaussianHead,
            # "flow": FlowHead,
            }

class GaussianCopulaHead(BaseDistribution):
    """
    Gaussian copula head with flexible marginals.
    - Models dependencies using a Gaussian copula (correlation matrix).
    - Marginals can be any head from the DISTRIBUTIONS dict.
    Args (in args dict):
        marginal_type, marginal_args
    """
    def __init__(self, input_dim, output_dim, prob_args={}):
        super().__init__(input_dim, output_dim, prob_args=prob_args)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.marginal_type = args.get('marginal_type', 'gaussian')
        self.marginal_args = args.get('marginal_args', {})
        # Marginal head (shared for all time steps/variables)
        marginal_class = DISTRIBUTIONS[self.marginal_type]
        self.marginal_head = marginal_class(input_dim, 96, self.marginal_args)
        # For the copula: output_dim * (output_dim-1) // 2 parameters for the lower triangle (excluding diagonal)
        self.corr_param_layer = nn.Linear(input_dim, output_dim * (output_dim - 1) // 2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Args:
            x: [batch, output_dim, input_dim] (embedding for each time step or variable)
        Returns:
            prediction: torch.Tensor [batch, output_dim, param_dim + corr_dim] (marginal params and copula params stacked)
        """
        batch, output_dim, input_dim = x.shape
        # Marginals
        marginal_params = self.marginal_head(x)
        # Copula correlation params
        corr_params = self.tanh(self.corr_param_layer(x)).unsqueeze(2).repeat(1, 1, self.output_dim, 1)
        prediction = torch.cat([marginal_params, corr_params], dim=-1)
        return prediction

    class GaussianCopulaDistribution:
        def __init__(self, marginal_head, prediction, output_dim):
            self.marginal_head = marginal_head
            self.prediction = prediction
            self.output_dim = output_dim
            # Unpack
            param_dim = marginal_head.output_dim if hasattr(marginal_head, 'output_dim') else 2
            corr_dim = output_dim * (output_dim - 1) // 2
            self.marginal_params = prediction[..., :param_dim]
            self.corr_params = prediction[..., param_dim: param_dim + corr_dim]
            self.corr = self._build_correlation_matrix(self.corr_params)

        def _build_correlation_matrix(self, corr_params):
            corr = torch.eye(self.output_dim, device=corr_params.device)
            tril_indices = torch.tril_indices(row=self.output_dim, col=self.output_dim, offset=-1)
            corr[tril_indices[0], tril_indices[1]] = corr_params
            corr = corr + corr.T - torch.diag(torch.diag(corr))
            return corr

            def __get_dist__(self, prediction):
                distributions = []
                rank = self.rank
                mean, V_full, S_full = prediction[..., 0], prediction[..., 1:-rank], prediction[..., -rank:]
                S_full = S_full[:, 0, :, :] #torch.abs(S[:, 0, :, :])
                S_full = torch.clamp(S_full, min=1e-6)
                for i in range(prediction.shape[-2]):
                    S = S_full[:, i, :]
                    V = V_full[:, :, i, :]
                    # Q = ensure_positive_definite_matrix(V, S, method='robust_cov')
                    distributions.append(dist.LowRankMultivariateNormal(mean[:, :, i], cov_diag=S, cov_factor=V))
                return distributions

        def sample(self, num_samples=1):
            # Sample from Gaussian copula
            mvn = dist.MultivariateNormal(torch.zeros(self.output_dim, device=self.corr.device), covariance_matrix=self.corr)
            z = mvn.sample((num_samples,))  # [num_samples, output_dim]
            u = dist.Normal(0, 1).cdf(z)
            # Transform to marginals
            samples = []
            for v in range(self.output_dim):
                dist_v = self.marginal_head.__get_dist__(self.marginal_params[:, v, ...])
                samples.append(dist_v.icdf(u[..., v]))
            samples = torch.stack(samples, dim=-1)
            return samples

        def rsample(self, num_samples=1):
            return self.sample(num_samples)

        def log_prob(self, target):
            # Transform to uniforms
            u = []
            log_marginals = 0
            for v in range(self.output_dim):
                dist_v = self.marginal_head.__get_dist__(self.marginal_params[:, v, ...])
                u_v = dist_v.cdf(target[:, v])
                u.append(u_v)
                log_marginals = log_marginals + dist_v.log_prob(target[:, v])
            u = torch.stack(u, dim=1)
            # Inverse normal CDF
            z = dist.Normal(0, 1).icdf(u)
            # Copula log density
            mvn = dist.MultivariateNormal(torch.zeros(self.output_dim, device=self.corr.device), covariance_matrix=self.corr)
            log_copula = mvn.log_prob(z) - z.pow(2).sum(dim=1) * 0.5 + 0.5 * self.output_dim * math.log(2 * math.pi)
            return log_marginals + log_copula

    def __get_dist__(self, prediction):
        return self.GaussianCopulaDistribution(self.marginal_head, prediction, self.output_dim)

    def sample(self, head_output, num_samples=1):
        dist = self.__get_dist__(head_output)
        return dist.sample(num_samples=num_samples)

class StudentTCopulaHead(BaseDistribution):
    """
    Student-t copula head with flexible marginals.
    - Models dependencies using a Student-t copula (correlation matrix + df).
    - Marginals can be any head from the DISTRIBUTIONS dict.
    Args (in args dict):
        marginal_type, marginal_args, df
    """
    def __init__(self, input_dim, output_dim, prob_args={}):
        super().__init__(input_dim, output_dim, prob_args=prob_args)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.marginal_type = args.get('marginal_type', 'gaussian')
        self.marginal_args = args.get('marginal_args', {})
        self.df = args.get('df', 4.0)
        marginal_class = DISTRIBUTIONS[self.marginal_type]
        self.marginal_head = marginal_class(input_dim, 1, self.marginal_args)
        self.corr_param_layer = nn.Linear(input_dim, output_dim * (output_dim - 1) // 2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        batch, output_dim, input_dim = x.shape
        marginal_params = []
        for d in range(output_dim):
            marginal_params.append(self.marginal_head(x[:, d, :].unsqueeze(1)))
        if isinstance(marginal_params[0], torch.Tensor):
            marginal_params = torch.cat(marginal_params, dim=1)
        else:
            marginal_params = torch.cat(marginal_params, dim=-1)
        corr_params = self.tanh(self.corr_param_layer(x.mean(dim=1)))
        df_tensor = torch.full((batch, 1), self.df, device=x.device)
        prediction = torch.cat([marginal_params, corr_params.unsqueeze(1).expand(-1, output_dim, -1), df_tensor.unsqueeze(1).expand(-1, output_dim, -1)], dim=-1)
        return prediction

    class StudentTCopulaDistribution:
        def __init__(self, marginal_head, prediction, output_dim):
            self.marginal_head = marginal_head
            self.prediction = prediction
            self.output_dim = output_dim
            param_dim = marginal_head.output_dim if hasattr(marginal_head, 'output_dim') else 2
            corr_dim = output_dim * (output_dim - 1) // 2
            self.marginal_params = prediction[..., :param_dim]
            self.corr_params = prediction[0, 0, param_dim: param_dim + corr_dim]
            self.df = prediction[0, 0, -1].item()
            self.corr = self._build_correlation_matrix(self.corr_params)

        def _build_correlation_matrix(self, corr_params):
            corr = torch.eye(self.output_dim, device=corr_params.device)
            tril_indices = torch.tril_indices(row=self.output_dim, col=self.output_dim, offset=-1)
            corr[tril_indices[0], tril_indices[1]] = corr_params
            corr = corr + corr.T - torch.diag(torch.diag(corr))
            return corr

        def sample(self, num_samples=1):
            # Sample from Student-t copula
            # Use the Gaussian copula logic, but scale by sqrt(df / chi2)
            g = torch.randn(num_samples, self.output_dim, device=self.corr.device)
            chi2 = torch.distributions.Chi2(self.df).sample((num_samples,)).to(self.corr.device)
            z = g / torch.sqrt(chi2.unsqueeze(-1) / self.df)
            L = torch.linalg.cholesky(self.corr)
            z = z @ L.T
            u = dist.StudentT(self.df).cdf(z)
            samples = []
            for v in range(self.output_dim):
                dist_v = self.marginal_head.__get_dist__(self.marginal_params[:, v, ...])
                samples.append(dist_v.icdf(u[..., v]))
            samples = torch.stack(samples, dim=-1)
            return samples

        def rsample(self, num_samples=1):
            return self.sample(num_samples)

        def log_prob(self, target):
            # Transform to uniforms
            u = []
            log_marginals = 0
            for v in range(self.output_dim):
                dist_v = self.marginal_head.__get_dist__(self.marginal_params[:, v, ...])
                u_v = dist_v.cdf(target[:, v])
                u.append(u_v)
                log_marginals = log_marginals + dist_v.log_prob(target[:, v])
            u = torch.stack(u, dim=1)
            # Inverse t CDF
            z = dist.StudentT(self.df).icdf(u)
            # Copula log density (approximate, as torch does not have multivariate StudentT)
            # Use the formula for the density of the multivariate t copula
            # log_copula = ... (implement if needed)
            # For now, return only marginal log-prob
            return log_marginals

    def __get_dist__(self, prediction):
        return self.StudentTCopulaDistribution(self.marginal_head, prediction, self.output_dim)

    def sample(self, head_output, num_samples=1):
        dist = self.__get_dist__(head_output)
        return dist.sample(num_samples=num_samples)



class CopulaHead(BaseDistribution):
    """
    Attention-based copula head with flexible marginals.
    - Supports any marginal head from the DISTRIBUTIONS dict (e.g., Gaussian, Laplace, Beta, etc.).
    - Uses multi-head attention to model dependencies between variables (copula).
    - Returns a custom distribution object for NLL and sampling.
    Args (in args dict):
        marginal_type, marginal_args, attention_heads, attention_layers, attention_dim, mlp_layers, mlp_dim, resolution, dropout, fixed_permutation
    """
    def __init__(self, input_dim, output_dim, prob_args={}):
        super().__init__(input_dim, output_dim, prob_args=prob_args)
        self.output_dim = output_dim # prediction length
        self.input_dim = input_dim
        self.attention_heads = args.get('attention_heads', 2)
        self.attention_layers = args.get('attention_layers', 2)
        self.attention_dim = args.get('attention_dim', 16)
        self.mlp_layers = args.get('mlp_layers', 2)
        self.mlp_dim = args.get('mlp_dim', 32)
        self.resolution = args.get('resolution', 10)
        self.dropout = args.get('dropout', 0.1)
        self.fixed_permutation = args.get('fixed_permutation', False)
        self.marginal_type = args.get('marginal_type', 'gaussian')
        self.marginal_args = args.get('marginal_args', {})

        # Marginal head (shared for all variables, but can be extended to per-variable)
        marginal_class = DISTRIBUTIONS[self.marginal_type]
        self.marginal_head = marginal_class(input_dim, output_dim, self.marginal_args)

        # Attention copula as before
        self.dimension_shifting_layer = nn.Linear(self.input_dim, self.attention_heads * self.attention_dim)
        def _easy_mlp(input_dim, hidden_dim, output_dim, num_layers, activation):
            elayers = [nn.Linear(input_dim, hidden_dim), activation()]
            for _ in range(1, num_layers):
                elayers += [nn.Linear(hidden_dim, hidden_dim), activation()]
            elayers += [nn.Linear(hidden_dim, output_dim)]
            return nn.Sequential(*elayers)
        self.key_creators = nn.ModuleList([
            nn.ModuleList([
                _easy_mlp(self.input_dim + 1, self.mlp_dim, self.attention_dim, self.mlp_layers, nn.ReLU)
                for _ in range(self.attention_heads)
            ]) for _ in range(self.attention_layers)
        ])
        self.value_creators = nn.ModuleList([
            nn.ModuleList([
                _easy_mlp(self.input_dim + 1, self.mlp_dim, self.attention_dim, self.mlp_layers, nn.ReLU)
                for _ in range(self.attention_heads)
            ]) for _ in range(self.attention_layers)
        ])
        self.attention_dropouts = nn.ModuleList([nn.Dropout(self.dropout) for _ in range(self.attention_layers)])
        self.attention_layer_norms = nn.ModuleList([
            nn.LayerNorm(self.attention_heads * self.attention_dim) for _ in range(self.attention_layers)
        ])
        self.feed_forwards = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.attention_heads * self.attention_dim, self.attention_heads * self.attention_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.attention_heads * self.attention_dim, self.attention_heads * self.attention_dim),
                nn.Dropout(self.dropout),
            ) for _ in range(self.attention_layers)
        ])
        self.feed_forward_layer_norms = nn.ModuleList([
            nn.LayerNorm(self.attention_heads * self.attention_dim) for _ in range(self.attention_layers)
        ])
        self.dist_extractors = _easy_mlp(
            self.attention_heads * self.attention_dim, self.mlp_dim, self.resolution, self.mlp_layers, nn.ReLU
        )
        

    def forward(self, x):
        """
        Args:
            x: [batch, output_dim, input_dim] (embedding for each variable)
        Returns:
            prediction: torch.Tensor [batch, output_dim, param_dim] (marginal params and copula logits stacked)
        """
        batch, num_series, input_dim = x.shape
        marginal_params = self.marginal_head(x) # [batch, num_series, output_dim, param_dim]
        
        u = torch.zeros(batch, num_series, self.output_dim, device=x.device)
        key_value_input = torch.cat([x, u], dim=2)
        att_value = self.dimension_shifting_layer(x)
        for layer in range(self.attention_layers):
            att_value_heads = att_value.view(batch, output_dim, self.attention_heads, self.attention_dim)
            keys = torch.stack([
                self.key_creators[layer][h](key_value_input) for h in range(self.attention_heads)
            ], dim=1)
            values = torch.stack([
                self.value_creators[layer][h](key_value_input) for h in range(self.attention_heads)
            ], dim=1)
            product = torch.einsum('bohi,bhwi->bhwo', att_value_heads, keys)
            product = self.attention_dim ** (-0.5) * product
            weights = nn.functional.softmax(product, dim=-1)
            att = torch.einsum('bhwo,bhwi->bohi', weights, values)
            att_merged_heads = att.reshape(batch, output_dim, self.attention_heads * self.attention_dim)
            att_merged_heads = self.attention_dropouts[layer](att_merged_heads)
            att_value = att_value + att_merged_heads
            att_value = self.attention_layer_norms[layer](att_value)
            att_feed_forward = self.feed_forwards[layer](att_value)
            att_value = att_value + att_feed_forward
            att_value = self.feed_forward_layer_norms[layer](att_value)
        logits = self.dist_extractors(att_value)  # [batch, output_dim, resolution]
        # Stack all outputs into a single tensor for compatibility
        if isinstance(marginal_params, tuple):
            # For tuple marginals, stack along last dim
            marginal_params = torch.cat(marginal_params.squeeze(-1), dim=-1)  # [batch, output_dim, total_param_dim]
        print(marginal_params.shape)
        print(logits.shape)
        prediction = torch.cat([marginal_params, logits], dim=-1)  # [batch, output_dim, total_param_dim + resolution]
        return prediction

    class CopulaDistribution:
        def __init__(self, marginal_head, prediction, output_dim, resolution):
            """
            Args:
                marginal_head: the marginal head instance
                prediction: [batch, output_dim, total_param_dim + resolution] (stacked)
                output_dim: number of variables
                resolution: number of bins for copula
            """
            self.marginal_head = marginal_head
            self.prediction = prediction
            self.output_dim = output_dim
            self.resolution = resolution
            # Unpack marginal params and logits
            param_dim = prediction.shape[-1] - resolution
            self.marginal_params = prediction[..., :param_dim]
            self.logits = prediction[..., param_dim:]

        def sample(self, num_samples=1):
            # Sample from copula (categorical over bins, then uniform in bin)
            batch, output_dim, resolution = self.logits.shape
            device = self.logits.device
            samples_u = torch.zeros(batch, num_samples, output_dim, device=device)
            for v in range(output_dim):
                probs = torch.softmax(self.logits[:, v, :], dim=-1)
                cat = torch.distributions.Categorical(probs)
                idx = cat.sample((num_samples,)).transpose(0, 1)
                bin_width = 1.0 / resolution
                u = torch.rand(batch, num_samples, device=device) * bin_width
                samples_u[:, :, v] = idx * bin_width + u
            # Transform uniforms to marginals
            samples = []
            for v in range(self.output_dim):
                # Unpack params for v
                param_v = self.marginal_params[:, v, ...]
                dist_v = self.marginal_head.__get_dist__(param_v)
                samples.append(dist_v.icdf(samples_u[:, :, v]))
            samples = torch.stack(samples, dim=2)  # [batch, num_samples, output_dim]
            return samples

        def rsample(self, num_samples=1):
            """
            Alias for sample, for compatibility with torch.distributions API.
            """
            return self.sample(num_samples=num_samples)

        def log_prob(self, target):
            # target: [batch, output_dim]
            u = []
            log_marginals = 0
            for v in range(self.output_dim):
                param_v = self.marginal_params[:, v, ...]
                dist_v = self.marginal_head.__get_dist__(param_v)
                u_v = dist_v.cdf(target[:, v])
                u.append(u_v)
                log_marginals = log_marginals + dist_v.log_prob(target[:, v])
            u = torch.stack(u, dim=1)  # [batch, output_dim]
            idx = torch.clamp((u * self.resolution).long(), 0, self.resolution - 1)
            log_copula = 0
            for v in range(self.output_dim):
                logit = self.logits[:, v, :]
                log_prob = torch.log_softmax(logit, dim=-1)
                log_copula = log_copula + log_prob[torch.arange(logit.shape[0]), idx[:, v]]
            return log_marginals + log_copula

    def __get_dist__(self, prediction):
        """
        Returns a custom CopulaDistribution object with .log_prob(), .sample(), and .rsample().
        """
        return self.CopulaDistribution(self.marginal_head, prediction, self.output_dim, self.resolution)

    def sample(self, head_output, num_samples=1):
        """
        Samples from the copula-marginal distribution.
        Args:
            head_output: output of forward (stacked tensor)
            num_samples: int
        Returns:
            samples: [batch, num_samples, output_dim]
        """
        dist = self.__get_dist__(head_output)
        return dist.sample(num_samples=num_samples)

def check_positive_definite(verbose=False):
    """
    Decorator to check if the output matrix is positive definite.
    
    Args:
        verbose (bool): If True, print detailed information about matrix definiteness
    
    Returns:
        Decorator function that checks matrix definiteness
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Call the original function
            matrix = func(*args, **kwargs)
            
            try:
                # Compute eigenvalues
                eigenvalues = torch.linalg.eigvals(matrix).real
                
                # Determine definiteness
                if torch.all(eigenvalues > 0):
                    status = "Positive Definite"
                elif torch.all(eigenvalues >= 0):
                    status = "Positive Semi-Definite"
                else:
                    status = "Not Positive Definite"
                
                # Verbose output if requested
                if verbose:
                    print(f"Matrix Definiteness: {status}")
                    print(f"Eigenvalues: {eigenvalues}")
                    print(f"Min Eigenvalue: {torch.min(eigenvalues).item()}")
                    print(f"Max Eigenvalue: {torch.max(eigenvalues).item()}")
                
                return matrix
            
            except Exception as e:
                if verbose:
                    print(f"Error in definiteness check: {e}")
                raise
        
        return wrapper
    return decorator

#@check_positive_definite(verbose=True)
def ensure_positive_definite_matrix(V, S, method='spectral'):
    """
    Ensure the matrix is positive definite using multiple methods.
    
    Args:
    V (torch.Tensor): Input matrix of shape [output_dim, rank]
    S (torch.Tensor): Input tensor of shape [rank]
    method (str): Method to enforce positive definiteness
    
    Returns:
    torch.Tensor: Guaranteed positive definite matrix
    """
    # Q.fill_diagonal_(torch.abs(torch.sqrt(S[b, i, :])) + 1e-6)
    # Q = V #torch.tril(V)
    # matrix = Q * torch.sqrt(S).unsqueeze(0)
    # Q = (Q + Q.T) / 2
    # Ensure the matrix is symmetric
    # matrix = 0.5 * (matrix + matrix.transpose(-2, -1))
    
    if method == 'spectral':
        matrix = ensure_positive_definite_matrix(V, S, method='robust_cov')
        # Spectral method: Eigenvalue correction
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        
        # Clip negative eigenvalues to a small positive value
        min_eigenvalue = 1e-6
        eigenvalues = torch.clamp(eigenvalues, min=min_eigenvalue)
        
        # Reconstruct the matrix
        return eigenvectors @ torch.diag_embed(eigenvalues) @ eigenvectors.transpose(-2, -1)
    
    elif method == 'cholesky':
        # Cholesky decomposition method
        try:
            # Attempt Cholesky decomposition
            L = torch.linalg.cholesky(matrix)
            return matrix
        except RuntimeError:
            # If Cholesky fails, fall back to spectral method
            return ensure_positive_definite_matrix(matrix, method='spectral')

    elif method == 'frobenius':
        # Try initial decomposition
        U, S, Vh = torch.linalg.svd(matrix)
        
        # Keep only non-negative singular values
        S = torch.clamp(S, min=0)
        
        # Reconstruct matrix
        return U @ torch.diag_embed(S) @ Vh
    
    elif method == 'nearest':
        print('Resorting to nearest')
        # Nearest positive definite matrix
        # Higham's (2002) algorithm for nearest positive definite matrix
        def is_positive_definite(A):
            try:
                torch.linalg.cholesky(A)
                return True
            except RuntimeError:
                return False
        
        k = 0
        while not is_positive_definite(matrix):
            # Add scaled identity matrix
            k += 1
            matrix += torch.eye(matrix.size(-1), 
                                 dtype=matrix.dtype, 
                                 device=matrix.device) * (10 ** -k)
        
        return matrix
    
    elif method == 'dist.transform':
        print(matrix.shape)
        scale_tril = dist.transforms.LowerCholeskyTransform(matrix)
        return scale_tril
    
    elif method == 'robust_cov':
        V = _orthogonalize(V)
        # Create a robust covariance matrix
        # Get matrix dimensions
        n = V.shape[1]
        
        # Create lower triangular matrix with small diagonal stabilization
        L = torch.tril(V)
        
        # Add a small identity matrix to ensure positive definiteness
        L = L + torch.eye(n, device=V.device) * 1
        
        # Scale diagonal with the provided scaling vector
        L = L * torch.sqrt(S).unsqueeze(1)
        
        # Compute covariance: L * L.T ensures positive definiteness
        # return L @ L.T
        return torch.bmm(L, L.transpose(1, 2))

    elif method == '':
        matrix = torch.tril(torch.ones(V.shape[0], V.shape[0]),diagonal=-1).to(device=V.device) * V
        Q = matrix + torch.diag_embed(S) #torch.nn.functional.softplus(diags))
        # cov=(torch.tril(torch.ones(obs_dim,obs_dim),diagonal=-1)*a)+(torch.diag_embed(F.softplus(diags)))
        # L = torch.tril(matrix)  # Make it lower triangular
        # L = L + torch.eye(matrix.shape[0]).to(matrix.device) # Ensure diagonal is positive. Adding the identity matrix will ensure positivety.
        # Q = torch.mm(L, L.t())  # Compute A = LL^T
        return Q
    else:
        raise ValueError(f"Unsupported method: {method}")

def _orthogonalize(V):
    """
    Perform QR decomposition to orthogonalize the V matrix
    
    Args:
    V: Input matrix of shape (batch_size, output_dim, rank)
    
    Returns:
    Orthogonalized V matrix
    """
    # Reshape V for batch QR decomposition
    _, output_dim, rank = V.shape
    
    # STEP 1: Transpose the input tensor
    # Original shape: (batch_size, output_dim, rank)
    # Transposed shape: (batch_size, rank, output_dim)
    # This is required for torch.linalg.qr() to work correctly with batched inputs
    V_transposed = V.transpose(-2, -1)
   
    # STEP 2: Perform Batch QR Decomposition
    # QR decomposition breaks the matrix into:
    # Q: An orthogonal matrix (columns are orthonormal)
    # R: An upper triangular matrix
    # After decomposition:
    # Q shape: (batch_size, rank, output_dim)
    # R shape: (batch_size, rank, rank)
    Q, R = torch.linalg.qr(V_transposed)
   
    # STEP 3: Transpose Q back to original dimension order
    # Restores shape to (batch_size, output_dim, rank)
    Q = Q.transpose(-2, -1)
   
    # STEP 4: Sign Correction
    # Determine the signs of the diagonal elements of R
    # This ensures the first column of Q has a positive orientation
    # Signs shape: (batch_size, rank)
    signs = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
   
    # Multiply Q by the signs
    # unsqueeze(-1) adds a dimension to broadcast signs across output_dim
    # This adjusts the sign of each column in Q
    Q = Q * signs.unsqueeze(-1)
   
    return Q
