import pyro
import pyro.distributions as pdist
from pyro.distributions.transforms import Planar, AffineAutoregressive, ComposeTransform, ComposeTransformModule
import torch
import torch.nn as nn
from pyro.nn import DenseNN
import torch.distributions as dist
from .dist_heads import *


class BaseFlow(nn.Module):
    """Base class for normalizing flow layers."""
    def forward(self, x, context=None):
        """Forward transformation: x -> z with log determinant."""
        raise NotImplementedError
    
    def inverse(self, z, context=None):
        """Inverse transformation: z -> x."""
        raise NotImplementedError

class SigmoidalFlow(BaseFlow):
    """Single layer of Sigmoidal Flow."""
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.a = nn.Parameter(torch.randn(input_dim))
        self.b = nn.Parameter(torch.randn(input_dim))
        self.network = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x, context=None):
        """Forward transformation: x -> z."""
        x_in = x.unsqueeze(-1)
        h = self.network(x_in).squeeze(-1)
        z = x + self.a * torch.sigmoid(self.b * h)
        # Compute log determinant of Jacobian
        sigmoid_arg = self.b * h
        log_det = torch.log(1 + self.a * self.b * nn.functional.sigmoid(sigmoid_arg) * (1 - nn.functional.sigmoid(sigmoid_arg)))
        return z, log_det

    def inverse(self, z, context=None, max_iter=100, tol=1e-6):
        """Inverse transformation: z -> x using fixed-point iteration."""
        x = z.clone()  # Initial guess
        for _ in range(max_iter):
            x_next = z - self.a * torch.sigmoid(self.b * self.network(x.unsqueeze(-1)).squeeze(-1))
            if torch.max(torch.abs(x_next - x)) < tol:
                break
            x = x_next
        return x

class RectifiedFlow(BaseFlow):
    """Rectified Flow layer implementation."""
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        # Scaling factor for stability
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        self.n_hutchinson_samples = 1
    
    def forward(self, x, context=None):
        """Forward transformation: x -> z."""
        v = self.network(x)
        z = x + self.scale * V
        if self.training:
            # Define the transformation function for the estimator
            def transform_fn(inputs):
                return inputs + self.scale * self.network(inputs)
            
            # Estimate log determinant
            log_det = hutchinson_trace_estimator(transform_fn, x, self.n_hutchinson_samples)
        else:
            # Skip calculation during inference for efficiency
            log_det = torch.zeros(x.shape[0], device=x.device)
            
        return z, log_det
    
    def inverse(self, z, context=None, max_iter=50, tol=1e-6):
        """Inverse transformation using fixed point iteration."""
        x = z.clone()  # Initial guess
        for _ in range(max_iter):
            x_next = z - self.scale * self.network(x)
            if torch.max(torch.abs(x_next - x)) < tol:
                break
            x = x_next
        return x

class AffineFlow(BaseFlow):
    """Simple affine transformation flow."""
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        # Scale and shift parameters
        self.scale_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.shift_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
        )
        # Small constant for numerical stability
        self.eps = 1e-6
    
    def forward(self, x, context=None):
        """Forward transformation: x -> z."""
        dummy_input = torch.ones((x.shape[0], 1), device=x.device)
        log_scale = self.scale_net(dummy_input)
        shift = self.shift_net(dummy_input)
        scale = torch.exp(log_scale)
        z = scale * x + shift
        log_det = log_scale.sum(dim=1)
        return z, log_det
    
    def inverse(self, z, context=None):
        """Inverse transformation: z -> x."""
        dummy_input = torch.ones((z.shape[0], 1), device=z.device)
        log_scale = self.scale_net(dummy_input)
        shift = self.shift_net(dummy_input)
        scale = torch.exp(log_scale)
        x = (z - shift) / scale
        return x

class IdentityFlow(BaseFlow):
    """Identity flow that doesn't transform the input."""
    def __init__(self, input_dim, hidden_dim=None):
        super().__init__()
    
    def forward(self, x, context=None):
        """Forward transformation: x -> x."""
        return x, torch.zeros_like(x).to(x.device)
    
    def inverse(self, z, context=None):
        """Inverse transformation: z -> z."""
        return z

def hutchinson_trace_estimator(transform_fn, x, n_samples=1):
    """
    Estimate the trace of the Jacobian using the Hutchinson trace estimator.
    
    Args:
        transform_fn: Callable that computes the transformation whose Jacobian trace we want to estimate.
                     Should accept x as input and return the transformed output.
        x: Input tensor of shape [batch_size, dim]
        n_samples: Number of random samples to use for the estimation (more samples = more accurate)
    
    Returns:
        log_det: Estimated log determinant of shape [batch_size]
    """
    batch_size = x.shape[0]
    log_det = torch.zeros(batch_size, device=x.device)
    
    # Ensure we can compute gradients
    with torch.enable_grad():
        x_requires_grad = x.detach().requires_grad_(True)
        transform_output = transform_fn(x_requires_grad)
        
        for _ in range(n_samples):
            # Random probing vector
            epsilon = torch.randn_like(x_requires_grad)
            
            # Compute Jacobian-vector product
            jvp = torch.autograd.grad(
                outputs=transform_output,
                inputs=x_requires_grad,
                grad_outputs=epsilon,
                create_graph=True,
                retain_graph=True
            )[0]
            
            # Estimate trace using dot product
            trace_estimate = torch.sum(jvp * epsilon, dim=1)
            log_det += trace_estimate
    
    # Average over samples
    log_det = log_det / n_samples
    return log_det

class FlowSequence(nn.Module):
    """A sequence of flow transformations."""
    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)
    
    def forward(self, x):
        """Forward transformation with log determinant."""
        log_det_sum = torch.zeros_like(x).to(x.device)
        z = x
        for flow in self.flows:
            z, log_det = flow(z)
            if log_det.isnan().any():
                print(log_det)
                log_det = torch.nan_to_num(log_det, nan=0.0)
            log_det_sum = log_det_sum + log_det
        return z, log_det_sum
    
    def inverse(self, z):
        """Inverse transformation."""
        x = z
        for flow in reversed(self.flows):
            x = flow.inverse(x)
        return x

class FlowDistribution(dist.Distribution):
    """Custom PyTorch distribution backed by a normalizing flow."""
    def __init__(self, flow, base_dist):
        super().__init__(validate_args=False)
        self.flow = flow
        self.base_dist = base_dist
    
    def log_prob(self, value):
        # Transform value to base distribution space
        z, log_det = self.flow(value)
        # Compute log probability in base distribution space
        if type(self.base_dist) == list:
            nll = []
            assert len(self.base_dist) == z.shape[-1]
            for i, dist in enumerate(self.base_dist):
                # compute log prob across all batches
                log_prob = dist.log_prob(z[:, :, i])        
                nll.append(log_prob)
            # Sum log likelihood values for the single series, then it has shape equal to batch size
            log_likelihood = sum(nll)
            log_prob = -torch.mean(log_likelihood) # * mask)
        else:
            log_prob = self.base_dist.log_prob(z)
        # Correct for change of variables
        # print(f"{log_prob.mean()}- {log_det.mean()} = {(log_prob - log_det).mean()}")
        return log_prob - log_det
    
    def sample(self, sample_shape=torch.Size()):
        # Sample from base distribution
        z = self.base_dist.sample(sample_shape)
        # Transform to desired distribution
        x = self.flow.inverse(z)
        return x
    
    def rsample(self, sample_shape=torch.Size()):
        # Reparameterized sampling from base distribution
        if type(self.base_dist) == list:
            samples = []
            for i, distribution in enumerate(self.base_dist):
                samples.append(distribution.rsample(sample_shape))
            z = torch.stack(samples, dim=-1)
        else:
            z = self.base_dist.rsample(sample_shape)
        # Transform to desired distribution (keeping gradient flow)
        x = self.flow.inverse(z)
        return x
    
    def cdf(self, value):
        z, _ = self.flow(value)
        return self.base_dist.cdf(z)


class PyroFlowWrapper(nn.Module):
    """Wraps Pyro flows as a torch.nn.Module for compatibility."""
    def __init__(self, flow_type, input_dim, n_flows=1):
        super().__init__()
        flow_types = {
            'planar': lambda: Planar(input_dim),
            'affine_autoregressive': lambda: AffineAutoregressive(input_dim),
            # Add more from pyro.distributions.transforms if needed
        }

        if flow_type not in flow_types:
            raise ValueError(f"Unknown flow type: {flow_type}")

        self.flow = ComposeTransformModule([flow_types[flow_type]() for _ in range(n_flows)])

    def forward(self, x):
        return self.flow(x), self.flow.log_abs_det_jacobian(x, self.flow(x))

    def inverse(self, z):
        return self.flow.inv(z)

class PyroFlowDistribution(dist.Distribution):
    """Distribution with a Pyro transform-based flow."""
    def __init__(self, flow, base_dist):
        super().__init__(validate_args=False)
        self.flow = flow
        self.base_dist = base_dist
        self.trans_dist = pdist.TransformedDistribution(base_dist, self.flow.flow)

    def log_prob(self, value):
        return self.trans_dist.log_prob(value)

    def sample(self, sample_shape=torch.Size()):
        return self.trans_dist.sample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        return self.trans_dist.rsample(sample_shape)

    def cdf(self, value):
        return self.trans_dist.cdf(value)


class FlowHead(BaseDistribution):
    """Flow-based distribution output."""
    def __init__(self, input_dim, output_dim, prob_args={}):
        super().__init__(input_dim, output_dim, prob_args=prob_args)
        # self.context_net = nn.Sequential(
        #     nn.Linear(input_dim, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 32)
        # )
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
        # "copula": CopulaHead,  # Copula-based multivariate head (see args for copula/marginal config)
        # "gcopula": GaussianCopulaHead,  # Copula-based multivariate head (see args for copula/marginal config)
        }

        if prob_args['base_distribution'] not in DISTRIBUTIONS:
            raise ValueError(f"Unsupported distribution type: {prob_args['base_distribution']}")
        self.base_head = DISTRIBUTIONS[prob_args['base_distribution']](input_dim, output_dim, prob_args=prob_args['base_prob_args'])
        
        # Number of flow layers to use
        n_flows = getattr(prob_args, 'n_flows', 1)
        hidden_dim = getattr(prob_args, 'flow_hidden_dim', 16)
        flow_type = getattr(prob_args, 'flow_type', 'sigmoidal')
        
        # Select flow type based on the argument
        flow_types = {
            'sigmoidal': SigmoidalFlow,
            'rectified': RectifiedFlow,
            'affine': AffineFlow,
            # 'identity': IdentityFlow,
        }
        
        if flow_type not in flow_types:
            raise ValueError(f"Unknown flow type: {flow_type}. Available types: {list(flow_types.keys())}")
        
        FlowClass = flow_types[flow_type]
        
        # Create deeper flows
        flows = [FlowClass(output_dim, hidden_dim) for _ in range(n_flows)]
        # self.flow = FlowSequence(flows)
        # self.flow = PyroFlowWrapper(flow_type='affine_autoregressive', input_dim=output_dim, n_flows=n_flows)
    
        # def forward(self, x):
        #     # Note: It may look odd to only return the base head output, 
        #     # however when training with __get_dist__, i.e. NLL, the parameters of the Flow are also updated
        #     return self.base_head(x)
        
        # def __get_dist__(self, prediction):
        #     base_dist = self.base_head.__get_dist__(prediction) 
        #     return PyroFlowDistribution(self.flow.to(prediction.device), base_dist)
        n_flows = prob_args.get('n_flows', 1)
        hidden_dims = prob_args.get('flow_hidden_dims', [64, 64])
        self.flow = self._build_pyro_flow('affine_autoregressive', output_dim, hidden_dims, n_flows)

    def _build_pyro_flow(self, flow_type, input_dim, hidden_dims, n_flows):
        """Construct a stable, cache-free Pyro flow."""
        flow_type = lambda: AffineAutoregressive(
                input_dim,
                arn=DenseNN(input_dim, hidden_dims)
            ),
        transforms = []
        for _ in range(n_flows):
            if flow_type == 'affine_autoregressive':
                arn = DenseNN(input_dim, hidden_dims, param_dims=[input_dim, input_dim])
                transforms.append(AffineAutoregressive(arn))
            elif flow_type == 'spline':
                nn = DenseNN(input_dim // 2, hidden_dims, param_dims=[input_dim * 3])  # 3 for spline (width, height, derivative)
                transforms.append(SplineCoupling(input_dim // 2, nn))
            else:
                raise ValueError(f"Unsupported flow type: {flow_type}")
        return ComposeTransformModule(transforms)

    def forward(self, x):
        return self.base_head(x)

    def __get_dist__(self, prediction):
        base_dist = self.base_head.__get_dist__(prediction)

        # Handle single or list of base distributions
        if isinstance(base_dist, list):
            # Each distribution must be transformed separately
            transformed = [
                pdist.TransformedDistribution(bd, self.flow.to(prediction.device))
                for bd in base_dist
            ]
            return transformed
        else:
            return pdist.TransformedDistribution(base_dist, self.flow.to(prediction.device))