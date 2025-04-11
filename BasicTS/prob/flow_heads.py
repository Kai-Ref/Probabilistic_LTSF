import torch
import torch.nn as nn
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
        z = self.base_dist.rsample(sample_shape)
        # Transform to desired distribution (keeping gradient flow)
        x = self.flow.inverse(z)
        return x
    
    def cdf(self, value):
        z, _ = self.flow(value)
        return self.base_dist.cdf(z)

class FlowHead(BaseDistribution):
    """Flow-based distribution output."""
    def __init__(self, input_dim, output_dim, args):
        super().__init__(input_dim, output_dim, args)
        # self.context_net = nn.Sequential(
        #     nn.Linear(input_dim, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 32)
        # )
        self.base_head = GaussianHead(input_dim, output_dim, args)
        
        # Number of flow layers to use
        n_flows = getattr(args, 'n_flows', 1)
        hidden_dim = getattr(args, 'flow_hidden_dim', 16)
        flow_type = getattr(args, 'flow_type', 'sigmoidal')
        
        # Select flow type based on the argument
        flow_types = {
            'sigmoidal': SigmoidalFlow,
            'rectified': RectifiedFlow,
            'affine': AffineFlow,
            'identity': IdentityFlow,
        }
        
        if flow_type not in flow_types:
            raise ValueError(f"Unknown flow type: {flow_type}. Available types: {list(flow_types.keys())}")
        
        FlowClass = flow_types[flow_type]
        
        # Create deeper flows
        flows = [FlowClass(output_dim, hidden_dim) for _ in range(n_flows)]
        self.flow = FlowSequence(flows)
    
    def forward(self, x):
        return self.base_head(x)
    
    def __get_dist__(self, prediction):
        base_dist = self.base_head.__get_dist__(prediction) 
        
        # Return flow-based distribution
        return FlowDistribution(self.flow.to(prediction.device), base_dist)