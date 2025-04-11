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

    def __get_dist__(self, prediction):
        raise NotImplementedError("Subclasses must implement __get_dist__ method.")

    def sample(self, head_output, num_samples=1):
        """Sample from the probabilistic head output."""
        return self.__get_dist__(head_output).rsample((num_samples,))

class MultivariateGaussianHead(BaseDistribution):
    def __init__(self, input_dim, output_dim, args):
        """
        Predict a low-rank factorization of the covariance matrix
        
        Args:
        input_dim: Dimensionality of input features
        output_dim: Dimensionality of the output distribution
        rank: Rank of the low-rank factorization (default is 5)
        """
        super().__init__(input_dim, output_dim, args) 
        self.rank = 96
        # Mean prediction layer
        self.mean_layer = nn.Linear(input_dim, output_dim)
        
        # Low-rank factorization layers
        # V will be output_dim x rank matrix
        # S will be rank-sized vector of scaling factors
        self.V_layer = nn.Linear(input_dim, output_dim * self.rank)
        self.S_layer = nn.Linear(input_dim, self.rank)
    
    def forward(self, x):
        # Predict mean
        mean = self.mean_layer(x)
        # Predict low-rank factorization components
        # Reshape V to be (batch_size, output_dim * var_dim, rank)
        V = self.V_layer(x).view(x.shape[0], self.output_dim, self.rank)

        # Predict scaling factors with softplus to ensure positivity
        # Shape will be (batch_size, rank)
        S = torch.nn.functional.softplus(self.S_layer(x))
        S = S.unsqueeze(1).repeat(1, V.shape[1], 1)
        
        # Combine mean, V, and S into a single tensor
        # This allows for potential dropout or other operations
        return torch.cat([
            mean.unsqueeze(-1),  # First: mean tensor
            V,     # Second: V matrix
            S  # Last: scaling factors
        ], dim=-1)

    def __get_dist__(self, prediction):
        distributions = []
        rank = self.rank
        mean, V_full, S_full = prediction[..., 0], prediction[..., 1:-rank], prediction[..., -rank:]
        S_full = S_full[:, 0, :, :] #torch.abs(S[:, 0, :, :])
        S_full = torch.clamp(S_full, min=1e-6)
        for i in range(prediction.shape[-2]):
            S = S_full[:, i, :]
            V = V_full[:, :, i, :]
            Q = ensure_positive_definite_matrix(V, S, method='robust_cov')
            distributions.append(dist.MultivariateNormal(mean[:, :, i], Q))
        return distributions

    def sample(self, head_output, num_samples=1):
        """Since slightly different behavior overwrite the sampling function."""
        batch_size, output_dim, num_series = head_output[..., 0].shape
        samples = torch.zeros(num_samples, batch_size, output_dim, num_series, device=head_output.device)
        distributions = self.__get_dist__(head_output)
        for i, dist in enumerate(distributions):
            samples[:, :, :, i] = dist.rsample((num_samples,))
        return samples

class LowRankMultivariateGaussianHead(BaseDistribution):
    def __init__(self, input_dim, output_dim, args):
        """
        Predict a low-rank factorization of the covariance matrix
        
        Args:
        input_dim: Dimensionality of input features
        output_dim: Dimensionality of the output distribution
        rank: Rank of the low-rank factorization (default is 5)
        """
        super().__init__(input_dim, output_dim, args) 
        self.rank = 96
        # Mean prediction layer
        self.mean_layer = nn.Linear(input_dim, output_dim)
        
        # Low-rank factorization layers
        # V will be output_dim x rank matrix
        # S will be rank-sized vector of scaling factors
        self.V_layer = nn.Linear(input_dim, output_dim * self.rank)
        self.S_layer = nn.Linear(input_dim, self.rank)
    
    def forward(self, x):
        # Predict mean
        mean = self.mean_layer(x)
        # Predict low-rank factorization components
        # Reshape V to be (batch_size, output_dim * var_dim, rank)
        V = self.V_layer(x).view(x.shape[0], self.output_dim, self.rank)

        # Predict scaling factors with softplus to ensure positivity
        # Shape will be (batch_size, rank)
        S = torch.nn.functional.softplus(self.S_layer(x))
        S = S.unsqueeze(1).repeat(1, V.shape[1], 1)
        
        # Combine mean, V, and S into a single tensor
        # This allows for potential dropout or other operations
        return torch.cat([
            mean.unsqueeze(-1),  # First: mean tensor
            V,     # Second: V matrix
            S  # Last: scaling factors
        ], dim=-1)

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

    def sample(self, head_output, num_samples=1):
        """Since slightly different behavior overwrite the sampling function."""
        batch_size, output_dim, num_series = head_output[..., 0].shape
        samples = torch.zeros(num_samples, batch_size, output_dim, num_series, device=head_output.device)
        distributions = self.__get_dist__(head_output)
        for i, dist in enumerate(distributions):
            samples[:, :, :, i] = dist.rsample((num_samples,))
        return samples

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
    def __init__(self, input_dim, output_dim, args):
        super().__init__(input_dim, output_dim, args)
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
    def __init__(self, input_dim, output_dim, args):
        super().__init__(input_dim, output_dim, args)
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
    def __init__(self, input_dim, output_dim, args):
        super().__init__(input_dim, output_dim, args)
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
    def __init__(self, input_dim, output_dim, args):
        super().__init__(input_dim, output_dim, args)
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
    def __init__(self, input_dim, output_dim, args):
        super().__init__(input_dim, output_dim, args)
        self.concentration_layer = nn.Linear(input_dim, output_dim)
        self.activation = nn.Softplus()

    def forward(self, x):
        concentration = self.activation(self.concentration_layer(x)) + 1e-6
        return concentration.unsqueeze(-1)

    def __get_dist__(self, prediction):
        concentration = prediction[..., 0]
        concentration = torch.clamp(concentration, min=1e-6)
        return dist.Dirichlet(concentration)


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
