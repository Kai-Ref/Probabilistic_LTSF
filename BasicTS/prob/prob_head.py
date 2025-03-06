import torch
import torch.nn as nn

class ProbabilisticHead(nn.Module):
    """
    Probabilistic prediction head that supports multiple distributions and quantile regression.
    
    Args:
        input_dim (int): Input dimension.
        output_dim (int): Output dimension.
        distribution_type (str): Type of distribution to use ("gaussian", "laplace", or "quantile").
        num_quantiles (int): Number of quantiles to predict (only used if distribution_type == "quantile").
    """
    def __init__(self, input_dim, output_dim, distribution_type="gaussian", num_quantiles=None):
        super(ProbabilisticHead, self).__init__()
        self.distribution_type = distribution_type
        self.layers = nn.ModuleList()
        if self.distribution_type in ["gaussian", "laplace"]:
            self.num_layers = 2
            self.activation = nn.Softplus()  # Activation for standard deviation or scale
        elif self.distribution_type == "quantile":
            assert num_quantiles is not None, "num_quantiles must be specified for quantile regression"
            self.num_layers = num_quantiles
        else:
            raise ValueError(f"Unsupported distribution type: {distribution_type}")
        for _ in range(self.num_layers):
            self.layers.append(nn.Linear(input_dim, output_dim))

    def forward(self, x):
        if self.distribution_type in ["gaussian", "laplace"]:
            predictions = [self.layers[0](x)]
            predictions.append(self.activation(self.layers[1](x)))
        elif self.distribution_type == "quantile":
            predictions = [layer(x) for layer in self.layers]
        
        predictions = torch.stack(predictions, dim=-1)
        return predictions