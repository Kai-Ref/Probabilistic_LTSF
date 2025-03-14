import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch
import torch.nn as nn

class ProbabilisticHead(nn.Module):
    def __init__(self, input_dim, output_dim, distribution_type="gaussian", quantiles=[]):
        super(ProbabilisticHead, self).__init__()
        self.distribution_type = distribution_type
        self.quantiles = quantiles

        if self.distribution_type in ["gaussian", "laplace"]:
            self.num_layers = 2
            self.activation = nn.Softplus()
            self.layers = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(self.num_layers)])
        elif self.distribution_type == "quantile":
            assert quantiles is not None, "num_quantiles must be specified for quantile regression"
            self.num_layers = len(quantiles)
            self.layers = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(self.num_layers)])
        elif self.distribution_type == "i_quantile":
            self.num_layers = 1
            self.quantile_embed_dim = 60
            self.cos_embedding_dim = self.quantile_embed_dim
            self.quantile_embedding = nn.Linear(self.cos_embedding_dim, self.quantile_embed_dim)
            self.qr_head = nn.Linear(input_dim + self.quantile_embed_dim, output_dim)
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported distribution type: {distribution_type}")

    def forward(self, x):
        batch_size = x.size(0)

        if self.distribution_type in ["gaussian", "laplace"]:
            mu = self.layers[0](x)
            sigma = self.activation(self.layers[1](x))
            return torch.stack([mu, sigma], dim=-1)
        elif self.distribution_type == "quantile":
            return torch.stack([layer(x) for layer in self.layers], dim=-1)
        elif self.distribution_type == "i_quantile":
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

    def predict_with_quantiles(model, x, quantile_levels):
        """
        Generate predictions for specific quantile levels during inference.

        Args:
            model (nn.Module): The model containing the probabilistic head.
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].
            quantile_levels (list or torch.Tensor): List of quantile levels to predict.

        Returns:
            torch.Tensor: Predictions of shape [batch_size, num_quantiles, output_dim].
        """
        self.eval()  # Ensure the model is in evaluation mode
        quantile_levels = torch.tensor(quantile_levels, device=x.device).float()  # Convert to tensor

        predictions = []
        with torch.no_grad():
            for tau in quantile_levels:
                tau_tensor = torch.full((x.size(0), 1), tau, device=x.device)  # Same tau for entire batch
                pred = model.head(x, tau_tensor)  # Forward pass for this quantile
                predictions.append(pred)

        return torch.stack(predictions, dim=1)  # [batch_size, num_quantiles, output_dim]

class ProbabilisticHead_1(nn.Module):
    def __init__(self, input_dim, output_dim, distribution_type="gaussian", quantiles=None):
        super(ProbabilisticHead, self).__init__()
        self.distribution_type = distribution_type
        self.layers = nn.ModuleList()
        if self.distribution_type in ["gaussian", "laplace"]:
            self.num_layers = 2
            self.activation = nn.Softplus()  # Activation for standard deviation or scale
        elif self.distribution_type == "quantile":
            assert quantiles is not None, "num_quantiles must be specified for quantile regression"
            self.num_layers = len(quantiles)
        elif self.distribution_type == "i_quantile":
            self.num_layers = 1
            input_dim = input_dim + 1
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
        elif self.distribution_type == "i_quantile":
            u = torch.rand(x.size(0), 1).to(x.device)  # quantile level for which to produce the prediction
            x_with_quantile = torch.cat([x, u], dim=-1)  # Concatenate quantile to input
            print(x_with_quantile.shape)
            predictions = self.layers[0](x_with_quantile)
            print(predictions.shape)
            # Append the sampled quantile value to the predictions
            predictions_with_quantile = torch.cat([predictions, u], dim=-1)
            print(predictions_with_quantile.shape)
            return predictions_with_quantile

        predictions = torch.stack(predictions, dim=-1)
        return predictions

