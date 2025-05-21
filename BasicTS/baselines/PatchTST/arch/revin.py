# code from https://github.com/ts-kim/RevIN, with minor modifications

import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False, distribution_type=None, prob_args={}):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()
        self.distribution_type = distribution_type
        self.prob_args = prob_args

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        x = x.clone()  # Clone to avoid in-place ops on a view
        if self.distribution_type in ['gaussian', 'laplace', 'student_t']:
            if self.affine:
                denom = self.affine_weight + self.eps**2
                x_mean = (x[..., 0] - self.affine_bias) / denom
                x_var = x[..., 1] / denom
                x_var = x_var * self.stdev
            else:
                x_mean = x[..., 0]
                x_var = x[..., 1] * self.stdev

            x_mean = x_mean * self.stdev
            if self.subtract_last:
                x_mean = x_mean + self.last
            else:
                x_mean = x_mean + self.mean

            x = torch.stack([x_mean, x_var], dim=-1)

        elif self.distribution_type in ['m_lr_gaussian']:
            # adjust mean
            if self.affine:
                denom = self.affine_weight + self.eps**2
                x_mean = (x[..., 0] - self.affine_bias) / denom
            else:
                x_mean = x[..., 0]
            x_mean = x_mean * self.stdev

            if self.subtract_last:
                x_mean = x_mean + self.last
            else:
                x_mean = x_mean + self.mean

            # then adjust variance
            rank = self.prob_args['rank']
            V = x[..., 1:1+rank]
            S = x[..., 1+rank:].squeeze()

            if self.affine:
                denom_sq = (self.affine_weight + self.eps**2)**2
                V = V / (self.affine_weight + self.eps**2).view(1, 1, -1, 1)  # Match std scale
                S = S / denom_sq.view(1, 1, -1)  # Undo variance scaling

            std = self.stdev.view(-1, 1, self.stdev.shape[-1], 1)  # [B, 1, D, 1]
            V = V * std
            
            # Rescale S: [batch, ..., D]
            S = S * self.stdev * self.stdev
            x_var = torch.cat([V, S.unsqueeze(-1)], dim=-1)

            # Stack back the mean and variance/std (keep last dim size same)
            x = torch.cat([x_mean.unsqueeze(-1), x_var], dim=-1)

        elif self.distribution_type in ['quantile', 'i_quantile']:
            if self.affine:
                x = x - self.affine_bias.unsqueeze(-1)
                x = x / (self.affine_weight.unsqueeze(-1) + self.eps*self.eps)
            x = x * self.stdev.unsqueeze(-1)
            if self.subtract_last:
                x = x + self.last.unsqueeze(-1)
            else:
                x = x + self.mean.unsqueeze(-1)
        else: # previous implementation for point forecasts
            if self.affine:
                x = x - self.affine_bias
                x = x / (self.affine_weight + self.eps*self.eps)
            x = x * self.stdev
            if self.subtract_last:
                x = x + self.last
            else:
                x = x + self.mean
        return x
