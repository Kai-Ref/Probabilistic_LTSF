# ---------------------------------------------------------------------------------
# Portions of this file are derived from PatchTST
# - Source: https://github.com/yuqinie98/PatchTST/tree/main
# - Paper: PatchTST: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers
# - License: Apache-2.0

# We thank the authors for their contributions.
# -----
# ----------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from probts.model.forecaster import Forecaster
from probts.model.nn.arch.PatchTSTModule.PatchTST_backbone import PatchTST_backbone
from probts.model.nn.arch.PatchTSTModule.PatchTST_layers import series_decomp

# =============================
# Example Loss Function (Gaussian NLL)
# =============================
import torch.distributions as dist
def gaussian_nll_loss(targets, mean, std):
    """Computes Gaussian Negative Log-Likelihood (NLL) Loss."""
    dist_normal = dist.Normal(mean, std)
    log_likelihood = dist_normal.log_prob(targets)
    return -log_likelihood.mean()
    
import torch.nn.functional as F
def quantile_loss(y_true, y_pred, quantiles):
    """
    Computes the quantile loss for multiple quantiles.

    Parameters:
        y_pred (torch.Tensor): Predicted quantile values, shape [bs x n_vars x target_window x num_quantiles].
        y_true (torch.Tensor): Ground truth values, shape [bs x n_vars x target_window].
        quantiles (list): List of quantiles (e.g., [0.1, 0.5, 0.9]).

    Returns:
        torch.Tensor: Scalar loss value.
    """
    losses = []
    for i, q in enumerate(quantiles):
        errors = y_true - y_pred[..., i]  # Compute residuals (true - predicted)
        loss = torch.max(q * errors, (q - 1) * errors)  # Pinball loss per quantile
        losses.append(loss.mean())  # Average over all samples
    
    return torch.mean(torch.stack(losses))  # Average across all quantiles

class PatchTST_prob(Forecaster):
    def __init__(
        self,
        stride: int,
        patch_len: int,
        padding_patch: str = None,
        max_seq_len: int = 1024,
        n_layers:int = 3,
        n_heads = 16,
        d_k: int = None,
        d_v: int = None,
        d_ff: int = 256,
        attn_dropout: float = 0.,
        dropout: float = 0.,
        act: str = "gelu", 
        res_attention: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = 'zeros',
        learn_pe: bool = True,
        attn_mask: Optional[Tensor] = None,
        individual: bool = False,
        head_type: str = 'flatten',
        padding_var: Optional[int] = None, 
        revin: bool = True,
        key_padding_mask: str = 'auto',
        affine: bool = False,
        subtract_last: bool = False,
        decomposition: bool = False,
        kernel_size: int = 3,
        fc_dropout: float = 0.,
        head_dropout: float = 0.,
        f_hidden_size: int = 40,
        quantiles: list = [0.01, 0.05, 0.1, 0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 0.95, 0.99],
        **kwargs
    ):
        super().__init__(**kwargs)
        
        if self.input_size != self.target_dim:
            self.enc_linear = nn.Linear(
                in_features=self.input_size, out_features=self.target_dim
            )
        else:
            self.enc_linear = nn.Identity()

        # Load parameters
        c_in = self.input_size
        context_window = self.context_length
        target_window = self.prediction_length

        # Model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=f_hidden_size,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=False, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last)
            self.model_res = PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=f_hidden_size,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=False, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last)
        else:
            self.model = PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=f_hidden_size,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=False, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, quantiles=quantiles)
        if head_type == "prob":
            self.loss_fn = gaussian_nll_loss #nn.MSELoss(reduction='none')
        elif head_type == "quantile":
            self.quantiles = quantiles
            self.loss_fn = quantile_loss
        self.head_type = head_type
    
    def forward(self, x):
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            if self.head_type == 'prob':
                mean, std = self.model(x)
                mean, std = mean.permute(0,2,1), std.permute(0,2,1)    # x: [Batch, Input length, Channel]
                return mean, std
            elif self.head_type == 'quantile':
                output = self.model(x)
                output = output.permute(0,2,1,3)    # x: [Batch, Input length, Channel, Quantiles]
                return output
        
    def loss(self, batch_data):
        inputs = self.get_inputs(batch_data, 'encode')
        inputs = self.enc_linear(inputs)
        # outputs = self(inputs)
        if self.head_type == 'prob':
            mean, std = self(inputs)
            loss = self.loss_fn(batch_data.future_target_cdf, mean, std).unsqueeze(-1)
        elif self.head_type == 'quantile':
            outputs = self(inputs)
            loss = self.loss_fn(batch_data.future_target_cdf, outputs, self.quantiles)
        loss = self.get_weighted_loss(batch_data, loss)
        return loss.mean()

    def forecast(self, batch_data, num_samples=None):
        inputs = self.get_inputs(batch_data, 'encode')
        inputs = self.enc_linear(inputs)
        # outputs = self(inputs)
        if self.head_type == 'prob':
            mean, std = self(inputs)
            # Optionally return samples
            if num_samples:
                dist_normal = dist.Normal(mean, std)
                samples = dist_normal.rsample((num_samples,))  # [Samples, Batch, Target Length, Target Dim]
                return samples.permute(1, 0, 2, 3)  # [Batch, Samples, Target Length, Target Dim]
            return mean, std  # Return mean and std if no sampling
        elif self.head_type == 'quantile':
            outputs = self(inputs)
            return outputs#.unsqueeze(2)
            
