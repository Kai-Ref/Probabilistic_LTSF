import torch
import torch.nn as nn
import torch.nn.functional as F
from .Transformer_EncDec import Encoder, EncoderLayer
from .SelfAttention_Family import FullAttention, AttentionLayer
from .Embed import DataEmbedding_inverted
import numpy as np
from basicts.utils import data_transformation_4_xformer

from prob.prob_head import ProbabilisticHead

class iTransformer(nn.Module):
    """
    Paper: iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
    Official Code: https://github.com/thuml/iTransformer
    Link: https://arxiv.org/abs/2310.06625
    Venue: ICLR 2024
    Task: Long-term Time Series Forecasting
    """
    def __init__(self, **model_args):
        super(iTransformer, self).__init__()
        self.pred_len = model_args['pred_len']
        self.seq_len = model_args['seq_len']
        self.output_attention = model_args['output_attention']
        self.enc_in = model_args['enc_in']
        self.dec_in = model_args['dec_in']
        self.c_out = model_args['c_out']
        self.factor = model_args["factor"]
        self.d_model = model_args['d_model']
        self.n_heads = model_args['n_heads']
        self.d_ff = model_args['d_ff']
        self.embed = model_args['embed']
        self.freq = model_args["freq"]
        self.dropout = model_args["dropout"]
        self.activation = model_args['activation']
        self.e_layers = model_args['e_layers']
        self.d_layers = model_args['d_layers']
        self.head_type = model_args['head_type']

        self.use_norm =model_args['use_norm']
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(self.seq_len, self.d_model, self.embed, self.freq,
                                                    self.dropout)

        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, self.factor, attention_dropout=self.dropout,
                                      output_attention=self.output_attention), self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        if self.head_type == 'probabilistic':
            self.distribution_type = model_args['distribution_type']
            self.prob_args = model_args['prob_args']
            self.projector = ProbabilisticHead(self.d_model, self.pred_len, self.distribution_type, prob_args=self.prob_args)
        else:
            self.projector = nn.Linear(self.d_model, self.pred_len, bias=True)

    def forward_xformer(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, x_dec: torch.Tensor,
                        x_mark_dec: torch.Tensor,
                        enc_self_mask: torch.Tensor = None, dec_self_mask: torch.Tensor = None,
                        dec_enc_mask: torch.Tensor = None) -> torch.Tensor:

        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape  # B L N
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # covariates (e.g timestamp) can be also embedded as tokens

        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N
        if self.head_type == 'probabilistic':
            dec_out = self.projector(enc_out).permute(0, 2, 1, 3)[:, :, :N, :] # bs x seq_len x num_series x num_params
            if self.use_norm:
                if self.distribution_type in ["gaussian", "laplace", "student_t", "m_lr_gaussian"]:
                    pred_means = dec_out[:, :, :, 0]
                    pred_means = pred_means * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
                    pred_means = pred_means + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
                    dec_out[:, :, :, 0] = pred_means
                    if self.distribution_type in ["m_lr_gaussian"]:
                        rank = self.prob_args['rank']
                        V = dec_out[..., 1:1+rank]
                        S = dec_out[..., 1+rank:].squeeze()

                        std = stdev.view(-1, 1, stdev.shape[-1], 1)  # [B, 1, D, 1]
                        V = V * std
                        S = S * stdev * stdev
                        dec_out[..., 1:] = torch.cat([V, S.unsqueeze(-1)], dim=-1)
                    else:
                        # For standard deviation parameter, scale it by the original stdev
                        pred_stds = dec_out[:, :, :, 1]
                        pred_stds = pred_stds * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
                        dec_out[:, :, :, 1] = pred_stds
                else: # for example quantile forecasts
                    # filter by dec_out[:, :self.pred_len, :, :] -> because i_quantile has +1 shape for the quantile levels
                    dec_out[:, :self.pred_len, :, :] = dec_out[:, :self.pred_len, :, :] * (stdev[:, 0, :].unsqueeze(1).unsqueeze(-1).repeat(1, self.pred_len, 1, dec_out.shape[-1]))
                    dec_out[:, :self.pred_len, :, :] = dec_out[:, :self.pred_len, :, :] + (means[:, 0, :].unsqueeze(1).unsqueeze(-1).repeat(1, self.pred_len, 1, dec_out.shape[-1]))
        else:
            dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]  # filter the covariates
            if self.use_norm:
                # De-Normalization from Non-stationary Transformer
                dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
                dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
                dec_out = dec_out.unsqueeze(-1) # was originally in forward return prediction.unsqueeze(-1)
        return dec_out

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                **kwargs) -> torch.Tensor:
        """

        Args:
            history_data (Tensor): Input data with shape: [B, L1, N, C]
            future_data (Tensor): Future data with shape: [B, L2, N, C]

        Returns:
            torch.Tensor: outputs with shape [B, L2, N, 1]
        """

        x_enc, x_mark_enc, x_dec, x_mark_dec = data_transformation_4_xformer(history_data=history_data,
                                                                             future_data=future_data,
                                                                             start_token_len=0)
        #print(x_mark_enc.shape, x_mark_dec.shape)
        prediction = self.forward_xformer(x_enc=x_enc, x_mark_enc=x_mark_enc, x_dec=x_dec, x_mark_dec=x_mark_dec)
        return prediction