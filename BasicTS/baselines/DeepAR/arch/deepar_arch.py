import torch
import torch.nn as nn
import torch.nn.functional as F

# from .distributions import Gaussian
from prob.prob_head import ProbabilisticHead


class DeepAR(nn.Module):
    """
    Paper: DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks;
    Link: https://arxiv.org/abs/1704.04110;
    Ref Code:
        https://github.com/jingw2/demand_forecast
        https://github.com/husnejahan/DeepAR-pytorch
        https://github.com/arrigonialberto86/deepar
    Venue: International Journal of Forecasting 2020
    Task: Probabilistic Time Series Forecasting
    """

    def __init__(self, cov_feat_size, embedding_size, hidden_size, num_layers, use_ts_id, 
                id_feat_size=0, num_nodes=0, distribution_type="gaussian", prob_args={}) -> None:
        """Init DeepAR.

        Args:
            cov_feat_size (int): covariate feature size (e.g. time in day, day in week, etc.).
            embedding_size (int): output size of the input embedding layer.
            hidden_size (int): hidden size of the LSTM.
            num_layers (int): number of LSTM layers.
            use_ts_id (bool): whether to use time series id to construct spatial id embedding as additional features.
            id_feat_size (int, optional): size of the spatial id embedding. Defaults to 0.
            num_nodes (int, optional): number of nodes. Defaults to 0.
        """
        super().__init__()
        self.use_ts_id = use_ts_id
        # input embedding layer
        self.input_embed = nn.Linear(1, embedding_size)
        # spatial id embedding layer
        if use_ts_id:
            assert id_feat_size > 0, "id_feat_size must be greater than 0 if use_ts_id is True"
            assert num_nodes > 0, "num_nodes must be greater than 0 if use_ts_id is True"
            self.id_feat = nn.Parameter(torch.empty(num_nodes, id_feat_size))
            nn.init.xavier_uniform_(self.id_feat)
        else:
            id_feat_size = 0
        # the LSTM layer
        self.encoder = nn.LSTM(embedding_size+cov_feat_size+id_feat_size, hidden_size, num_layers, bias=True, batch_first=True)
        # the likelihood function
        self.distribution_type = distribution_type
        self.quantiles = prob_args['quantiles'] if 'quantiles' in prob_args.keys() else None
        self.prob_head = ProbabilisticHead(hidden_size, 1, distribution_type=self.distribution_type, prob_args=prob_args) #Gaussian(hidden_size, 1)

    def _sample_from_head(self, head_output, hidden_output=None):
        """Sample from the probabilistic head based on distribution type"""
        if self.distribution_type == "gaussian":
            # Extract mean and std from head output
            mu = head_output[..., 0]
            sigma = head_output[..., 1]
            # Sample from Gaussian
            gaussian = torch.distributions.Normal(mu, sigma)
            sample = gaussian.sample()
            
        elif self.distribution_type == "laplace":
            # Extract location and scale from head output
            loc = head_output[..., 0]
            scale = head_output[..., 1]
            # Sample from Laplace
            laplace = torch.distributions.Laplace(loc, scale)
            sample = laplace.sample()
            
        elif self.distribution_type == "quantile":
            # For quantile regression, use the median (50% quantile) as point forecast
            median_idx = self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2
            sample = head_output[..., median_idx]
            
        elif self.distribution_type == "i_quantile":
            raise NotImplementedError # -> Need to decide how to choose sample during training/where the quantile values can be arbitrary... -> try to reiterate for the 0.5 quantile?
            # For implicit quantile regression during training,
            # the prediction is already included in head_output
            if self.training:
                sample = head_output[..., 0]  # Just use the prediction without the tau
            else:
                # During inference, find the prediction for the median quantile
                median_idx = len(self.prob_head.quantiles) // 2
                sample = head_output[..., median_idx]
        return sample

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, train: bool, **kwargs) -> torch.Tensor:
        """Feed forward of DeepAR.
        Reference code: https://github.com/jingw2/demand_forecast/blob/master/deepar.py

        Args:
            history_data (torch.Tensor): history data. [B, L, N, C].
            future_data (torch.Tensor): future data. [B, L, N, C].
            train (bool): is training or not.
        """
        history_next = None
        # samples = []
        dist_params = []
        len_in, len_out = history_data.shape[1], future_data.shape[1]
        B, _, N, C = history_data.shape
        input_feat_full = torch.cat([history_data[:, :, :, 0:1], future_data[:, :, :, 0:1]], dim=1) # B, L_in+L_out, N, 1
        covar_feat_full = torch.cat([history_data[:, :, :, 1:], future_data[:, :, :, 1:]], dim=1) # B, L_in+L_out, N, C-1

        for t in range(1, len_in + len_out):
            if not (t > len_in and not train): # not in the decoding stage when inferecing
                history_next = input_feat_full[:, t-1:t, :, 0:1]
            embed_feat = self.input_embed(history_next)
            covar_feat = covar_feat_full[:, t:t+1, :, :]
            if self.use_ts_id:
                id_feat = self.id_feat.unsqueeze(0).expand(history_data.shape[0], -1, -1).unsqueeze(1)
                encoder_input = torch.cat([embed_feat, covar_feat, id_feat], dim=-1)
            else:
                encoder_input = torch.cat([embed_feat, covar_feat], dim=-1)
            # lstm
            B, _, N, C = encoder_input.shape # _ is 1
            encoder_input = encoder_input.transpose(1, 2).reshape(B * N, -1, C)
            _, (h, c) = self.encoder(encoder_input) if t == 1 else self.encoder(encoder_input, (h, c))

            # distribution proj
            if self.distribution_type in ["i_quantile"]: # since we need the batch size shape to construct the uniform quantile levels this is a bit ugly
                head_output = self.prob_head(F.relu(h[-1, :, :].view(B, N, -1))).reshape(B * N, -1)
                # print(head_output[..., 0].view(B, N, -1).unsqueeze(1).shape)
                # print(head_output[..., 1].view(B, N, -1)[:, 0:1, :].unsqueeze(1).shape)
                
                dist_params.append(torch.cat([head_output[..., 0].view(B, N, -1).unsqueeze(1), head_output[..., 1].view(B, N, -1)[:, 0:1, :].unsqueeze(1)], dim=1))
                # quantiles = head_output[..., 1].view(B, N, -1)[:, 0, :]
                # print(quantiles)
                # print(quantiles.shape)
                # head_output = torch.cat([], dim=1)
            else: 
                head_output = self.prob_head(F.relu(h[-1, :, :]))
                dist_params.append(head_output.view(B, N, -1).unsqueeze(1))
            if (t > len_in and not train): # not in the decoding stage when inferecing
                sample = self.prob_head.sample(head_output) #self._sample_from_head(head_output, None)
                # print(sample.shape)
                # print(sample)
                # print(B, N)
                history_next = sample.view(B, N).view(B, 1, N, 1)
            # print(head_output.view(B, N, -1).unsqueeze(1).shape)
            #history_next = self.gaussian_sample(mu, sigma).view(B, N).view(B, 1, N, 1)
            #mus.append(mu.view(B, N, 1).unsqueeze(1))
            #sigmas.append(sigma.view(B, N, 1).unsqueeze(1))
            # samples.append(history_next)
            assert not torch.isnan(history_next).any()

        # samples = torch.concat(samples, dim=1)
        #TODO also try to return the full prediction horizon and optimize it on that
        params = torch.concat(dist_params, dim=1)[:, -len_out:, :, :]
        # print(params[:, :, :, 0])
        # print(params[:, :, :, 1])
        # print(params.shape)
        # print(tut)
        #reals = input_feat_full[:, -params.shape[1]:, :, :]
        return params # {"prediction": params, }#"target": reals,}# "mus": mus, "sigmas": sigmas}
