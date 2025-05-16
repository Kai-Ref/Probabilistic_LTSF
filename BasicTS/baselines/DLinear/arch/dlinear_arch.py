import torch
import torch.nn as nn
from prob.prob_head import ProbabilisticHead


class moving_avg(nn.Module):
    """Moving average block to highlight the trend of time series"""

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size,
                                stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """Series decomposition block"""

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    """
        Paper: Are Transformers Effective for Time Series Forecasting?
        Link: https://arxiv.org/abs/2205.13504
        Official Code: https://github.com/cure-lab/DLinear
        Venue: AAAI 2023
        Task: Long-term Time Series Forecasting
    """
    def __init__(self, **model_args):
        super(DLinear, self).__init__()
        self.seq_len = model_args["seq_len"]
        self.pred_len = model_args["pred_len"]

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = model_args["individual"]
        self.channels = model_args["enc_in"]

        self.head_type = model_args['head_type']

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            if self.head_type == 'probabilistic':
                distribution_type = model_args['distribution_type']
                quantiles = model_args['quantiles']
                self.Heads = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len, self.pred_len))
                if self.head_type == 'probabilistic':
                    self.Heads.append(
                    ProbabilisticHead(self.pred_len, self.pred_len, distribution_type=distribution_type, quantiles=quantiles))

        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            if self.head_type == 'probabilistic':
                distribution_type = model_args['distribution_type']
                quantiles = model_args['quantiles']
                self.head = ProbabilisticHead(self.pred_len, self.pred_len, distribution_type=distribution_type, quantiles=quantiles)
        

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """Feed forward of DLinear.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """

        assert history_data.shape[-1] == 1      # only use the target feature
        x = history_data[..., 0]     # B, L, N
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(
                1), self.pred_len], dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(
                1), self.pred_len], dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        prediction = seasonal_output + trend_output
        # after the normal calulation append the probabilistic head
        if self.head_type == 'probabilistic':
            # the individual heads also have to be modelled differently
            if self.individual:
                predictions = []
                for i in range(self.channels):
                    out = self.Heads[i](prediction[:, i, :]).unsqueeze(1)
                    predictions.append(out)
                prediction = torch.cat(predictions, dim=1)
            else: 
                prediction = self.head(prediction)
            return prediction.permute(0, 2, 1, 3) # [B, L, N, Params]
        else: 
            return prediction.permute(0, 2, 1).unsqueeze(-1)  # [B, L, N, 1]