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

    # def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, train: bool, **kwargs) -> torch.Tensor:
    #     """Feed forward of DeepAR.
    #     Reference code: https://github.com/jingw2/demand_forecast/blob/master/deepar.py

    #     Args:
    #         history_data (torch.Tensor): history data. [B, L, N, C].
    #         future_data (torch.Tensor): future data. [B, L, N, C].
    #         train (bool): is training or not.
    #     """
    #     len_in, len_out = history_data.shape[1], future_data.shape[1]
    #     B, _, N, C = history_data.shape
    #     input_feat_full = torch.cat([history_data[:, :, :, 0:1], future_data[:, :, :, 0:1]], dim=1) # B, L_in+L_out, N, 1
    #     covar_feat_full = torch.cat([history_data[:, :, :, 1:], future_data[:, :, :, 1:]], dim=1) # B, L_in+L_out, N, C-1

    #     # For non-training (inference), we need to generate step by step
    #     if not train:
    #         return self._forward_inference(input_feat_full, covar_feat_full, len_in, len_out, B, N)
        
    #     # Training mode: prepare all inputs at once for better gradient flow
    #     all_encoder_inputs = []
        
    #     for t in range(len_in + len_out):
    #         if t < len_in:
    #             # Use ground truth history
    #             history_next = input_feat_full[:, t:t+1, :, 0:1]
    #         else:
    #             # During training, use ground truth for teacher forcing
    #             history_next = input_feat_full[:, t:t+1, :, 0:1]
            
    #         embed_feat = self.input_embed(history_next)
    #         covar_feat = covar_feat_full[:, t:t+1, :, :]
            
    #         if self.use_ts_id:
    #             id_feat = self.id_feat.unsqueeze(0).expand(B, -1, -1).unsqueeze(1)
    #             encoder_input = torch.cat([embed_feat, covar_feat, id_feat], dim=-1)
    #         else:
    #             encoder_input = torch.cat([embed_feat, covar_feat], dim=-1)
            
    #         all_encoder_inputs.append(encoder_input)
        
    #     # Stack all inputs: B, T, N, C
    #     all_encoder_inputs = torch.cat(all_encoder_inputs, dim=1)
    #     B, T, N, C_enc = all_encoder_inputs.shape
        
    #     # Reshape for LSTM: (B*N, T, C)
    #     lstm_input = all_encoder_inputs.transpose(1, 2).reshape(B * N, T, C_enc)
        
    #     # Single LSTM forward pass
    #     lstm_output, (h_final, c_final) = self.encoder(lstm_input)
        
    #     # Process outputs for each timestep
    #     dist_params = []
        
    #     for t in range(T):
    #         # Get hidden state for timestep t: (B*N, hidden_size)
    #         hidden_t = lstm_output[:, t, :]
            
    #         # Reshape to (B, N, hidden_size)
    #         hidden_t = hidden_t.view(B, N, -1)
            
    #         # Distribution projection
    #         if self.distribution_type in ["i_quantile"]:
    #             resample_u = True if t == 0 else False
    #             head_output = self.prob_head(F.relu(hidden_t), resample_u=resample_u).reshape(B * N, -1)
                
    #             if train:
    #                 u = head_output[..., 1].view(B, N, -1).unsqueeze(1)
    #                 dist_params.append(head_output[..., 0].view(B, N, -1).unsqueeze(1))
    #             else: 
    #                 dist_params.append(head_output.view(B, N, -1).unsqueeze(1))
    #         else: 
    #             head_output = self.prob_head(F.relu(hidden_t))
    #             dist_params.append(head_output.unsqueeze(1))
            
    #         # Debug information for prediction phase
    #         if (t >= len_in) and (t%100 == 0):
    #             print(f"t={t}, hidden_norm: {torch.norm(hidden_t).item():.4f}")
        
    #     # Prepare output
    #     if train and self.distribution_type in ["i_quantile"]:
    #         params = torch.concat(dist_params, dim=1)[:, -len_out:, :, :]
    #         params = torch.concat([params, u], dim=1)
    #     else:            
    #         # Prepare output
    #         params = torch.concat(dist_params, dim=1)[:, -len_out:, :, :].squeeze(-2)
    #         print(f"params shape: {params.shape}")
    #         print(f"first value params: {params[0, 0, 0, :]}")
    #         print(f"penultimate value params: {params[0, -2, 0, :]}")
    #         print(f"last value params: {params[0, -1, 0, :]}")
        
    #     return params

    # def _forward_inference(self, input_feat_full, covar_feat_full, len_in, len_out, B, N):
    #     """Inference mode with step-by-step generation"""
    #     history_next = None
    #     h, c = None, None
    #     dist_params = []
        
    #     for t in range(len_in + len_out):
    #         if t < len_in:
    #             # Use ground truth history
    #             history_next = input_feat_full[:, t:t+1, :, 0:1]
    #         # else: use sampled history_next from previous step
            
    #         embed_feat = self.input_embed(history_next)
    #         covar_feat = covar_feat_full[:, t:t+1, :, :]
            
    #         if self.use_ts_id:
    #             id_feat = self.id_feat.unsqueeze(0).expand(B, -1, -1).unsqueeze(1)
    #             encoder_input = torch.cat([embed_feat, covar_feat, id_feat], dim=-1)
    #         else:
    #             encoder_input = torch.cat([embed_feat, covar_feat], dim=-1)
            
    #         # LSTM forward pass
    #         B_cur, _, N_cur, C_cur = encoder_input.shape
    #         encoder_input = encoder_input.transpose(1, 2).reshape(B_cur * N_cur, 1, C_cur)
            
    #         if t == 0:
    #             _, (h, c) = self.encoder(encoder_input)
    #         else:
    #             _, (h, c) = self.encoder(encoder_input, (h, c))
            
    #         # Truncated BPTT for long sequences
    #         if t > 0 and t % 20 == 0:
    #             h = h.detach()
    #             c = c.detach()
            
    #         # Distribution projection
    #         if self.distribution_type in ["i_quantile"]:
    #             resample_u = True if t == 0 else False
    #             head_output = self.prob_head(F.relu(h[-1, :, :].view(B, N, -1)), resample_u=resample_u).reshape(B * N, -1)
    #             dist_params.append(head_output.view(B, N, -1).unsqueeze(1))
    #         else: 
    #             head_output = self.prob_head(F.relu(h[-1, :, :]))
    #             dist_params.append(head_output.view(B, N, -1).unsqueeze(1))
            
    #         # Sampling for prediction phase
    #         if t >= len_in:
    #             sample = self.prob_head.sample(head_output)
    #             history_next = sample.view(B, N).view(B, 1, N, 1)
    #             print(f"t={t}, hidden_norm: {torch.norm(h).item():.4f}")
            
    #     # Add this in your training loop
    #     total_grad_norm = 0
    #     for param in self.parameters():
    #         if param.grad is not None:
    #             total_grad_norm += param.grad.norm().item() ** 2
    #     total_grad_norm = total_grad_norm ** 0.5
    #     print(f"Total gradient norm: {total_grad_norm}")
        
    #     # Prepare output
    #     params = torch.concat(dist_params, dim=1)[:, -len_out:, :, :]
    #     print(f"params shape: {params.shape}")
    #     print(f"first value params: {params[0, 0, 0, :]}")
    #     print(f"penultimate value params: {params[0, -2, 0, :]}")
    #     print(f"last value params: {params[0, -1, 0, :]}")
        
    #     return params
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
            out, hidden = self.encoder(encoder_input) if t == 1 else self.encoder(encoder_input, hidden)
            # print(f"t={t}, hidden_norm: {torch.norm(h).item():.4f}")

            # distribution proj
            if self.distribution_type in ["i_quantile"]: # since we need the batch size shape to construct the uniform quantile levels this is a bit ugly
                resample_u = True if t==1 else False
                head_output = self.prob_head(F.relu(out[:, -1, :].view(B, N, -1)), resample_u=resample_u).reshape(B * N, -1)
                # print(head_output[..., 0].view(B, N, -1).unsqueeze(1).shape)
                # print(head_output[..., 1].view(B, N, -1)[:, 0:1, :].unsqueeze(1).shape)
                if train:
                    u = head_output[..., 1].view(B, N, -1).unsqueeze(1)
                # if t in [1,len_in + len_out-1, len_in + len_out]:
                #     print(u.shape)
                #     print(u.mean())
                #     print(u.max())
                #     print(head_output[..., 0].view(B, N, -1).unsqueeze(1).shape)
                    dist_params.append(head_output[..., 0].view(B, N, -1).unsqueeze(1))
                else: 
                    dist_params.append(head_output.view(B, N, -1).unsqueeze(1))
                # else:
                #     dist_params.append(torch.cat([head_output[..., 0].view(B, N, -1).unsqueeze(1), head_output[..., 1].view(B, N, -1)[:, 0:1, :].unsqueeze(1)], dim=1))
            else: 
                # print(f"t={t}, hidden state shape: {h.shape}")
                # print(f"hidden state values: {h[-1, :, :]}")
                # print(f"t={t}, out shape: {out.shape}")
                # print(f"out values: {out[:, -1, :]}")
                head_output = self.prob_head(F.relu(out[:, -1, :]))
                dist_params.append(head_output.view(B, N, -1).unsqueeze(1))
            # if t in [1, len_in + len_out-2, len_in + len_out-1, len_in + len_out]:
            #     print(f"t={t}, head_output: {head_output.shape}")
            #     print(f"head_output: {head_output.view(B, N, -1)[0, 0, :]}")
            #     print(f"True values: {input_feat_full[0, t-1, 0, 0]}")
            #     print(f"True values: {input_feat_full[0, t, 0, 0]}")
            if (t > len_in and not train): # not in the decoding stage when inferecing
                sample = self.prob_head.sample(head_output) #self._sample_from_head(head_output, None)
                history_next = sample.view(B, N).view(B, 1, N, 1)
            # print(head_output.view(B, N, -1).unsqueeze(1).shape)
            #history_next = self.gaussian_sample(mu, sigma).view(B, N).view(B, 1, N, 1)
            #mus.append(mu.view(B, N, 1).unsqueeze(1))
            #sigmas.append(sigma.view(B, N, 1).unsqueeze(1))
            # samples.append(history_next)
            assert not torch.isnan(history_next).any()

        # samples = torch.concat(samples, dim=1)
        if train and self.distribution_type in ["i_quantile"]:
            params = torch.concat(dist_params, dim=1)[:, -len_out:, :, :]
            params = torch.concat([params, u], dim=1)
        else:
            #TODO also try to return the full prediction horizon and optimize it on that
            params = torch.concat(dist_params, dim=1)[:, -len_out:, :, :]
        #reals = input_feat_full[:, -params.shape[1]:, :, :]
        return params # {"prediction": params, }#"target": reals,}# "mus": mus, "sigmas": sigmas}

    def sample_trajectories_vec(self, history_data, future_data, num_samples=100):
        """
        Vectorized sampling of multiple trajectories from the DeepAR model.
    
        Args:
            history_data (torch.Tensor): Past inputs [B, L_in, N, C]
            future_data (torch.Tensor): Future covariates [B, L_out, N, C]
            num_samples (int): Number of samples to generate
    
        Returns:
            torch.Tensor: Samples of shape [num_samples, B, L_out, N]
        """
        self.eval()
    
        B, L_in, N, _ = history_data.shape
        L_out = future_data.shape[1]
    
        if self.distribution_type in ['i_quantile']:
            # sample batch size x 100 random quantile levels, those will be used throughout the series
            # Sample quantile levels: shape [num_samples, 1]
            quantile_levels = torch.tensor(self.quantiles, device=history_data.device).float()
            q2 = torch.rand(num_samples, device=history_data.device)
            _u = torch.cat([quantile_levels, q2], dim=0)
            num_samples = _u.shape[0]
            
            # Repeat quantiles across batch and series
            # _u = quantiles #.repeat_interleave(B, dim=0)  # [SB, 1]
        elif self.distribution_type in ['quantile']:
            quantile_levels = torch.tensor(self.quantiles, device=history_data.device).float()
            num_samples = quantile_levels.shape[0]
        
        # Repeat data for each sample
        history_data = history_data.unsqueeze(0).repeat(num_samples, 1, 1, 1, 1)  # [S, B, L_in, N, C]
        future_data = future_data.unsqueeze(0).repeat(num_samples, 1, 1, 1, 1)    # [S, B, L_out, N, C]
    
            
        # Merge sample and batch dims
        SB = num_samples * B
    
        input_feat_full = torch.cat([
            history_data[:, :, :, :, 0:1], future_data[:, :, :, :, 0:1]
        ], dim=2).reshape(SB, L_in + L_out, N, 1)
    
        covar_feat_full = torch.cat([
            history_data[:, :, :, :, 1:], future_data[:, :, :, :, 1:]
        ], dim=2).reshape(SB, L_in + L_out, N, -1)
        
        if self.use_ts_id:
            id_feat = self.id_feat.unsqueeze(0).expand(B, -1, -1)  # [B, N, D]
            if self.use_ts_id:
                id_feat = id_feat.unsqueeze(0).repeat(num_samples, 1, 1, 1).reshape(SB, 1, N, -1)  # [SB, 1, N, D]

            
        sample_seq = []
    
        h, c = None, None
        for t in range(1, L_in + L_out):
            if t <= L_in:
                history_next = input_feat_full[:, t - 1:t, :, :]  # [SB, 1, N, 1]
            else:
                history_next = pred.unsqueeze(1).unsqueeze(-1)  # [SB, 1, N, 1]
    
            embed_feat = self.input_embed(history_next)  # [SB, 1, N, D]
            covar_feat = covar_feat_full[:, t:t + 1, :, :]  # [SB, 1, N, C']
    
            if self.use_ts_id:
                encoder_input = torch.cat([embed_feat, covar_feat, id_feat], dim=-1)
            else:
                encoder_input = torch.cat([embed_feat, covar_feat], dim=-1)
    
            SBxN, _, C = encoder_input.transpose(1, 2).reshape(SB * N, -1, encoder_input.shape[-1]).shape
            encoder_input = encoder_input.transpose(1, 2).reshape(SB * N, -1, C)
    
            if t == 1:
                _, (h, c) = self.encoder(encoder_input)
            else:
                _, (h, c) = self.encoder(encoder_input, (h, c))

            if self.distribution_type in ['gaussian', 'student_t', 'laplace']:
                dist_params = self.prob_head(F.relu(h[-1]))  # [SB*N, ?]
                pred = self.prob_head.sample(dist_params).view(SB, N)  # [SB, N]
            elif self.distribution_type in ['i_quantile']:                
                dist_params = self.prob_head(F.relu(h[-1]), _u=_u, batch_size=B)  # Accepts quantile levels
                pred = dist_params.reshape(SB, N)
            elif self.distribution_type == 'quantile':
                head_output = self.prob_head(F.relu(h[-1]))  # [SB*N, output_dim, num_quantiles]
                # head_output = head_output.view(num_samples, B, N, -1)

                pred = torch.stack([
                    self.prob_head.sample(head_output[i], quantile_idx=i)
                    for i in range(num_samples)
                ], dim=0)
                print(pred.shape)
                # .reshape(SB, N)
            
            if t + 1 > L_in: # +1 since t starts at 0, so the first L_in steps are disregarded
                sample_seq.append(pred.view(num_samples, B, 1, N))  # [S, B, 1, N]
    
        samples = torch.cat(sample_seq, dim=2)  # [S, B, L_out, N]
        return samples

    def sample_trajectories(self, history_data: torch.Tensor, future_data: torch.Tensor, num_samples: int, **kwargs) -> torch.Tensor:
        self.eval()
        
        with torch.no_grad():
            len_in, len_out = history_data.shape[1], future_data.shape[1]
            B, _, N, C = history_data.shape
            
            # Prepare full covariate features
            covar_feat_full = torch.cat([history_data[:, :, :, 1:], future_data[:, :, :, 1:]], dim=1)
            
            all_samples = []
            
            if self.distribution_type in ['i_quantile']:
                # sample batch size x 100 random quantile levels, those will be used throughout the series
                # Sample quantile levels: shape [num_samples, 1]
                quantile_levels = torch.tensor(self.quantiles, device=history_data.device).float()
                q2 = torch.rand(num_samples, device=history_data.device)
                _u = torch.cat([quantile_levels, q2], dim=0)
                num_samples = _u.shape[0]

            for sample_idx in range(num_samples):

                sample_trajectory = []
                history_next = None
                h, c = None, None
                
                # Process each timestep
                for t in range(len_in + len_out):
                    if t < len_in:
                        # Use ground truth history
                        history_next = history_data[:, t:t+1, :, 0:1]
                    # else: use history_next from previous iteration
                    
                    # Embed the input
                    embed_feat = self.input_embed(history_next)
                    covar_feat = covar_feat_full[:, t:t+1, :, :]
                    
                    # Add time series ID if used
                    if self.use_ts_id:
                        id_feat = self.id_feat.unsqueeze(0).expand(B, -1, -1).unsqueeze(1)
                        encoder_input = torch.cat([embed_feat, covar_feat, id_feat], dim=-1)
                    else:
                        encoder_input = torch.cat([embed_feat, covar_feat], dim=-1)
                    
                    # Reshape for LSTM: (B, 1, N, C) -> (B*N, 1, C)
                    B_cur, _, N_cur, C_cur = encoder_input.shape
                    encoder_input = encoder_input.transpose(1, 2).reshape(B_cur * N_cur, 1, C_cur)
                    
                    # LSTM forward pass
                    if t == 0:
                        out, hidden = self.encoder(encoder_input)
                    else:
                        out, hidden = self.encoder(encoder_input, hidden)
                    
                    # Generate distribution parameters
                    if self.distribution_type in ["i_quantile"]:
                        # resample_u = True if t == 0 else False
                        
                        # Sample a new quantile level for each prediction step
                        _u_t = torch.rand(1, device=history_data.device)  # [1]
                        head_output = self.prob_head(F.relu(out[:, -1, :].view(B, N, -1)), _u=_u_t).reshape(B * N, -1)
                        # head_output = self.prob_head(F.relu(out[:, -1, :].view(B, N, -1)), _u=_u[sample_idx:sample_idx+1]).reshape(B * N, -1)
                        if t >= len_in:
                            sample = head_output
                            history_next = sample.view(B, 1, N, 1)
                            sample_trajectory.append(history_next)
                    else:
                        head_output = self.prob_head(F.relu(out[:, -1, :]))
                        # For prediction steps, sample from the distribution
                        if t >= len_in:
                            sample = self.prob_head.sample(head_output)
                            history_next = sample.view(B, 1, N, 1)
                            sample_trajectory.append(history_next)
                    
                    assert not torch.isnan(history_next).any()
                
                # Collect this sample's trajectory
                if sample_trajectory:
                    trajectory = torch.cat(sample_trajectory, dim=1)  # [B, L_out, N, 1]
                    all_samples.append(trajectory)
            
            # Stack all samples
            all_samples = torch.stack(all_samples, dim=1)
            return all_samples 
    
    def sample_trajectories_vectorized(self, history_data: torch.Tensor, future_data: torch.Tensor, num_samples: int, **kwargs) -> torch.Tensor:
        """Vectorized version that processes all samples simultaneously for better efficiency.
        
        Args:
            history_data (torch.Tensor): history data. [B, L, N, C].
            future_data (torch.Tensor): future data containing covariates. [B, L, N, C].
            num_samples (int): number of trajectory samples to generate.
            
        Returns:
            torch.Tensor: sampled trajectories. [B, num_samples, L_out, N, 1]
        """
        self.eval()
        
        with torch.no_grad():
            len_in, len_out = history_data.shape[1], future_data.shape[1]
            B, _, N, C = history_data.shape
            
            # Expand all inputs to handle multiple samples
            # Shape: [B * num_samples, L, N, C]
            history_expanded = history_data.repeat(num_samples, 1, 1, 1)
            future_expanded = future_data.repeat(num_samples, 1, 1, 1)
            
            covar_feat_full = torch.cat([history_expanded[:, :, :, 1:], future_expanded[:, :, :, 1:]], dim=1)
            
            # Initialize
            history_next = None
            h, c = None, None
            sample_trajectories = []
            
            # Prepare input features
            input_feat_full = torch.cat([history_expanded[:, :, :, 0:1], 
                                       torch.zeros_like(future_expanded[:, :, :, 0:1])], dim=1)
            
            for t in range(1, len_in + len_out):
                if t <= len_in:
                    history_next = input_feat_full[:, t-1:t, :, 0:1]
                
                # Embed and prepare features
                embed_feat = self.input_embed(history_next)
                covar_feat = covar_feat_full[:, t:t+1, :, :]
                
                if self.use_ts_id:
                    id_feat = self.id_feat.unsqueeze(0).expand(B * num_samples, -1, -1).unsqueeze(1)
                    encoder_input = torch.cat([embed_feat, covar_feat, id_feat], dim=-1)
                else:
                    encoder_input = torch.cat([embed_feat, covar_feat], dim=-1)
                
                # LSTM forward
                B_cur, _, N_cur, C_cur = encoder_input.shape
                encoder_input = encoder_input.transpose(1, 2).reshape(B_cur * N_cur, -1, C_cur)
                
                if t == 1:
                    _, (h, c) = self.encoder(encoder_input)
                else:
                    _, (h, c) = self.encoder(encoder_input, (h, c))
                
                # Generate predictions
                if self.distribution_type in ["i_quantile"]:
                    resample_u = True if t == 1 else False
                    head_output = self.prob_head(F.relu(h[-1, :, :].view(B * num_samples, N, -1)), resample_u=resample_u).reshape(B * num_samples * N, -1)
                else:
                    head_output = self.prob_head(F.relu(h[-1, :, :]))
                
                # Sample for prediction steps
                if t > len_in:
                    sample = self.prob_head.sample(head_output)
                    history_next = sample.view(B * num_samples, N).view(B * num_samples, 1, N, 1)
                    sample_trajectories.append(history_next)
                    
                    # Update input for next step
                    input_feat_full[:, t:t+1, :, 0:1] = history_next
                
                assert not torch.isnan(history_next).any()
            
            # Reshape and return
            if sample_trajectories:
                trajectories = torch.cat(sample_trajectories, dim=1)  # [B * num_samples, L_out, N, 1]
                trajectories = trajectories.view(B, num_samples, len_out, N, 1)
                return trajectories
            
            return torch.zeros(B, num_samples, len_out, N, 1, device=history_data.device)

