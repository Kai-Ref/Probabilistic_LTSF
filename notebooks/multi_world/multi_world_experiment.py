import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random

sns.set(style="whitegrid")


class DataGenerator:
    """Class for generating multi-world time series data."""
    
    def __init__(self, seq_len=100, prefix_len=50, noise_std=0.05, seed=42):
        """
        Initialize the data generator.
        
        Args:
            seq_len: Total sequence length
            prefix_len: Length of the common prefix before branching
            noise_std: Standard deviation of base noise
            seed: Random seed for reproducibility
        """
        self.seq_len = seq_len
        self.prefix_len = prefix_len
        self.noise_std = noise_std
        self.seed = seed
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def generate_prefix(self, seed=None):
        """Generate a common prefix for all trajectories."""
        if seed is not None:
            np.random.seed(seed)
        
        x = np.arange(self.prefix_len)
        base_signal = np.sin(0.1 * x)
        noise = np.random.normal(0, self.noise_std, self.prefix_len)
        return base_signal + noise
    
    def generate_world_branch(self, world_fn, start_val, noise_scale=0.2, smooth_blend=10, seed=None):
        """
        Generate a smooth branch starting from start_val and following world_fn.
        
        Args:
            world_fn: Function defining the trajectory shape
            start_val: Starting value (last value of prefix)
            noise_scale: Scale of random noise
            smooth_blend: Number of points to smoothly blend from prefix
            seed: Random seed
        """
        if seed is not None:
            np.random.seed(seed)
            
        branch_len = self.seq_len - self.prefix_len
        x = np.arange(branch_len)
        target = world_fn(x)
        noise = np.random.normal(0, noise_scale, branch_len)
        target += noise

        # Smooth transition from start_val to world_fn using cosine blend
        transition = np.linspace(0, 1, smooth_blend)
        smooth_start = (1 - transition) * start_val + transition * target[:smooth_blend]
        return np.concatenate([smooth_start, target[smooth_blend:]])
    
    def sample_trajectories(self, n_samples=30, n_worlds=3):
        """
        Generate multiple trajectories across different worlds.
        
        Args:
            n_samples: Total number of samples to generate
            n_worlds: Number of distinct "world" patterns
            
        Returns:
            samples: [n_samples, seq_len] array of trajectories
            world_ids: [n_samples] array of world assignments
        """
        np.random.seed(self.seed)
        prefix = self.generate_prefix(seed=self.seed)
        branch_len = self.seq_len - self.prefix_len

        # Define distinct world behaviors
        world_functions = [
            lambda x: np.sin(0.1 * (x + self.prefix_len)),     # World 0: continues sine
            # lambda x: 0.5 * np.sin(0.2 * (x + self.prefix_len)),  # World 1: faster, smaller sine
            lambda x: -np.cos(0.1 * (x + self.prefix_len)),     # World 2: cosine
            lambda x: np.sin(0.1 * (x + self.prefix_len) + np.pi / 4),  # World 3: phase-shifted sine
            lambda x: np.zeros_like(x),                # World 4: flattens out
        ]

        assert n_worlds <= len(world_functions), "Too many worlds requested"

        # Distribute samples across worlds
        world_sizes = [n_samples // n_worlds] * n_worlds
        for i in range(n_samples % n_worlds):
            world_sizes[i] += 1

        print(world_sizes)

        samples, world_ids = [], []

        for world_id, count in enumerate(world_sizes):
            world_fn = world_functions[world_id]
            for j in range(count):
                branch = self.generate_world_branch(
                    world_fn,
                    start_val=prefix[-1],
                    smooth_blend=10,
                    seed=self.seed + j + world_id * 100
                )
                full_series = np.concatenate([prefix, branch])
                samples.append(full_series)
                world_ids.append(world_id)

        return np.stack(samples), np.array(world_ids)


class ForecastModels:
    """Class containing different time series forecasting models."""
    
    class GaussianForecastModel(nn.Module):
        """Direct Multi-Step (DMS) forecasting model."""
        def __init__(self, input_len, forecast_horizon):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_len, 64),
                nn.ReLU(),
                nn.Linear(64, 2 * forecast_horizon)  # output mu and log_sigma
            )
            self.forecast_horizon = forecast_horizon

        def forward(self, x):
            out = self.net(x)
            mu, log_sigma = out[..., :self.forecast_horizon], out[..., self.forecast_horizon:]
            sigma = torch.exp(log_sigma.clamp(-5, 5))  # stability
            return mu, sigma
    
    class IMSLSTM(nn.Module):
        """Iterative Multi-Step (IMS) forecasting model using LSTM."""
        def __init__(self, input_size=1, hidden_size=64, num_layers=1):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fn = nn.ReLU()
            self.out = nn.Linear(hidden_size, 2)  # mean and log_sigma
            self.device = "cuda" if torch.cuda.is_available() else 'cpu'
            
        def forward(self, x, hidden=None):
            # x: [batch_size, seq_len, input_size]
            lstm_out, hidden = self.lstm(x, hidden)  # hidden: (h_n, c_n)
            last_out = lstm_out[:, -1, :]  # take output from last timestep
            last_out = self.fn(last_out)
            out = self.out(last_out)
            mu, log_sigma = out[:, 0], out[:, 1]
            sigma = torch.exp(log_sigma.clamp(-5, 5))
            return mu, sigma, hidden
            
        def forecast(self, initial_sequence, steps, use_mean=False, num_samples=10, random_state=42):
            """
            Generate multi-step forecast starting from initial_sequence.
            
            Args:
                initial_sequence: [batch_size, seq_len, input_size] initial input sequence
                steps: number of steps to forecast
                use_mean: if True, use predicted mean for next step; otherwise sample from distribution
                
            Returns:
                forecasts: [batch_size, steps] tensor of forecasted values
                sigmas: [batch_size, steps] tensor of predicted standard deviations
            """
            self.eval()
            with torch.no_grad():
                batch_size = initial_sequence.shape[0]
                
                num_samples = num_samples if not use_mean else 1
                forecasts = torch.zeros(num_samples, batch_size, steps)
                sigmas = torch.zeros(num_samples, batch_size, steps)
                for sample in range(num_samples):
                    # Start with the initial sequence
                    current_input = initial_sequence.to(self.device)
                    hidden = None
                    for t in range(steps):
                        # Get prediction for current step
                        mu, sigma, hidden = self(current_input, hidden)
                        
                        # Store predictions
                        forecasts[sample, :, t] = mu
                        sigmas[sample, :, t] = sigma
                        
                        # Generate next input by either sampling or using mean
                        if use_mean:
                            next_val = mu
                        else:
                            # Sample from normal distribution
                            # g = torch.Generator().manual_seed(random_state).to(mu.device)  # Replace 42 with your desired seed
                            next_val = torch.normal(mu, sigma)#, generator=g)
                        
                        # Update input for next prediction (remove oldest, add newest)
                        # Shape of next_val: [batch_size], need to reshape to [batch_size, 1, 1]
                        next_input = next_val.view(batch_size, 1, 1)
                        current_input = torch.cat([current_input[:, 1:, :], next_input], dim=1)
                    
                return forecasts.squeeze(0), sigmas.squeeze(0)
    
    @staticmethod
    def build_dms_dataset(samples, prefix_len, forecast_horizon):
        """
        Build training data for DMS model.
        
        Args:
            samples: [n_samples, seq_len] array of trajectories
            prefix_len: Length of input sequence
            forecast_horizon: Length of output sequence
            
        Returns:
            X_train: [n_samples, prefix_len] tensor
            Y_train: [n_samples, forecast_horizon] tensor
        """
        X = samples[:, :prefix_len]
        Y = samples[:, prefix_len:prefix_len + forecast_horizon]
        X_train = torch.tensor(X, dtype=torch.float32)
        Y_train = torch.tensor(Y, dtype=torch.float32)
        return X_train, Y_train
    
    @staticmethod
    def build_ims_dataset(sequences, input_len):
        """
        Build training data for IMS model.
        
        Args:
            sequences: [num_series, seq_len] or [num_samples, seq_len]
            input_len: number of steps to use as input
            
        Returns:
            X: [n_samples, input_len]
            Y: [n_samples]
        """
        X, Y = [], []
        for seq in sequences:
            for t in range(len(seq) - input_len):
                X.append(seq[t:t+input_len])
                Y.append(seq[t+input_len])
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
    
    @staticmethod
    def train_dms(model, X_train, Y_train, epochs=1000, lr=1e-3):
        """
        Train a DMS forecast model.
        
        Args:
            model: GaussianForecastModel instance
            X_train: [n_samples, prefix_len] tensor
            Y_train: [n_samples, forecast_horizon] tensor
            epochs: Number of training epochs
            lr: Learning rate
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in tqdm(range(epochs), total=epochs):
            model.train()
            optimizer.zero_grad()
            mu, sigma = model(X_train)
            loss = (0.5 * torch.log(2 * torch.pi * sigma ** 2) + ((Y_train - mu) ** 2) / (2 * sigma ** 2)).mean()
            loss.backward()
            optimizer.step()
            if epoch % 200 == 0:
                print(f"Epoch {epoch}: NLL = {loss.item():.4f}")
    
    @staticmethod
    def train_ims(model, X_train, Y_train, epochs=500, lr=1e-3):
        """
        Train an IMS LSTM forecast model.
        
        Args:
            model: IMSLSTM instance
            X_train: [n_samples, input_len] tensor
            Y_train: [n_samples] tensor
            epochs: Number of training epochs
            lr: Learning rate
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in tqdm(range(epochs), total=epochs):
            model.train()
            optimizer.zero_grad()
            
            x_seq = X_train.unsqueeze(-1)  # [batch, seq_len, 1]
            y_true = Y_train
            mu, sigma, _ = model(x_seq)  # mu, sigma: [batch]
            
            # Gaussian NLL loss
            loss = ((y_true - mu) ** 2) / (2 * sigma**2) + 0.5 * torch.log(2 * torch.pi * sigma**2)
            loss = loss.mean()
            
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(f"[IMS] Epoch {epoch} | NLL: {loss.item():.4f}")

    @staticmethod
    def train_ims_with_teacher_forcing(model, sequences, input_len, forecast_horizon,
                                  epochs=500, lr=1e-3, batch_size=64):
        """
        Train an IMS LSTM model with teacher forcing (optimized version).
        
        Args:
            model: IMSLSTM instance
            sequences: [num_series, seq_len] list/array of time series
            input_len: length of input sequence
            forecast_horizon: number of steps to forecast during training
            epochs: number of training epochs
            lr: learning rate
            batch_size: size of mini-batches for training
        """
        from torch.utils.data import TensorDataset, DataLoader
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        device = next(model.parameters()).device
        print(f'using device {device}')
        
        # Pre-process all data at once
        print("Preparing training data...")
        inputs, targets = [], []
        print(sequences.shape)
        for seq in tqdm(sequences):
            for t in range(len(seq) - input_len - forecast_horizon):
                # Extract input sequence and target sequence
                input_seq = seq[t:t+input_len]
                target_seq = seq[t+input_len:t+input_len+forecast_horizon]
                inputs.append(input_seq)
                targets.append(target_seq)
        
        # Convert to tensors once
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(-1)  # [n_samples, input_len, 1]
        targets = torch.tensor(targets, dtype=torch.float32)  # [n_samples, forecast_horizon]
        
        # Create dataset and dataloader for efficient batching
        dataset = TensorDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                                pin_memory=True if torch.cuda.is_available() else False)
        
        print(f"Training with {len(dataset)} samples in batches of {batch_size}")
        
        for epoch in tqdm(range(epochs), total=epochs):
            model.train()
            epoch_loss = 0
            
            for batch_inputs, batch_targets in dataloader:
                # Move batch to device
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)
                
                optimizer.zero_grad()
                
                batch_size = batch_inputs.size(0)
                hidden = None
                
                # Initial input is the given sequence
                current_input = batch_inputs
                
                # Store all step losses for batch loss calculation
                step_losses = []
                
                # Iteratively predict each step in the forecast horizon
                for t in range(forecast_horizon):
                    # Forward pass
                    mu, sigma, hidden = model(current_input, hidden)
                    
                    # Calculate loss for this step (vectorized)
                    step_loss = 0.5 * torch.log(2 * torch.pi * sigma**2) + ((batch_targets[:, t] - mu)**2) / (2 * sigma**2)
                    step_losses.append(step_loss)
                    
                    if t < forecast_horizon - 1:
                        # Use ground truth as next input (teacher forcing)
                        next_val = batch_targets[:, t].view(batch_size, 1, 1)
                        
                        # Update input for next prediction (vectorized operation)
                        current_input = torch.cat([current_input[:, 1:, :], next_val], dim=1)
                
                # Combine all step losses and take mean
                loss = torch.stack(step_losses).mean()
                
                # Backpropagation
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * batch_size
            
            # Print progress
            if epoch % int(epochs/5) == 0:
                print(f"Epoch {epoch} | Avg NLL: {epoch_loss/len(dataset):.4f}")

    
    @staticmethod
    def autoregressive_lstm_forecast(model, prefix, forecast_horizon):
        """
        Performs autoregressive multi-step forecasting using LSTM hidden states.
        
        Args:
            model: trained IMSLSTM model
            prefix: [seq_len] torch.Tensor, 1D
            forecast_horizon: int
            
        Returns:
            predicted_means: [forecast_horizon] np.array
            predicted_stds: [forecast_horizon] np.array
        """
        model.eval()
        with torch.no_grad():
            # Initialize with prefix data
            input_seq = prefix.unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
            
            # Run the model on the prefix to get the initial hidden state
            _, _, hidden = model(input_seq, None)
            
            # Store the last value of the prefix as the first input for forecasting
            last_value = input_seq[:, -1:, :]  # Shape [1, 1, 1]
            
            preds_mu, preds_sigma = [], []
            current_input = last_value
            
            for _ in range(forecast_horizon):
                # Use only the last predicted value as input, but keep hidden state
                mu, sigma, hidden = model(current_input, hidden)
                preds_mu.append(mu.item())
                preds_sigma.append(sigma.item())
                
                # Update input for next step with the prediction
                current_input = mu.view(1, 1, 1)  # shape [1, 1, 1]
                
        return np.array(preds_mu), np.array(preds_sigma)


class Metrics:
    """Class for computing various metrics for time series forecasts."""
    
    @staticmethod
    def nll_gaussian(y_true, mu, sigma):
        """
        Compute Gaussian negative log-likelihood.
        
        Args:
            y_true: Ground truth values
            mu: Predicted means
            sigma: Predicted standard deviations
            
        Returns:
            NLL values
        """
        return 0.5 * np.log(2 * np.pi * sigma**2) + ((y_true - mu) ** 2) / (2 * sigma**2)
    
    @staticmethod
    def gaussian_entropy(sigma):
        """
        Compute entropy of Gaussian distribution.
        
        Args:
            sigma: Standard deviations
            
        Returns:
            Entropy values
        """
        return 0.5 * np.log(2 * np.pi * np.e * sigma**2)
    
    @staticmethod
    def estimate_ground_truth_entropy(samples, world_ids=None, prefix_len=50, forecast_horizon=50):
        """
        Estimate ground-truth entropy at each future time step from multi-world samples.
        Assumes Gaussian marginals.

        Args:
            samples: [n_samples, total_seq_len]
            world_ids: [n_samples] array indicating which world each sample belongs to
            prefix_len: number of observed time steps
            forecast_horizon: number of future steps to analyze

        Returns:
            entropies: [forecast_horizon] array of entropy values
        """
        if world_ids is not None:
            n_worlds = len(np.unique(world_ids))
            world_prob = 1.0 / n_worlds  # Equal probability for each world
            
            forecast_slice = samples[:, prefix_len:prefix_len+forecast_horizon]  # [n_samples, horizon]
            entropies = np.zeros(forecast_horizon)
            
            # Between-world entropy component (discrete choice between worlds)
            between_world_entropy = -n_worlds * (world_prob * np.log(world_prob))
            for t in range(forecast_horizon):
                # Within-world entropy component (variance within each world)
                within_world_entropy = 0
                for world_id in np.unique(world_ids):
                    world_mask = world_ids == world_id
                    world_samples = forecast_slice[world_mask, t]
                    world_std = np.std(world_samples)
                    within_world_entropy += world_prob * Metrics.gaussian_entropy(world_std)
                
                # Total entropy is the sum of between-world and within-world components
                entropies[t] = between_world_entropy + within_world_entropy
            
            return entropies
        else:
            forecast_slice = samples[:, prefix_len:prefix_len+forecast_horizon]  # [n_samples, horizon]
            std_per_timestep = np.std(forecast_slice, axis=0)  # [forecast_horizon]
            return Metrics.gaussian_entropy(std_per_timestep)
    
    @staticmethod
    def compute_sample_nll(sample, samples, world_ids, prefix_len=50, forecast_horizon=50):
        """
        Compute negative log-likelihood for a specific sample against the ground truth distribution.
        
        Args:
            sample: [seq_len] single time series sample to evaluate
            samples: [n_samples, seq_len] all ground truth samples
            world_ids: [n_samples] array indicating which world each sample belongs to
            prefix_len: number of observed time steps
            forecast_horizon: number of future steps to analyze
            
        Returns:
            nll: [forecast_horizon] array of negative log-likelihood values at each forecast step
        """
        n_worlds = len(np.unique(world_ids))
        world_prob = 1.0 / n_worlds  # Equal probability for each world
        
        # Extract forecast region
        forecast_slice = samples[:, prefix_len:prefix_len+forecast_horizon]  # [n_samples, horizon]
        sample_forecast = sample[prefix_len:prefix_len+forecast_horizon]  # [horizon]
        
        # Prepare output
        nll = np.zeros(forecast_horizon)
        
        for t in range(forecast_horizon):
            # Calculate likelihood of sample_forecast[t] in each world
            world_likelihoods = []
            
            for world_id in np.unique(world_ids):
                world_mask = world_ids == world_id
                world_samples = forecast_slice[world_mask, t]
                
                # Assuming Gaussian distribution within each world
                world_mean = np.mean(world_samples)
                world_var = np.var(world_samples)
                
                # Avoid numerical issues
                if world_var == 0:
                    world_var = 1e-10
                    
                # Calculate likelihood using Gaussian PDF
                diff = sample_forecast[t] - world_mean
                log_likelihood = -0.5 * np.log(2 * np.pi * world_var) - 0.5 * (diff**2 / world_var)
                world_likelihoods.append(log_likelihood)
            
            # Weighted sum of likelihoods (mixture model)
            # p(x) = Σ p(x|world_i) * p(world_i)
            log_world_prob = np.log(world_prob)
            mixed_likelihood = np.log(np.sum(np.exp(np.array(world_likelihoods) + log_world_prob)))
            
            # Compute negative log-likelihood
            nll[t] = -mixed_likelihood
        return nll


class Visualization:
    """Class for visualizing time series forecasts."""
    
    @staticmethod
    def plot_multi_world_distribution(
        samples,
        world_ids=None,
        prefix_len=50,
        forecast_horizon=50,
        show_mean=True,
        show_std=True,
        title="Multi-World Ground Truth Distribution",
    ):
        """
        Visualizes sampled multi-world time series.
        
        Args:
            samples (np.ndarray): [n_samples, seq_len]
            world_ids: [n_samples] array indicating which world each sample belongs to
            prefix_len: number of observed time steps
            forecast_horizon: number of future steps to analyze
            show_mean (bool): plot mean of trajectories
            show_std (bool): show ±1 std deviation band
            title (str): plot title
        """
        n_samples, seq_len = samples.shape
        branching_point = prefix_len
        total_len = prefix_len + forecast_horizon
        x = np.arange(total_len)

        plt.figure(figsize=(12, 6))
        if world_ids is not None:        
            # Get unique worlds and assign colors
            unique_worlds = np.unique(world_ids)
            n_worlds = len(unique_worlds)
            colors = plt.cm.tab10(np.linspace(0, 1, n_worlds))
            
            # Plot prefix samples (should be the same for all)
            for i in range(n_samples):
                plt.plot(x[:branching_point], samples[i, :branching_point], 
                        color="blue", alpha=0.1)
            
            # Plot each world with different colors
            for w, world_id in enumerate(unique_worlds):
                world_mask = world_ids == world_id
                world_samples = samples[world_mask]
                
                # Plot individual trajectories with world-specific color
                for i in range(world_samples.shape[0]):
                    plt.plot(x[branching_point:total_len], 
                            world_samples[i, branching_point:total_len], 
                            color=colors[w], alpha=0.2)
                
                # World-specific mean and std
                if show_mean or show_std:
                    world_mean = world_samples[:, branching_point:total_len].mean(axis=0)
                    
                    if show_mean:
                        plt.plot(x[branching_point:total_len], world_mean, 
                                color=colors[w], linewidth=2, 
                                label=f"World {world_id} Mean")
                    
                    if show_std:
                        world_std = world_samples[:, branching_point:total_len].std(axis=0)
                        plt.fill_between(
                            x[branching_point:total_len], 
                            world_mean - 2*world_std, 
                            world_mean + 2*world_std, 
                            color=colors[w], alpha=0.1
                        )
        else:    
            # Plot all sample trajectories
            for i in range(n_samples):
                plt.plot(x, samples[i, :total_len], color="blue", alpha=0.03)
        
            # Branching line
            plt.axvline(branching_point, color="black", linestyle="--", label="Branching Point")
        
            # Mean and std band
            if show_mean or show_std:
                mean = samples[:, :total_len].mean(axis=0)
                std = samples[:, :total_len].std(axis=0)
        
                if show_std:
                    plt.fill_between(x, mean - 2*std, mean + 2*std, color="orange", alpha=0.2, label="±2 Std Dev")
        
                if show_mean:
                    plt.plot(x, mean, color="orange", label="Mean Trajectory", linewidth=2)
                
        plt.axvline(branching_point, color="black", linestyle="--", label="Branching Point")
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_forecast(x_prefix, y_true, mu, sigma, label="Model"):
        """
        Plot forecast with uncertainty.
        
        Args:
            x_prefix: Input prefix values
            y_true: Ground truth future values
            mu: Predicted means
            sigma: Predicted standard deviations
            label: Model name for legend
        """
        plt.figure(figsize=(12, 6))
        
        # Plot prefix
        prefix_len = len(x_prefix)
        forecast_horizon = len(mu)
        x = np.arange(prefix_len + forecast_horizon)
        
        # Plot prefix
        plt.plot(x[:prefix_len], x_prefix, color='blue', label='Prefix')
        
        # Plot ground truth
        plt.plot(x[prefix_len:], y_true, color='black', label='Ground Truth')
        
        # Plot forecast with uncertainty
        plt.plot(x[prefix_len:], mu, color='red', label=f'{label} Forecast')
        plt.fill_between(
            x[prefix_len:],
            mu - 2 * sigma,
            mu + 2 * sigma,
            color='red',
            alpha=0.2,
            label=f'{label} ±2σ'
        )
        
        # Add branching point line
        plt.axvline(prefix_len, color='black', linestyle='--', label='Branching Point')
        
        plt.title(f"{label} Forecast with Uncertainty")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_ims_samples(x_prefix, y_true, samples, label="Model"):
        """
        Plot sample trajectories.
        
        Args:
            x_prefix: Input prefix values
            y_true: Ground truth future values
            samples: Predicted samples
            sigma: Predicted standard deviations
            label: Model name for legend
        """
        plt.figure(figsize=(12, 6))
        
        # Plot prefix
        prefix_len = len(x_prefix)
        forecast_horizon = len(y_true)
        x = np.arange(prefix_len + forecast_horizon)
        
        # Plot prefix
        plt.plot(x[:prefix_len], x_prefix, color='blue', label='Prefix')
        
        # Plot ground truth
        plt.plot(x[prefix_len:], y_true, color='black', label='Ground Truth')
        
        # Plot forecast with uncertainty
        alpha = max(0.1, 1.0/samples.shape[0])
        for i in range(samples.shape[0]):
            plt.plot(x[prefix_len:], samples[i, 0, :], color='red', alpha=alpha)
        
        # Add branching point line
        plt.axvline(prefix_len, color='black', linestyle='--', label='Branching Point')
        
        plt.title(f"{label} Sample Forecast with Uncertainty")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_entropy_comparison(entropy_data, plot_styles=None):
        """
        Plot entropy comparison between different models.
        
        Args:
            entropy_data: Dictionary mapping model names to entropy arrays
            plot_styles: Dictionary mapping model names to plot styles
        """
        if plot_styles is None:
            plot_styles = {
                "IMS (LSTM)": {"color": "blue", "linestyle": "-"},
                "DMS": {"color": "green", "linestyle": "--"},
                "Ground Truth": {"color": "red", "linestyle": ":"}
            }
            
        plt.figure(figsize=(10, 5))
        for label, entropy_vals in entropy_data.items():
            style = plot_styles.get(label, {})
            plt.plot(entropy_vals, label=label, **style)
            
        plt.title("Entropy Over Forecast Horizon")
        plt.xlabel("Forecast Step")
        plt.ylabel("Entropy (nats)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class MultiWorldExperiment:
    """
    Class for running forecasting experiments with multi-world time series data.
    """
    
    def __init__(self, seq_len=100, prefix_len=50, forecast_horizon=50, seed=42):
        """
        Initialize the experiment.
        
        Args:
            seq_len: Total sequence length
            prefix_len: Length of common prefix before branching
            forecast_horizon: Number of steps to forecast
            seed: Random seed for reproducibility
        """
        self.seq_len = seq_len
        self.prefix_len = prefix_len
        self.forecast_horizon = forecast_horizon
        self.seed = seed
        
        # Initialize data generator
        self.data_gen = DataGenerator(seq_len=seq_len, prefix_len=prefix_len, seed=seed)
        
        # Initialize containers for data and models
        self.samples = None
        self.world_ids = None
        self.model_dms = None
        self.model_ims = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def generate_data(self, n_samples=30, n_worlds=3):
        """
        Generate multi-world time series data.
        
        Args:
            n_samples: Number of trajectories to generate
            n_worlds: Number of distinct worlds
            
        Returns:
            self: For method chaining
        """
        self.samples, self.world_ids = self.data_gen.sample_trajectories(
            n_samples=n_samples, 
            n_worlds=n_worlds
        )
        return self
    
    def train_models(self, dms_epochs=1000, ims_epochs=500):
        """
        Train DMS and IMS forecasting models on the generated data.
        
        Args:
            dms_epochs: Number of training epochs for DMS model
            ims_epochs: Number of training epochs for IMS model
            
        Returns:
            self: For method chaining
        """
        # Ensure data is generated
        if self.samples is None:
            raise ValueError("Data not generated. Call generate_data() first.")
        
        # Prepare data for DMS model
        X_dms, Y_dms = ForecastModels.build_dms_dataset(
            self.samples, 
            self.prefix_len, 
            self.forecast_horizon
        )
        
        # Create and train DMS model
        self.model_dms = ForecastModels.GaussianForecastModel(
            input_len=self.prefix_len, 
            forecast_horizon=self.forecast_horizon
        )
        ForecastModels.train_dms(self.model_dms, X_dms, Y_dms, epochs=dms_epochs)
        
        # Prepare data for IMS model
        # X_ims, Y_ims = ForecastModels.build_ims_dataset(self.samples, self.prefix_len)
        
        # Create and train IMS model
        self.model_ims = ForecastModels.IMSLSTM(hidden_size=64).to(self.device)
        print(self.forecast_horizon)
        ForecastModels.train_ims_with_teacher_forcing(self.model_ims, self.samples, self.prefix_len, self.forecast_horizon, 
                                  epochs=ims_epochs, lr=1e-3)
        # self.model_ims = self.model_ims.to("cpu")
        # ForecastModels.train_ims(self.model_ims, X_ims, Y_ims, epochs=ims_epochs)
        
        return self
    
    def evaluate_models(self, test_idx=None, num_samples_ims=10):
        """
        Evaluate trained models on a test sample.
        
        Args:
            test_idx: Index of test sample (randomly chosen if None)
            
        Returns:
            results: Dictionary of evaluation results
        """
        # Ensure models are trained
        if self.model_dms is None or self.model_ims is None:
            raise ValueError("Models not trained. Call train_models() first.")
        
        # Choose test sample
        if test_idx is None:
            test_idx = np.random.randint(0, len(self.samples))
        
        # Extract test data
        x_test = torch.tensor(self.samples[test_idx, :self.prefix_len], dtype=torch.float32)
        y_true = self.samples[test_idx, self.prefix_len:self.prefix_len + self.forecast_horizon]

        # Get IMS forecast
        x_test = x_test.unsqueeze(0).unsqueeze(-1)
        mu_ims, sigma_ims = self.model_ims.forecast(x_test, self.forecast_horizon, use_mean=True)
        samples_ims, _ = self.model_ims.forecast(x_test, self.forecast_horizon, use_mean=False, num_samples=num_samples_ims)
        x_test = x_test.squeeze(0).squeeze(-1)
        mu_ims, sigma_ims = mu_ims.squeeze().detach().numpy(), sigma_ims.squeeze().detach().numpy()
        
        #autoregressive_lstm_forecast(
        #    self.model_ims, x_test, self.forecast_horizon
        #)
        
        # Get DMS forecast
        with torch.no_grad():
            mu_dms, sigma_dms = self.model_dms(x_test)
            mu_dms, sigma_dms = mu_dms.numpy(), sigma_dms.numpy()
        # Compute metrics
        ground_truth_entropy = Metrics.estimate_ground_truth_entropy(
            self.samples, 
            self.world_ids, 
            self.prefix_len, 
            self.forecast_horizon
        )
        
        nll_gt = Metrics.compute_sample_nll(
            self.samples[test_idx], 
            self.samples, 
            self.world_ids, 
            self.prefix_len, 
            self.forecast_horizon
        )
        
        nll_dms = Metrics.nll_gaussian(y_true, mu_dms, sigma_dms)
        nll_ims = Metrics.nll_gaussian(y_true, mu_ims, sigma_ims)
        
        entropy_dms = Metrics.gaussian_entropy(sigma_dms)
        entropy_ims = Metrics.gaussian_entropy(sigma_ims)
        
        # Compile results
        results = {
            "test_idx": test_idx,
            "x_test": x_test,
            "y_true": y_true,
            "dms": {
                "mu": mu_dms,
                "sigma": sigma_dms,
                "nll": nll_dms,
                "nll_mean": nll_dms.mean(),
                "entropy": entropy_dms,
                "entropy_mean": entropy_dms.mean()
            },
            "ims": {
                "mu": mu_ims,
                "samples": samples_ims,
                "sigma": sigma_ims,
                "nll": nll_ims,
                "nll_mean": nll_ims.mean(),
                "entropy": entropy_ims,
                "entropy_mean": entropy_ims.mean()
            },
            "ground_truth": {
                "nll": nll_gt,
                "nll_mean": nll_gt.mean(),
                "entropy": ground_truth_entropy,
                "entropy_mean": ground_truth_entropy.mean(),
                "samples":self.samples,
            }
        }
        
        return results
    
    def print_evaluation_summary(self, results):
        """
        Print a summary of evaluation results.
        
        Args:
            results: Dictionary of evaluation results from evaluate_models()
        """
        print("\n=== Model Evaluation Summary ===")
        print(f"Test sample index: {results['test_idx']}")
        print("\nNegative Log-Likelihood (NLL) - lower is better:")
        for name in ["DMS", "IMS", "Ground Truth"]:
            key = name.lower().replace(" ", "_")
            if name == "Ground Truth":
                nll = results["ground_truth"]["nll_mean"]
            else:
                nll = results[key.split("_")[0]]["nll_mean"]
            print(f"  {name:12s}: {nll:.3f}")
        
        print("\nEntropy - measures uncertainty:")
        for name in ["DMS", "IMS", "Ground Truth"]:
            key = name.lower().replace(" ", "_")
            if name == "Ground Truth":
                entropy = results["ground_truth"]["entropy_mean"]
            else:
                entropy = results[key.split("_")[0]]["entropy_mean"]
            print(f"  {name:12s}: {entropy:.3f}")
    
    def visualize_results(self, results):
        """
        Visualize evaluation results.
        
        Args:
            results: Dictionary of evaluation results from evaluate_models()
        """
        # Plot forecasts
        Visualization.plot_forecast(
            results["x_test"].numpy(), 
            results["y_true"], 
            results["dms"]["mu"], 
            results["dms"]["sigma"], 
            label="DMS"
        )
        
        Visualization.plot_forecast(
            results["x_test"].numpy(), 
            results["y_true"], 
            results["ims"]["mu"], 
            results["ims"]["sigma"], 
            label="IMS (LSTM)"
        )
        Visualization.plot_ims_samples(
            results["x_test"].numpy(), 
            results["y_true"], 
            results["ims"]["samples"],
            label="IMS (LSTM)"
        )
        
        # Plot ground truth distribution
        Visualization.plot_multi_world_distribution(
            self.samples, 
            world_ids=self.world_ids,
            prefix_len=self.prefix_len,
            forecast_horizon=self.forecast_horizon
        )
        
        # Plot entropy comparison
        entropy_data = {
            "IMS (LSTM)": results["ims"]["entropy"],
            "DMS": results["dms"]["entropy"],
            "Ground Truth": results["ground_truth"]["entropy"]
        }
        
        plot_styles = {
            "IMS (LSTM)": {"color": "blue", "linestyle": "-"},
            "DMS": {"color": "green", "linestyle": "--"},
            "Ground Truth": {"color": "red", "linestyle": ":"}
        }
        
        Visualization.plot_entropy_comparison(entropy_data, plot_styles)
    
    def run_experiment(self, n_samples=30, n_worlds=3, test_idx=None, dms_epochs=1000, ims_epochs=500, num_samples_ims=10):
        """
        Run a complete experiment: generate data, train models, evaluate, and visualize.
        
        Args:
            n_samples: Number of trajectories to generate
            n_worlds: Number of distinct worlds
            test_idx: Index of test sample (randomly chosen if None)
            dms_epochs: Number of training epochs for DMS model
            ims_epochs: Number of training epochs for IMS model
            
        Returns:
            results: Dictionary of evaluation results
        """
        print(f"=== Starting Multi-World Forecasting Experiment ===")
        print(f"Settings: {n_samples} samples, {n_worlds} worlds, prefix_len={self.prefix_len}, horizon={self.forecast_horizon}")
        
        # Generate data
        print("\nGenerating multi-world time series data...")
        self.generate_data(n_samples=n_samples, n_worlds=n_worlds)
        
        # Train models
        print("\nTraining forecasting models...")
        self.train_models(dms_epochs=dms_epochs, ims_epochs=ims_epochs)
        
        # Evaluate models
        print("\nEvaluating models...")
        results = self.evaluate_models(test_idx=test_idx, num_samples_ims=num_samples_ims)
        
        # Print summary
        self.print_evaluation_summary(results)
        
        # Visualize results
        print("\nGenerating visualizations...")
        self.visualize_results(results)
        
        return results