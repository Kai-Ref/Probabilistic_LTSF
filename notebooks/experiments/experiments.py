import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sns.set(style="whitegrid")
torch.manual_seed(42)
np.random.seed(42)



# DATA GENERATION
def generate_single_world_series(seq_len=100, noise_std=0.05):
    x = np.arange(seq_len)
    base_signal = np.sin(0.1 * x)
    noise = np.random.normal(0, noise_std, seq_len)
    return base_signal + noise

def generate_multi_world_series(seq_len=100, branching_point=50, seed=None):
    if seed is not None:
        np.random.seed(seed)
    x = np.arange(seq_len)
    base_signal = np.sin(0.1 * x)
    series = base_signal.copy()
    noise = np.zeros_like(series)
    noise[branching_point:] = np.cumsum(np.random.normal(0, 0.2, seq_len - branching_point))
    return series + noise

def sample_multi_trajectories(n_samples=20, seq_len=100, branching_point=50):
    return np.stack([
        generate_multi_world_series(seq_len, branching_point, seed=i)
        for i in range(n_samples)
    ])

def generate_base_prefix(seq_len, noise_std=0.05, seed=None):
    if seed is not None:
        np.random.seed(seed)
    x = np.arange(seq_len)
    base_signal = np.sin(0.1 * x)
    noise = np.random.normal(0, noise_std, seq_len)
    return base_signal + noise

def generate_world_branch(seq_len, world_fn, start_val, noise_scale=0.2, smooth_blend=10, seed=None):
    """
    Generate a smooth branch starting from start_val and following world_fn.
    """
    if seed is not None:
        np.random.seed(seed)
    x = np.arange(seq_len)
    target = world_fn(x)
    noise = np.random.normal(0, noise_scale, seq_len)
    target += noise

    # Smooth transition from start_val to world_fn using cosine blend
    transition = np.linspace(0, 1, smooth_blend)
    smooth_start = (1 - transition) * start_val + transition * target[:smooth_blend]
    return np.concatenate([smooth_start, target[smooth_blend:]])

def sample_multi_trajectories_with_worlds(n_samples=30, seq_len=100, branching_point=50, n_worlds=3, seed=42):
    """
    Each trajectory shares the same prefix, then diverges smoothly into different 'worlds'.
    """
    np.random.seed(seed)
    prefix = generate_base_prefix(branching_point, seed=seed)
    postfix_len = seq_len - branching_point

    # Define distinct world behaviors
    world_functions = [
        lambda x: np.sin(0.1 * x),                 # World 0: continues sine
        lambda x: 0.5 * np.sin(0.2 * x),           # World 3: faster, smaller sine
        lambda x: np.cos(0.1 * x),                 # World 1: cosine
        lambda x: np.sin(0.1 * x + np.pi / 4),     # World 2: phase-shifted sine
        lambda x: np.zeros_like(x),                # World 4: flattens out
    ]

    assert n_worlds <= len(world_functions), "Too many worlds requested"

    world_sizes = [n_samples // n_worlds] * n_worlds
    for i in range(n_samples % n_worlds):
        world_sizes[i] += 1

    samples, world_ids = [], []

    for world_id, count in enumerate(world_sizes):
        world_fn = world_functions[world_id]
        for j in range(count):
            postfix = generate_world_branch(
                postfix_len,
                world_fn,
                start_val=prefix[-1],
                smooth_blend=10,
                seed=seed + j + world_id * 100
            )
            full_series = np.concatenate([prefix, postfix])
            samples.append(full_series)
            world_ids.append(world_id)

    return np.stack(samples), np.array(world_ids)
    
def plot_colored_worlds(samples, world_ids, branching_point=50):
    plt.figure(figsize=(12, 6))
    x = np.arange(samples.shape[1])
    colors = plt.cm.tab10(world_ids % 10)

    for i in range(samples.shape[0]):
        plt.plot(x, samples[i], color=colors[i], alpha=0.4)

    plt.axvline(branching_point, color='black', linestyle='--', label='Branching Point')
    plt.title("Multi-World Trajectories (Same Prefix, Diverging Futures)")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# DMS MODEL: SIMPLE MLP
class GaussianForecastModel(nn.Module):
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

def train_dms(model, X_train, Y_train, epochs=1000, lr=1e-3):
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



class IMSLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fn = nn.ReLU()
        self.out = nn.Linear(hidden_size, 2)  # mean and log_sigma
        
    def forward(self, x, hidden=None):
        # x: [batch_size, seq_len, input_size]
        lstm_out, hidden = self.lstm(x, hidden)  # hidden: (h_n, c_n)
        last_out = lstm_out[:, -1, :]  # take output from last timestep
        last_out = self.fn(last_out)
        out = self.out(last_out)
        mu, log_sigma = out[:, 0], out[:, 1]
        sigma = torch.exp(log_sigma.clamp(-5, 5))
        return mu, sigma, hidden

def train_ims(model, X_train, Y_train, epochs=500, lr=1e-3):
    """
    Trains the IMS LSTM model using Gaussian NLL loss.
    Args:
        model: nn.Module
        X_train: [batch_size, input_len]
        Y_train: [batch_size]
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
        # This ensures all prefix information is captured in the hidden state
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



# LOSSES
def nll_gaussian(y_true, mu, sigma):
    return 0.5 * np.log(2 * np.pi * sigma**2) + ((y_true - mu) ** 2) / (2 * sigma**2)

def gaussian_entropy(sigma):
    return 0.5 * np.log(2 * np.pi * np.e * sigma**2)

    
def estimate_ground_truth_entropy(samples, world_ids=None, prefix_len=50, forecast_horizon=50):
    """
    Estimate ground-truth entropy at each future time step from multi-world samples.
    Assumes Gaussian marginals.

    Args:
        samples: [n_samples, total_seq_len]
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
                within_world_entropy += world_prob * gaussian_entropy(world_std)
            
            # Total entropy is the sum of between-world and within-world components
            entropies[t] = between_world_entropy + within_world_entropy
        
        return entropies
    else:
        forecast_slice = samples[:, prefix_len:prefix_len+forecast_horizon]  # [n_samples, horizon]
        std_per_timestep = np.std(forecast_slice, axis=0)  # [forecast_horizon]
        return gaussian_entropy(std_per_timestep)



def plot_multi_world_distribution(
    samples,
    world_ids=None,
    branching_point=50,
    show_mean=True,
    show_std=True,
    title="Multi-World Ground Truth Distribution",
    prefix_len=50,
    forecast_horizon=50, 
):
    """
    Visualizes sampled multi-world time series.
    
    Args:
        samples (np.ndarray): [n_samples, seq_len]
        branching_point (int): index where futures begin to diverge
        show_mean (bool): plot mean of trajectories
        show_std (bool): show ±1 std deviation band
    """
    n_samples, seq_len = samples.shape
    total_len = prefix_len+forecast_horizon
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