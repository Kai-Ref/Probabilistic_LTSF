import torch
from torch.optim import SGD
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultivariateNormal
import matplotlib.pyplot as plt
import seaborn as sns

class SeriesSampler:
    def __init__(self, gp_model, runner, num_samples=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gp_model = gp_model
        self.runner = runner
        self.base_model = runner.model 
        self.test_loader = runner.test_data_loader
        self.val_loader = runner.val_data_loader
        self.train_loader = runner.train_data_loader
        self.num_samples = num_samples

    def run(self):
        # 1. Fit the GP
        self.fit_gp()

        # 2. Make the predictions
        predictions = self.get_predictions_base_model()
        window = 0
        serie = 0
        mean_pred = predictions[window, :, serie, 0]  # Shape: [num_time_steps]
        std = predictions[window, :, serie, 1]   # Shape: [num_time_steps]
        
        # 4. define new posterior
        new_posterior = self.init_distribution(mean_pred, std_pred)

        return self.sample(new_posterior)

    def get_batch(self, data_loader, first=True):
        if first:
            # select the first batch
            _iter = iter(data_loader) 
            batch = next(_iter)
        else:
            # select the last batch
            num_batches = len(data_loader)
            if num_batches == 0:
                return None, None
            _iter = iter(data_loader)
            for i in range(num_batches - 1):
                next(_iter)
            batch = next(_iter)

        history = batch['inputs'][..., :1] # batch_size, horizon, num_series, features
        targets = batch['target'][..., :1] # batch_size, horizon, num_series, features
        targets = torch.cat([history, targets], dim=1)  # batch_size, horizon, num_series = (32, 360, 7)
        print(targets.shape)
        num_series = 0
        targets = targets[0, :, num_series]  # only select first batch/window at first (360, num_series)
        horizon_length = targets.shape[0]
    
        x = torch.arange(horizon_length).to(self.device, dtype=torch.float32).unsqueeze(-1)  # [360, 1]
        y = targets.to(self.device, dtype=torch.float32).unsqueeze(-1)  # [360, num_series, 1]
        return x, y

    def fit_gp(self):
        x, y = self.get_batch(self.train_loader, first=False)
        self.gp_model = self.gp_model(train_X=x, train_Y=y)
        
        # Register noise constraint
        self.gp_model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))
        
        mll = ExactMarginalLogLikelihood(likelihood=self.gp_model.likelihood, model=self.gp_model)
        # Set mll and all submodules to the specified dtype and device
        mll = mll.to(x.device)
    
        optimizer = SGD([{"params": self.gp_model.parameters()}], lr=0.025)
        NUM_EPOCHS = 1500
    
        self.gp_model.train()
    
        for epoch in range(NUM_EPOCHS):
            optimizer.zero_grad()
            output = self.gp_model(x)
            loss = -mll(output, self.gp_model.train_targets)
            loss.backward()
            if (epoch + 1) % 100 == 0:
                print(
                    f"Epoch {epoch+1:>3}/{NUM_EPOCHS} - Loss: {loss.item():>4.3f} "
                    f"lengthscale: {self.gp_model.covar_module.lengthscale.item():>4.3f} "
                    f"noise: {self.gp_model.likelihood.noise.item():>4.3f}"
                )
            optimizer.step()

    def get_predictions_base_model(self):
        self.base_model.eval()  # Ensure model is in eval mode
        model.to(self.device)  # Move to appropriate device
        predictions = []
        dataloader = self.val_loader
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(dataloader)):
                # TODO scaler
                model_return = self.runner.forward(batch, epoch=100, iter_num=100, train=False)
                #forecasts = model(batch['inputs'][..., :1].to(device), batch['target'].to(device), batch_seen=0, epoch=0, train=False)
                forecasts = model_return['prediction']
                predictions.append(forecasts)
        predictions = torch.cat(predictions, dim=0)
        return predictions

    def init_distribution(self, mean_pred, std_pred):
        x, y = self.get_batch(self.test_loader, first=True)
        
        self.gp_model.eval()  # Ensure model is in evaluation mode
        
        # Filter x to match the length of mean_pred
        x_filter = x[-len(mean_pred):]
        
        # 1. Compute the kernel covariance matrix for the filtered test points
        kernel_matrix = self.gp_model.covar_module(x_filter).evaluate()
        
        # 2. Adjust the covariance matrix using the standard deviation predictions
        updated_cov = kernel_matrix + torch.diag(std_pred**2)  # Convert std to variance and create diagonal covariance
        
        # 3. Construct the updated Multivariate Normal distribution
        new_posterior = MultivariateNormal(mean_pred, updated_cov)
        return new_posterior

    def sample(self, distribution):
        # Sample self.num_samples realization(s) from the adjusted GP
        new_sample = distribution.rsample(torch.Size([self.num_samples]))  # Shape: (num_samples, test_x.size(0))
        return new_sample

    def visualize_corr(self):
        # Get test data
        test_x, _ = self.get_batch(self.test_loader, first=True)
        
        # Compute kernel matrix
        kernel_matrix = self.gp_model.covar_module(test_x).evaluate()
        
        # Visualize correlation
        plt.figure(figsize=(8, 6))
        sns.heatmap(kernel_matrix.detach().to('cpu'), cmap="coolwarm", square=True, cbar=True)
        plt.title("Kernel Matrix Correlation")
        plt.show()