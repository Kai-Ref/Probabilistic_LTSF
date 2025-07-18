import json

import numpy as np
import torch

from .base_scaler import BaseScaler


class ZScoreScaler(BaseScaler):
    """
    ZScoreScaler performs Z-score normalization on the dataset, transforming the data to have a mean of zero 
    and a standard deviation of one. This is commonly used in preprocessing to normalize data, ensuring that 
    each feature contributes equally to the model.

    Attributes:
        mean (np.ndarray): The mean of the training data used for normalization. 
            If `norm_each_channel` is True, this is an array of means, one for each channel. Otherwise, it's a single scalar.
        std (np.ndarray): The standard deviation of the training data used for normalization.
            If `norm_each_channel` is True, this is an array of standard deviations, one for each channel. Otherwise, it's a single scalar.
        target_channel (int): The specific channel (feature) to which normalization is applied.
            By default, it is set to 0, indicating the first channel.
    """

    def __init__(self, dataset_name: str, train_ratio: float, norm_each_channel: bool, rescale: bool, prefix=None):
        """
        Initialize the ZScoreScaler by loading the dataset and fitting the scaler to the training data.

        The scaler computes the mean and standard deviation from the training data, which is then used to 
        normalize the data during the `transform` operation.

        Args:
            dataset_name (str): The name of the dataset used to load the data.
            train_ratio (float): The ratio of the dataset to be used for training. The scaler is fitted on this portion of the data.
            norm_each_channel (bool): Flag indicating whether to normalize each channel separately. 
                If True, the mean and standard deviation are computed for each channel independently.
            rescale (bool): Flag indicating whether to apply rescaling after normalization. This flag is included for consistency with 
                the base class but is not directly used in Z-score normalization.
        """

        super().__init__(dataset_name, train_ratio, norm_each_channel, rescale)
        self.target_channel = 0  # assuming normalization on the first channel

        # load dataset description and data
        if prefix is not None:
            description_file_path = f'{prefix}datasets/{dataset_name}/desc.json'
            data_file_path = f'{prefix}datasets/{dataset_name}/data.dat'
        else:
            description_file_path = f'datasets/{dataset_name}/desc.json'
            data_file_path = f'datasets/{dataset_name}/data.dat'
        with open(description_file_path, 'r') as f:
            description = json.load(f)
        data = np.memmap(data_file_path, dtype='float32', mode='r', shape=tuple(description['shape']))

        # split data into training set based on the train_ratio
        train_size = int(len(data) * train_ratio)
        train_data = data[:train_size, :, self.target_channel].copy()

        # compute mean and standard deviation
        self.norm_each_channel = norm_each_channel
        if norm_each_channel:
            self.mean = np.mean(train_data, axis=0, keepdims=True)
            self.std = np.std(train_data, axis=0, keepdims=True)
            self.std[self.std == 0] = 1.0  # prevent division by zero by setting std to 1 where it's 0
        else:
            self.mean = np.mean(train_data)
            self.std = np.std(train_data)
            if self.std == 0:
                self.std = 1.0  # prevent division by zero by setting std to 1 where it's 0
        self.mean, self.std = torch.tensor(self.mean), torch.tensor(self.std)

    def transform(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Apply Z-score normalization to the input data.

        This method normalizes the input data using the mean and standard deviation computed from the training data. 
        The normalization is applied only to the specified `target_channel`.

        Args:
            input_data (torch.Tensor): The input data to be normalized.

        Returns:
            torch.Tensor: The normalized data with the same shape as the input.
        """

        mean = self.mean.to(input_data.device)
        std = self.std.to(input_data.device)
        input_data[..., self.target_channel] = (input_data[..., self.target_channel] - mean) / std
        return input_data

    def inverse_transform(self, input_data: torch.Tensor, head='') -> torch.Tensor:
        """
        Reverse the Z-score normalization to recover the original data scale.

        This method transforms the normalized data back to its original scale using the mean and standard deviation 
        computed from the training data. This is useful for interpreting model outputs or for further analysis in the original data scale.

        Args:
            input_data (torch.Tensor): The normalized data to be transformed back.

        Returns:
            torch.Tensor: The data transformed back to its original scale.
        """

        mean = self.mean.to(input_data.device)
        std = self.std.to(input_data.device)
        # Clone the input data to prevent in-place modification (which is not allowed in PyTorch)
        input_data = input_data.clone()
        if head == ['gaussian', 'laplace', 'student_t']:#TODO also handle quantile normalization, i think this only normalizes the first channel...
            input_data[..., 0] = input_data[..., 0] * std + mean
            input_data[..., 1] = input_data[..., 1] * std
        elif head in ['quantile', 'i_quantile']: # apply the scaling across all quantile levels
            input_data = input_data * std.unsqueeze(-1) + mean.unsqueeze(-1)
        elif head in ['m_gaussian', 'm_lr_gaussian']:
            input_data[..., 0] = input_data[..., 0] * std + mean

            # determine the rank
            rank = input_data.shape[-1] - 1 - 1

            # rescale the low rank matrix
            V_full = input_data[..., 1:1+rank]  # [batch_size, nvars, output_dim, rank]

            if not self.norm_each_channel: # if not norm_each channel then std is a single value with shape torch.Size([])
                std = std.view(1, 1, 1, 1)
            else:
                std = std.view(1, 1, -1, 1)

            input_data[..., 1:1+rank] = V_full * std 
            
            # rescale the diagonal matrix
            S_full = input_data[..., 1+rank:]  # [batch_size, nvars, output_dim, 1]
            input_data[..., 1+rank:]  = S_full * std * std 
        else:
            input_data[..., self.target_channel] = input_data[..., self.target_channel] * std + mean
        return input_data
