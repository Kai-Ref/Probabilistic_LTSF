from .simple_tsf_dataset import TimeSeriesForecastingDataset
import numpy as np

class OneStepForecastingDataset(TimeSeriesForecastingDataset):
    def __getitem__(self, index: int) -> dict:
        """
        Returns one-step-ahead targets instead of a whole prediction horizon.
        """
        history_data = self.data[index:index + self.input_len]
        future_data = self.data[index + self.input_len]  # Just one step ahead
        return {
            'inputs': history_data,
            'target': np.expand_dims(future_data, axis=0)  # (1, D) shape
        }

    def __len__(self) -> int:
        return len(self.data) - self.input_len  # Adjusted for 1-step-ahead
