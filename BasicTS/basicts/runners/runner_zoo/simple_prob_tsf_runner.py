from typing import Dict, Optional, Tuple, Union
import functools
import torch
import inspect
import time
import datetime
import json
from tqdm import tqdm
import numpy as np

from ..base_tsf_runner import BaseTimeSeriesForecastingRunner

from easytorch.utils import master_only
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from easytorch.core.checkpoint import (backup_last_ckpt, clear_ckpt, load_ckpt,
                                       save_ckpt)

from ...metrics import (masked_mae, Evaluator)
import wandb

class SimpleProbTimeSeriesForecastingRunner(BaseTimeSeriesForecastingRunner):
    """
    A Simple Runner for Time Series Forecasting: 
    Selects forward and target features. This runner is designed to handle most cases.

    Args:
        cfg (Dict): Configuration dictionary.
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.forward_features = cfg['MODEL'].get('FORWARD_FEATURES', None)
        self.target_features = cfg['MODEL'].get('TARGET_FEATURES', None)
        self.target_time_series = cfg['MODEL'].get('TARGET_TIME_SERIES', None)

        # self.quantiles = cfg['MODEL']['PARAM'].get('quantiles', None)
        self.distribution_type = cfg['MODEL']['PARAM'].get('distribution_type', None)
        self.prob_args = cfg['MODEL']['PARAM'].get('prob_args', None)
        self.quantiles = cfg['MODEL']['PARAM']['prob_args'].get('quantiles', None)
        self.output_seq_len = cfg["DATASET"]["PARAM"]["output_len"]
        self.model_name = cfg["MODEL"]["NAME"]
        print(self.model_name)
        self.use_wandb = cfg['USE_WANDB']
        # start wandb
        # if self.use_wandb:
        #     timestamp = datetime.datetime.now().strftime("%b_%d_%H_%M")
        #     os.environ["WANDB_API_KEY"] = "fc6544bc618ba7ae3008cf7c53d9650e1668bf12"
        #     wandb.init(entity="kai-reffert-university-mannheim",
        #         project="Prob_LTSF",  
        #     #group=f"DDP_{config['model_name']}_{timestamp}",
        #     name=f"{self.distribution_type}_{self.model_name}_{timestamp}",
        #     config=cfg)

    def build_scaler(self, cfg: Dict):
        """Build scaler.

        Args:
            cfg (Dict): Configuration.

        Returns:
            Scaler instance or None if no scaler is declared.
        """

        if 'SCALER' in cfg:
            if cfg['SCALER']['TYPE'] == 'None':
                return None
            else:
                return cfg['SCALER']['TYPE'](**cfg['SCALER']['PARAM'])
        return None

    def preprocessing(self, input_data: Dict) -> Dict:
        """Preprocess data.

        Args:
            input_data (Dict): Dictionary containing data to be processed.

        Returns:
            Dict: Processed data.
        """

        if self.scaler is not None:
            input_data['target'] = self.scaler.transform(input_data['target'])
            input_data['inputs'] = self.scaler.transform(input_data['inputs'])
        # TODO: add more preprocessing steps as needed.
        return input_data

    def postprocessing(self, input_data: Dict) -> Dict:
        """Postprocess data.

        Args:
            input_data (Dict): Dictionary containing data to be processed.

        Returns:
            Dict: Processed data.
        """

        # rescale data
        if self.scaler is not None and self.scaler.rescale:
            # TODO decide what to do with the std predictions -> ALso consider qunatiles.....

            # Assuming the last dimension contains the mean and std predictions
            # mean_predictions = input_data['prediction'][..., 0]  # Mean is the first element in the last dimension
            # input_data['prediction'][..., 0] = self.scaler.inverse_transform(mean_predictions)
            input_data['prediction'] = self.scaler.inverse_transform(input_data['prediction'], head=self.distribution_type)
            # input_data['prediction'] = self.scaler.inverse_transform(input_data['prediction'])
            input_data['target'] = self.scaler.inverse_transform(input_data['target'])
            input_data['inputs'] = self.scaler.inverse_transform(input_data['inputs'])

        # subset forecasting
        if self.target_time_series is not None:# for PatchTST at least it is NONE
            input_data['target'] = input_data['target'][:, :, self.target_time_series, :]
            input_data['prediction'] = input_data['prediction'][:, :, self.target_time_series, :]

        # TODO: add more postprocessing steps as needed.
        return input_data

    def forward(self, data: Dict, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> Dict:
        """
        Performs the forward pass for training, validation, and testing. 

        Args:
            data (Dict): A dictionary containing 'target' (future data) and 'inputs' (history data) (normalized by self.scaler).
            epoch (int, optional): Current epoch number. Defaults to None.
            iter_num (int, optional): Current iteration number. Defaults to None.
            train (bool, optional): Indicates whether the forward pass is for training. Defaults to True.

        Returns:
            Dict: A dictionary containing the keys:
                  - 'inputs': Selected input features.
                  - 'prediction': Model predictions.
                  - 'target': Selected target features.

        Raises:
            AssertionError: If the shape of the model output does not match [B, L, N].
        """

        data = self.preprocessing(data)

        # Preprocess input data
        future_data, history_data = data['target'], data['inputs']
        history_data = self.to_running_device(history_data)  # Shape: [B, L, N, C]
        future_data = self.to_running_device(future_data)    # Shape: [B, L, N, C]
        batch_size, length, num_nodes, _ = future_data.shape

        # Select input features
        history_data = self.select_input_features(history_data)
        future_data_4_dec = self.select_input_features(future_data)

        if not train:
            # For non-training phases, use only temporal features
            future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        # Forward pass through the model
        model_return = self.model(history_data=history_data, future_data=future_data_4_dec,
                                batch_seen=iter_num, epoch=epoch, train=train)

        if (model_return.shape[1] != length) and (self.distribution_type == 'i_quantile'):
            quantiles = model_return[:, -1:, :, :].squeeze()
            model_return = model_return[:, :-1, :, :]

        # Parse model return
        if isinstance(model_return, torch.Tensor):
            model_return = {'prediction': model_return}
        if 'inputs' not in model_return:
            model_return['inputs'] = self.select_target_features(history_data)
        if 'target' not in model_return:
            model_return['target'] = self.select_target_features(future_data)

        if self.distribution_type == 'i_quantile':
            if 'quantiles' in locals(): 
                model_return['quantiles'] = quantiles
            else:
                model_return['quantiles'] = self.quantiles
        # Ensure the output shape is correct
        assert list(model_return['prediction'].shape)[:3] == [batch_size, length, num_nodes], \
            f"The shape of the output is incorrect. Ensure it matches [B, L, N, C]. Current {list(model_return['prediction'].shape)[:3]} != {[batch_size, length, num_nodes]}"

        model_return = self.postprocessing(model_return)
        return model_return

    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        """
        Selects input features based on the forward features specified in the configuration.

        Args:
            data (torch.Tensor): Input history data with shape [B, L, N, C1].

        Returns:
            torch.Tensor: Data with selected features with shape [B, L, N, C2].
        """

        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]
        return data

    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        """
        Selects target features based on the target features specified in the configuration.

        Args:
            data (torch.Tensor): Model prediction data with shape [B, L, N, C1].

        Returns:
            torch.Tensor: Data with selected target features and shape [B, L, N, C2].
        """

        data = data[:, :, :, self.target_features]
        return data

    def select_target_time_series(self, data: torch.Tensor) -> torch.Tensor:
        """
        Select target time series based on the target time series specified in the configuration.

        Args:
            data (torch.Tensor): Model prediction data with shape [B, L, N1, C].

        Returns:
            torch.Tensor: Data with selected target time series and shape [B, L, N2, C].
        """

        data = data[:, :, self.target_time_series, :]
        return data

    @master_only
    def save_best_model(self, epoch: int, metric_name: str, greater_best: bool = True):
        """Save the best model while training.

        Examples:
            >>> def on_validating_end(self, train_epoch: Optional[int]):
            >>>     if train_epoch is not None:
            >>>         self.save_best_model(train_epoch, 'val/loss', greater_best=False)

        Args:
            epoch (int): current epoch.
            metric_name (str): metric name used to measure the model, must be registered in `epoch_meter`.
            greater_best (bool, optional): `True` means greater value is best, such as `acc`
                `False` means lower value is best, such as `loss`. Defaults to True.
        """

        metric = self.meter_pool.get_avg(metric_name)
        best_metric = self.best_metrics.get(metric_name)
        if best_metric is None or (metric > best_metric if greater_best else metric < best_metric):
            self.best_metrics[metric_name] = metric
            model = self.model.module if isinstance(self.model, DDP) else self.model
            ckpt_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optim_state_dict': self.optim.state_dict(),
                'best_metrics': self.best_metrics
            }
            ckpt_path = os.path.join(
                self.ckpt_save_dir,
                '{}_best_{}.pt'.format(self.model_name, metric_name.replace('/', '_'))
            )
            save_ckpt(ckpt_dict, ckpt_path, self.logger)
            self.current_patience = self.early_stopping_patience # reset patience
        else:
            if self.early_stopping_patience is not None:
                self.current_patience -= 1

    def metric_forward(self, metric_func, args: Dict) -> torch.Tensor:
        """Compute metrics using the given metric function.

        Args:
            metric_func (function or functools.partial): Metric function.
            args (Dict): Arguments for metrics computation.
            TODO data (Dict): Data dictionary containing 'prediction' and 'target'.
                only used for Evaluator to get the seasonal statistics.

        Returns:
            torch.Tensor: Computed metric value.
        """

        covariate_names = inspect.signature(metric_func).parameters.keys()
        args = {k: v for k, v in args.items() if k in covariate_names}
        if 'null_val' in covariate_names:#'null_val' not in metric_func.keywords and
            args['null_val'] = self.null_val
        
        if isinstance(metric_func, functools.partial) or (type(metric_func) is Evaluator):
            metric_item = metric_func(**args)
        elif callable(metric_func):
            if ('quantile_loss' in str(metric_func)): # and ('quantiles' not in list(self.prob_args.keys())):
                args['quantiles'] = self.quantiles
            if ('nll_loss' in str(metric_func)) or ('crps' in str(metric_func)):
                args['distribution_type'] = self.distribution_type
                args['prob_args'] = self.prob_args

            metric_item = metric_func(**args)
        else:
            raise TypeError(f'Unknown metric type: {type(metric_func)}')
        return metric_item

    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        """Training iteration process.

        Args:
            epoch (int): Current epoch.
            iter_index (int): Current iteration index.
            data (Union[torch.Tensor, Tuple]): Data provided by DataLoader.

        Returns:
            torch.Tensor: Loss value.
        """
        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index
        forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)

        if self.cl_param:
            cl_length = self.curriculum_learning(epoch=epoch)
            forward_return['prediction'] = forward_return['prediction'][:, :cl_length, :, :]
            forward_return['target'] = forward_return['target'][:, :cl_length, :, :]
        
        loss = self.metric_forward(self.loss, forward_return)
        self.update_epoch_meter('train/loss', loss.item())
        if self.use_wandb and iter_index % 10 == 0:
            wandb_dict = {'train/epoch': epoch, 'train/iter': iter_num, 'train/loss': loss.item()}
        for metric_name, metric_func in self.metrics.items():
            if "Val_Evaluator" in metric_name:
                continue
            metric_item = self.metric_forward(metric_func, forward_return)
            if type(metric_func) is Evaluator:
                for key, value in metric_item.items():
                    try:
                        self.update_epoch_meter(f'ProbTS-train/{key}', value)
                    except KeyError:
                        self.register_epoch_meter(f'ProbTS-train/{key}', 'train', '{:.4f}')
                        self.update_epoch_meter(f'ProbTS-train/{key}', value)
            else:
                self.update_epoch_meter(f'train/{metric_name}', metric_item.item())
                if self.use_wandb and iter_index % 10 == 0:
                    wandb_dict[f'train/{metric_name}'] = metric_item.item()
                    
        if (self.use_wandb) and (iter_index % 10 == 0):   
            wandb.log(wandb_dict, step=iter_num)
        return loss

    def val_iters(self, iter_index: int, data: Union[torch.Tensor, Tuple]):
        """Validation iteration process.

        Args:
            iter_index (int): Current iteration index.
            data (Union[torch.Tensor, Tuple]): Data provided by DataLoader.
        """

        forward_return = self.forward(data=data, epoch=None, iter_num=iter_index, train=False)
        loss = self.metric_forward(self.loss, forward_return)
        self.update_epoch_meter('val/loss', loss.item())

        for metric_name, metric_func in self.metrics.items():            
            metric_item = self.metric_forward(metric_func, forward_return)
            if type(metric_func) is Evaluator:
                for key, value in metric_item.items():
                    try:
                        self.update_epoch_meter(f'ProbTS-val/{key}', value)
                    except KeyError:
                        self.register_epoch_meter(f'ProbTS-val/{key}', 'val', '{:.4f}')
                        self.update_epoch_meter(f'ProbTS-val/{key}', value)
            else:
                self.update_epoch_meter(f'val/{metric_name}', metric_item.item())

    @torch.no_grad()
    @master_only
    def validate(self, cfg: Dict = None, train_epoch: Optional[int] = None):
        """Validate model.

        Args:
            cfg (Dict, optional): config
            train_epoch (int, optional): current epoch if in training process.
        """

        # init validation if not in training process
        if train_epoch is None:
            self.init_validation(cfg)

        self.logger.info('Start validation.')

        self.on_validating_start(train_epoch)

        val_start_time = time.time()
        self.model.eval()

        # prediction, target, inputs = [], [], []

        # for data in tqdm(self.val_data_loader):
        #     forward_return = self.forward(data, epoch=None, iter_num=None, train=False)

        #     loss = self.metric_forward(self.loss, forward_return)
        #     self.update_epoch_meter('val/loss', loss.item())

        #     if not self.if_evaluate_on_gpu:
        #         forward_return['prediction'] = forward_return['prediction'].detach().cpu()
        #         forward_return['target'] = forward_return['target'].detach().cpu()
        #         forward_return['inputs'] = forward_return['inputs'].detach().cpu()

        #     prediction.append(forward_return['prediction'])
        #     target.append(forward_return['target'])
        #     inputs.append(forward_return['inputs'])

        # prediction = torch.cat(prediction, dim=0)
        # target = torch.cat(target, dim=0)
        # inputs = torch.cat(inputs, dim=0)

        # returns_all = {'prediction': prediction, 'target': target, 'inputs': inputs}
        # metrics_results = self.compute_evaluation_metrics(returns_all, mode='val')

        #tqdm process bar
        data_iter = tqdm(self.val_data_loader)

        # val loop
        for iter_index, data in enumerate(data_iter):
            self.val_iters(iter_index, data)

        val_end_time = time.time()
        self.update_epoch_meter('val/time', val_end_time - val_start_time)
        
        # Now log validation averages to wandb
        if (self.use_wandb) and (train_epoch is not None):
            # Calculate proper global step at end of validation
            global_step = train_epoch * self.iter_per_epoch
            
            # Log all validation metrics at once
            wandb_dict = {}            
            # Add metrics from meter pool
            for meter_name in self.meter_pool._pool.keys():
                if 'val' in meter_name:
                    meter_value = self.meter_pool._pool[meter_name]['meter'].avg
                    wandb_dict[f'epoch_summary/{meter_name}'] = meter_value
                    wandb_dict[f'{meter_name}'] = meter_value
            wandb.log(wandb_dict, step=global_step)
        
        # print val meters
        self.print_epoch_meters('val')
        if train_epoch is not None:
            # tensorboard plt meters
            self.plt_epoch_meters('val', train_epoch // self.val_interval)

        self.on_validating_end(train_epoch)

    def compute_evaluation_metrics(self, returns_all: Dict):
        """Compute metrics for evaluating model performance during the test process.

        Args:
            returns_all (Dict): Must contain keys: inputs, prediction, target.
        """

        metrics_results = {}
        wandb_dict = {} if self.use_wandb else None
        for i in self.evaluation_horizons:
            pred = returns_all['prediction'][:, i, :, :]
            real = returns_all['target'][:, i, :, :]

            metrics_results[f'horizon_{i + 1}'] = {}
            metric_repr = ''
            for metric_name, metric_func in self.metrics.items():
                if metric_name.lower() == 'mase':
                    continue # MASE needs to be calculated after all horizons
                metric_item = self.metric_forward(metric_func, {'prediction': pred, 'target': real})
                metric_repr += f', test {metric_name}: {metric_item.item():.4f}'
                metrics_results[f'horizon_{i + 1}'][metric_name] = metric_item.item()
            self.logger.info(f'Evaluate best model on Testing data for horizon {i + 1}{metric_repr}')
        

        metrics_results['overall'] = {}
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, returns_all)
            if type(metric_func) is Evaluator:
                for key, value in metric_item.items():
                    try:
                        self.update_epoch_meter(f'ProbTS-test/{key}', value)
                    except KeyError:
                        self.register_epoch_meter(f'ProbTS-test/{key}', 'test', '{:.4f}')
                        self.update_epoch_meter(f'ProbTS-test/{key}', value)
                    metrics_results['overall']['ProbTS-{key}'] = value
                    if self.use_wandb:
                        wandb_dict[f'ProbTS-test/{key}'] = value
            else:
                self.update_epoch_meter(f'test/{metric_name}', metric_item.item())
                metrics_results['overall'][metric_name] = metric_item.item()
                if self.use_wandb:
                    wandb_dict[f'test/{metric_name}'] = metric_item.item()
        if (self.use_wandb) and (self.use_wandb):   
            # If we're in the training process, use the current global step
            if hasattr(self, 'current_epoch') and self.current_epoch is not None:
                global_step = self.current_epoch * self.iter_per_epoch
                wandb.log(wandb_dict, step=global_step)
            else:
                # For standalone testing
                wandb.log(wandb_dict)
        return metrics_results


    def test_iters(self, iter_index: int, data: Union[torch.Tensor, Tuple]):
        """Validation iteration process.

        Args:
            iter_index (int): Current iteration index.
            data (Union[torch.Tensor, Tuple]): Data provided by DataLoader.
        """

        forward_return = self.forward(data=data, epoch=None, iter_num=iter_index, train=False)
        loss = self.metric_forward(self.loss, forward_return)
        self.update_epoch_meter('test/loss', loss.item())

        if not self.if_evaluate_on_gpu:
            forward_return['prediction'] = forward_return['prediction'].detach().cpu()
            forward_return['target'] = forward_return['target'].detach().cpu()
            forward_return['inputs'] = forward_return['inputs'].detach().cpu()

        for metric_name, metric_func in self.metrics.items():            
            metric_item = self.metric_forward(metric_func, forward_return)
            if type(metric_func) is Evaluator:
                for key, value in metric_item.items():
                    try:
                        self.update_epoch_meter(f'ProbTS-test/{key}', value)
                    except KeyError:
                        self.register_epoch_meter(f'ProbTS-test/{key}', 'test', '{:.4f}')
                        self.update_epoch_meter(f'ProbTS-test/{key}', value)
            else:
                self.update_epoch_meter(f'test/{metric_name}', metric_item.item())

    @torch.no_grad()
    @master_only
    def test(self, train_epoch: Optional[int] = None, save_metrics: bool = False, save_results: bool = False) -> Dict:
        """Test process.
        
        Args:
            train_epoch (Optional[int]): Current epoch if in training process.
            save_metrics (bool): Save the test metrics. Defaults to False.
            save_results (bool): Save the test results. Defaults to False.
        """

        # prediction, target, inputs = [], [], []
        data_iter = tqdm(self.test_data_loader)
        for iter_index, data in enumerate(data_iter):
            self.test_iters(iter_index, data)
            # forward_return = self.forward(data, epoch=None, iter_num=None, train=False)

            # loss = self.metric_forward(self.loss, forward_return)
            # self.update_epoch_meter('test/loss', loss.item())

        # Now log validation averages to wandb
        if (self.use_wandb) and (train_epoch is not None):
            # Calculate proper global step at end of validation
            global_step = train_epoch * self.iter_per_epoch
            
            # Log all validation metrics at once
            wandb_dict = {}            
            # Add metrics from meter pool
            for meter_name in self.meter_pool._pool.keys():
                if 'test' in meter_name:
                    meter_value = self.meter_pool._pool[meter_name]['meter'].avg
                    wandb_dict[f'epoch_summary/{meter_name}'] = meter_value
                    wandb_dict[f'{meter_name}'] = meter_value
            wandb.log(wandb_dict, step=global_step)
        #     if not self.if_evaluate_on_gpu:
        #         forward_return['prediction'] = forward_return['prediction'].detach().cpu()
        #         forward_return['target'] = forward_return['target'].detach().cpu()
        #         forward_return['inputs'] = forward_return['inputs'].detach().cpu()

        #     prediction.append(forward_return['prediction'])
        #     target.append(forward_return['target'])
        #     inputs.append(forward_return['inputs'])

        # prediction = torch.cat(prediction, dim=0)
        # target = torch.cat(target, dim=0)
        # inputs = torch.cat(inputs, dim=0)
        
        # if self.model_name == 'DeepAR': # taken from DeepAR_Runner
        #     returns_all = {'prediction': prediction[:, -self.output_seq_len:, :, :],
        #                 'target': target[:, -self.output_seq_len:, :, :],
        #                 'inputs': inputs}
        # else:
        # returns_all = {'prediction': prediction, 'target': target, 'inputs': inputs}
        
        
        # metrics_results = self.compute_evaluation_metrics(returns_all)

        # save
        save_metrics, save_results = False, False
        if save_results:
            # save returns_all to self.ckpt_save_dir/test_results.npz
            test_results = {k: v.cpu().numpy() for k, v in returns_all.items()}
            np.savez(os.path.join(self.ckpt_save_dir, 'test_results.npz'), **test_results)

        if save_metrics:
            # save metrics_results to self.ckpt_save_dir/test_metrics.json
            with open(os.path.join(self.ckpt_save_dir, 'test_metrics.json'), 'w') as f:
                json.dump(metrics_results, f, indent=4)

        # return returns_all

    def on_epoch_end(self, epoch: int) -> None:
        """
        Callback at the end of each epoch to handle validation and testing.

        Args:
            epoch (int): The current epoch number.
        """
        # Log epoch summary to wandb
        if (self.use_wandb):
            global_step = epoch * self.iter_per_epoch
            # Create summary dict with averages
            wandb_dict = {
                'epoch': epoch,
                # 'epoch_summary/train_loss': self.meter_pool['train']['loss']
            }

            # Add training metrics from meter pool
            for meter_name in self.meter_pool._pool.keys():
                if 'train' in meter_name:
                    meter_value = self.meter_pool._pool[meter_name]['meter'].avg
                    wandb_dict[f'epoch_summary/{meter_name}'] = meter_value
            
            wandb.log(wandb_dict, step=global_step)
        
        # print training meters
        self.print_epoch_meters('train')
        # plot training meters to TensorBoard
        self.plt_epoch_meters('train', epoch)
        # perform validation if configured
        if self.val_data_loader is not None and epoch % self.val_interval == 0:
            self.validate(train_epoch=epoch)
        # perform testing if configured
        if self.test_data_loader is not None and epoch % self.test_interval == 0:
            self.test_pipeline(train_epoch=epoch)
        # save the model checkpoint
        # self.save_model(epoch)
        # reset epoch meters
        self.reset_epoch_meters()