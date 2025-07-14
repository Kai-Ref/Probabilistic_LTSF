import os
import sys
prefix = "/home/kreffert/"
# prefix = "/pfs/data6/home/ma/ma_ma/ma_kreffert/" 
os.chdir(f'{prefix}Probabilistic_LTSF/BasicTS/')
project_root = os.path.join(prefix, "Probabilistic_LTSF", "BasicTS")
sys.path.append(project_root)
from basicts.metrics import masked_mae, masked_mse, nll_loss, crps, Evaluator, quantile_loss, empirical_crps
from easytorch.device import set_device_type
from easytorch.utils import get_logger, set_visible_devices
# set the device type (CPU, GPU, or MLU)
device_type ='gpu'
gpus = '0'
set_device_type(device_type)
set_visible_devices(gpus)
from easydict import EasyDict
from tqdm import tqdm

import torch
import pickle
import gc
import torch  # assuming PyTorch is being used
from prob.prob_head import ProbabilisticHead
from basicts.data import TimeSeriesForecastingDataset, OneStepForecastingDataset
import numpy as np

# extract the paths to the configs and weights
import yaml
    
def reconstruct_paths(
    simplified_dict,
    _dataset=['ETTh1', 'ETTm1'],
    _models=['DLinear', 'PatchTST', 'DeepAR'],
    _dists=['q', 'iq', 'u', 'm'],
    _seeds=[0, 1, 2, 3, 4],
    _model_dist_map=None  # optional: {'DLinear': ['m'], 'DeepAR': ['u']}
):
    base_path = "final_weights/"
    dist_mapping = {"iq": "i_quantile", "u": "univariate", "m": "multivariate", "q": "quantile"}
    filtered_dict = {}

    for dataset, models in simplified_dict.items():
        if dataset not in _dataset:
            continue
        for model, dists in models.items():
            if _model_dist_map:
                if model not in _model_dist_map:
                    continue
            elif model not in _models:
                continue

            allowed_dists = _model_dist_map[model] if _model_dist_map else _dists

            for dist, seeds in dists.items():
                if dist not in allowed_dists:
                    continue
                _cfg = f"{dataset}_prob_quantile.py" if dist in ["q", "iq"] else f"{dataset}_prob.py"
                _ckpt = "_best_val_QL.pt" if dist in ["q", "iq"] else "_best_val_NLL.pt"
                mapped_dist = dist_mapping.get(dist, dist)

                for seed, path_suffix in seeds.items():
                    if seed not in _seeds or path_suffix is None:
                        continue
                    prefix = f"{base_path}{dataset}/{model}/{mapped_dist}/{seed}/{path_suffix}"
                    ckpt_path = f"{prefix}/{model}{_ckpt}"
                    if os.path.isfile(ckpt_path): # check if samples where created as well?
                        filtered_dict.setdefault(dataset, {}).setdefault(model, {}).setdefault(dist, {})[seed] = {
                            'cfg': f"{prefix}/{_cfg}",
                            'ckpt': f"{prefix}/{model}{_ckpt}"
                        }
                    # else:
                    #     print(f"{ckpt_path} not found.")

    return filtered_dict


def load_cfg(cfg, random_state=None):
    from easytorch.config import init_cfg
    # cfg path which start with dot will crash the easytorch, just remove dot
    while isinstance(cfg, str) and cfg.startswith(('./','.\\')):
        cfg = cfg[2:]
    # while ckpt_path.startswith(('./','.\\')):
    #     ckpt_path = ckpt_path[2:]
    
    # initialize the configuration
    cfg = init_cfg(cfg, save=False)
    return cfg
    
@torch.no_grad()
def _forward(runner, cfg, data, epoch: int = None, iter_num: int = None, train: bool = False, ims=False, **kwargs):
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
        distribution_type = runner.distribution_type
        model_name = cfg["MODEL"]["NAME"]
        if distribution_type in ['gaussian', 'student_t', 'laplace', 'm_lr_gaussian']:
             prob_args = cfg['MODEL']['PARAM'].get('prob_args', None)
             prob_head = ProbabilisticHead(1, 1, distribution_type, prob_args=prob_args)
             sample = True
        elif distribution_type in ['i_quantile']:
            sample = True
        else:
            prob_head = None
            sample = False
        data = runner.preprocessing(data)

        # Preprocess input data
        future_data, history_data = data['target'], data['inputs']
        history_data = runner.to_running_device(history_data)  # Shape: [B, L, N, C]
        future_data = runner.to_running_device(future_data)    # Shape: [B, L, N, C]
        batch_size, length, num_nodes, _ = future_data.shape

        # Select input features
        history_data = runner.select_input_features(history_data)
        future_data_4_dec = runner.select_input_features(future_data)

        if not train:
            # For non-training phases, use only temporal features
            future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        # Forward pass through the model
        if not (model_name == 'DeepAR'):# and distribution_type in ['i_quantile']):
            with torch.no_grad():
                runner.model.eval()
                model_return = runner.model(history_data=history_data, future_data=future_data_4_dec,
                                            batch_seen=iter_num, epoch=epoch, train=train)
            if distribution_type in ['gaussian', 'student_t', 'laplace', 'm_lr_gaussian']:
                samples = prob_head.sample(model_return, num_samples=100, random_state=None) # shape torch.Size([num_samples, batch_size, output_len, num_series])
                samples = samples.permute(1, 2, 3, 0)
            elif distribution_type in ['i_quantile']:
                samples = model_return[..., -100:] # torch.Size([64, 720, 7, 100])
                model_return = model_return[..., :-100]
        else:
            with torch.no_grad():
                if distribution_type not in ['i_quantile']:
                    model_return = runner.model(history_data=history_data, future_data=future_data_4_dec,
                                                batch_seen=iter_num, epoch=epoch, train=train) #.sample_trajectories(history_data=history_data, future_data=future_data_4_dec, num_samples=100)
                else:
                    model_return = runner.model.sample_trajectories(history_data=history_data, future_data=future_data_4_dec, num_samples=100).squeeze(-1)
                    model_return = model_return.permute(0, 2, 3, 1)
                    samples = model_return[..., -100:] # torch.Size([64, 720, 7, 100])
                    model_return = model_return[..., :-100]
                # model_return = model_return.permute(1, 2, 3, 0)
                # print(model_return.shape)
                # print(model_return[0, :, 0, :])
                if distribution_type not in ['quantile', 'i_quantile']:
                    samples = runner.model.sample_trajectories(history_data=history_data, future_data=future_data_4_dec, num_samples=100).squeeze(-1)
                    samples = samples.permute(0, 2, 3, 1)
                # model_return = torch.zeros(batch_size, length, num_nodes, 4)
            

        # Parse model return
        if isinstance(model_return, torch.Tensor):
            model_return = {'prediction': model_return}
        if 'inputs' not in model_return:
            model_return['inputs'] = runner.select_target_features(history_data)
        if 'target' not in model_return:
            model_return['target'] = runner.select_target_features(future_data)
            
        # Ensure the output shape is correct
        assert list(model_return['prediction'].shape)[:3] == [batch_size, length, num_nodes], \
            f"The shape of the output is incorrect. Ensure it matches [B, L, N, C]. Current {list(model_return['prediction'].shape)[:3]} != {[batch_size, length, num_nodes]}"

        model_return = runner.postprocessing(model_return)
        if sample:
            model_return['samples'] = runner.scaler.inverse_transform(samples, head='quantile')
            assert list(model_return['samples'].shape)[:3] == [batch_size, length, num_nodes], \
            f"The shape of the output is incorrect. Ensure it matches [B, L, N, C]. Current {list(model_return['samples'].shape)} != {[batch_size, length, num_nodes]}"
        return model_return
            
@torch.no_grad()
def get_predictions(runner, cfg, first_only=False):
    print(cfg["MODEL"]["NAME"])
    # init test
    runner.test_interval = cfg['TEST'].get('INTERVAL', 1)
    runner.test_data_loader = runner.build_test_data_loader(cfg)

    runner.model.eval()
    prediction, target, inputs = [], [], []
    distribution_type = runner.distribution_type
    if distribution_type in ['gaussian', 'student_t', 'laplace', 'm_lr_gaussian', 'i_quantile']:
        sample = True
        samples = []
    else:
        sample = False
    pbar = tqdm(runner.test_data_loader)
    losses = []
    for i, data in enumerate(pbar):
        if (i >= 1) and (first_only):
            break
        # if model DeepAR -> forward with postprocessing of quantile and 100 sample trajectories!
        
        forward_return = _forward(runner, cfg, data, epoch=None, iter_num=None, train=False)
        if not runner.if_evaluate_on_gpu:
            forward_return['prediction'] = forward_return['prediction'].detach().cpu()
            forward_return['target'] = forward_return['target'].detach().cpu()
            forward_return['inputs'] = forward_return['inputs'].detach().cpu()
            if sample:
                forward_return['samples'] = forward_return['samples'].detach().cpu()
        # print(forward_return['prediction'].shape)
        loss = runner.metric_forward(runner.loss, forward_return)
        losses.append(loss)
        pbar.set_postfix(avg_loss=f"{torch.mean(torch.tensor(losses)):.4f}", loss=f"{loss:.4f}")
                

        prediction.append(forward_return['prediction'])
        target.append(forward_return['target'])
        inputs.append(forward_return['inputs'])
        if sample:
            samples.append(forward_return['samples'])

    prediction = torch.cat(prediction, dim=0)
    target = torch.cat(target, dim=0)
    inputs = torch.cat(inputs, dim=0)
    if sample:
        samples = torch.cat(samples, dim=0)
        return {'prediction': prediction, 'target': target, 'inputs': inputs, 'samples':samples}
    else:
        return {'prediction': prediction, 'target': target, 'inputs': inputs}
    # configs[dataset][model][dist][random_state]['returns_all'] = returns_all

def load_runner(configs, first_only=False):
    for dataset in configs.keys():
        for model in configs[dataset].keys():
            for dist in configs[dataset][model].keys():
                for random_state in configs[dataset][model][dist].keys():
                    # configs[dataset][model][dist][random_state]['cfg'] = load_cfg(configs[dataset][model][dist][random_state]['cfg'])
                    # cfg = configs[dataset][model][dist][random_state]['cfg']
                    cfg = load_cfg(configs[dataset][model][dist][random_state]['cfg'])
                    cfg['DATASET']['TYPE'] = TimeSeriesForecastingDataset
                    prefix = '/work/kreffert'
                    ckpt = configs[dataset][model][dist][random_state]['ckpt']
                    
                    ckpt_dir = cfg['TRAIN'].get('CKPT_SAVE_DIR', None)
                    ckpt_dir = ckpt_dir.replace("/home", "/work")
                    if os.path.isfile(f"{ckpt_dir}/model_return_.pkl"):
                        print(f'Model return already available {ckpt_dir}/model_return_.pkl')
                    else:
                        strict=True
                        runner = cfg['RUNNER'](cfg)
                        # setup the graph if needed
                        if runner.need_setup_graph:
                            runner.setup_graph(cfg=cfg, train=False)
                            
                        print(f'Loading model checkpoint from {ckpt}')
                        runner.load_model(ckpt_path=ckpt, strict=strict)
                        
                        # runner.test_pipeline(cfg=cfg, save_metrics=False, save_results=False)
                        # configs[dataset][model][dist][random_state]['runner'] = runner
    
                        # produce predictions
                        returns_all = get_predictions(runner, cfg, first_only=first_only)
                        print(returns_all['prediction'].shape)
                        print(returns_all['target'].shape)
                        print(returns_all['inputs'].shape)
                        if 'samples' in returns_all.keys():
                            print(returns_all['samples'].shape)

                        # save the input and target one time individually, if they have not been saved yet
                        if not os.path.isfile(f"/work/kreffert/Probabilistic_LTSF/BasicTS/final_weights/ETTh1/inputs.pkl"):
                            with open(f'/work/kreffert/Probabilistic_LTSF/BasicTS/final_weights/ETTh1/inputs.pkl', 'wb') as f:
                                pickle.dump(returns_all['inputs'], f)
                        if not os.path.isfile(f"/work/kreffert/Probabilistic_LTSF/BasicTS/final_weights/ETTh1/target.pkl"):
                            with open(f'/work/kreffert/Probabilistic_LTSF/BasicTS/final_weights/ETTh1/target.pkl', 'wb') as f:
                                pickle.dump(returns_all['target'], f)
                            
                        # Save predictions and samples together in one dict
                        results_dict = {"prediction": returns_all["prediction"]}
                        
                        if "samples" in returns_all:
                            results_dict["samples"] = returns_all["samples"]
                        
                        ckpt_dir = cfg['TRAIN'].get('CKPT_SAVE_DIR', None)
                        ckpt_dir = ckpt_dir.replace("/home", "/work")
                        os.makedirs(ckpt_dir, exist_ok=True)
                        print(f'saved under {ckpt_dir}/model_return_.pkl')
                        with open(f'{ckpt_dir}/model_return_.pkl', 'wb') as f:
                            pickle.dump(results_dict, f)
                        del runner
                        del cfg
                        del returns_all
                        gc.collect()
                        torch.cuda.empty_cache()
                        
if __name__ == '__main__':
    prefix = "/home/kreffert/"
    with open(f'{prefix}Probabilistic_LTSF/notebooks/Final plots/weights.yaml', 'r') as file:
        _configs = yaml.safe_load(file)
    
    _model_dist_map={'DLinear': ['m'], 'DeepAR': ['u']}
    _models = ['DeepAR'] #, 'DLinear'] 
    _dataset = ['ETTh1']
    _dists = ['iq']#['u', 'm', 'q', 'iq']
    _seeds = [2]
    # _configs = reconstruct_paths(_configs, _dataset=_dataset, _models=_models, _dists=_dists, _seeds=_seeds, _model_dist_map=_model_dist_map)
    _configs = reconstruct_paths(_configs, _dataset=_dataset, _models=_models, _dists=_dists)
    load_runner(_configs, first_only=False)
    
