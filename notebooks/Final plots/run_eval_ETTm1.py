import os
import argparse
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
import gzip
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
    
    # initialize the configuration
    cfg = init_cfg(cfg, save=False)
    return cfg
    
@torch.no_grad()
def _forward(runner, cfg, data, epoch: int = None, iter_num: int = None, train: bool = False, ims=False, predictions_only=False, **kwargs):
    """
    Performs the forward pass for training, validation, and testing. 
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
    history_data = runner.to_running_device(history_data)
    future_data = runner.to_running_device(future_data)
    batch_size, length, num_nodes, _ = future_data.shape

    # Select input features
    history_data = runner.select_input_features(history_data)
    future_data_4_dec = runner.select_input_features(future_data)

    if not train:
        # For non-training phases, use only temporal features
        future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

    # Forward pass through the model
    if not (model_name == 'DeepAR'):
        runner.model.eval()
        model_return = runner.model(history_data=history_data, future_data=future_data_4_dec,
                                    batch_seen=iter_num, epoch=epoch, train=train)
        if distribution_type in ['gaussian', 'student_t', 'laplace', 'm_lr_gaussian']:
            samples = prob_head.sample(model_return, num_samples=100, random_state=None)
            samples = samples.permute(1, 2, 3, 0)
        elif distribution_type in ['i_quantile']:
            samples = model_return[..., -100:]
            model_return = model_return[..., :-100]
    else:
        if distribution_type not in ['i_quantile']:
            model_return = runner.model(history_data=history_data, future_data=future_data_4_dec,
                                        batch_seen=iter_num, epoch=epoch, train=train)
        else:
            model_return = runner.model.sample_trajectories(history_data=history_data, future_data=future_data_4_dec, num_samples=100).squeeze(-1)
            model_return = model_return.permute(0, 2, 3, 1)
            samples = model_return[..., -100:]
            model_return = model_return[..., :-100]
        if distribution_type not in ['quantile', 'i_quantile']:
            samples = runner.model.sample_trajectories(history_data=history_data, future_data=future_data_4_dec, num_samples=100).squeeze(-1)
            samples = samples.permute(0, 2, 3, 1)

    # Parse model return
    if isinstance(model_return, torch.Tensor):
        model_return = {'prediction': model_return}

    if not predictions_only:
        if 'inputs' not in model_return:
            model_return['inputs'] = runner.select_target_features(history_data)
        if 'target' not in model_return:
            model_return['target'] = runner.select_target_features(future_data)
        
    # Ensure the output shape is correct
    assert list(model_return['prediction'].shape)[:3] == [batch_size, length, num_nodes], \
        f"The shape of the output is incorrect. Ensure it matches [B, L, N]. Current {list(model_return['prediction'].shape)[:3]} != {[batch_size, length, num_nodes]}"

    if runner.scaler:
        # model_return = runner.postprocessing(model_return)
        model_return['prediction'] = runner.scaler.inverse_transform(model_return['prediction'], head=runner.distribution_type)
        if not predictions_only:
            model_return['target'] = runner.scaler.inverse_transform(model_return['target'])
            model_return['inputs'] = runner.scaler.inverse_transform(model_return['inputs'])
    if sample:
        if runner.scaler:
            samples = runner.scaler.inverse_transform(samples, head='quantile')
        model_return['samples'] = samples
        assert list(model_return['samples'].shape)[:3] == [batch_size, length, num_nodes], \
        f"The shape of the samples is incorrect. Ensure it matches [B, L, N]. Current {list(model_return['samples'].shape)} != {[batch_size, length, num_nodes]}"
    
    # Clean up intermediate tensors
    del history_data, future_data, future_data_4_dec
    if 'samples' in locals():
        del samples
    torch.cuda.empty_cache()
    gc.collect()
    return model_return

@torch.no_grad()
def get_predictions(runner, cfg, first_only=False, batch_size_limit=None, predictions_only=False):
    """
    Get predictions with memory optimization.
    
    Args:
        batch_size_limit: Maximum number of batches to process before saving intermediate results
    """
    print(f"Processing model: {cfg['MODEL']['NAME']}")
    
    # init test
    runner.test_interval = cfg['TEST'].get('INTERVAL', 1)
    runner.test_data_loader = runner.build_test_data_loader(cfg)

    runner.model.eval()
    
    distribution_type = runner.distribution_type
    sample = distribution_type in ['gaussian', 'student_t', 'laplace', 'm_lr_gaussian', 'i_quantile']
    
    # Initialize storage lists
    prediction_list, target_list, inputs_list = [], [], []
    if sample:
        samples_list = []
    
    pbar = tqdm(runner.test_data_loader, desc="Processing batches")
    losses = []
    
    for i, data in enumerate(pbar):
        if first_only and i >= 1:
            break
            
        # Process batch
        forward_return = _forward(runner, cfg, data, epoch=None, iter_num=None, train=False, predictions_only=predictions_only)
        
        # Move to CPU immediately to free GPU memory
        if True: #not runner.if_evaluate_on_gpu:
            forward_return['prediction'] = forward_return['prediction'].detach().cpu()
            if not predictions_only:
                forward_return['target'] = forward_return['target'].detach().cpu()
                forward_return['inputs'] = forward_return['inputs'].detach().cpu()
            if sample and 'samples' in forward_return:
                forward_return['samples'] = forward_return['samples'].detach().cpu()
        
        # Calculate loss
        if not predictions_only:
            loss = runner.metric_forward(runner.loss, forward_return)
            losses.append(loss)
            pbar.set_postfix(avg_loss=f"{torch.mean(torch.tensor(losses)):.4f}", loss=f"{loss:.4f}")
        
        # Store results
        prediction_list.append(forward_return['prediction'])
        if not predictions_only:
            target_list.append(forward_return['target'])
            inputs_list.append(forward_return['inputs'])
        if sample and 'samples' in forward_return:
            samples_list.append(forward_return['samples'])
        
        # Clean up
        del forward_return, data
        gc.collect()
        torch.cuda.empty_cache()
        
        # Optional: Save intermediate results if memory is getting tight
        if batch_size_limit and (i + 1) % batch_size_limit == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    if 'runner' in locals():
        del runner
        gc.collect()
        torch.cuda.empty_cache()
    
    print("Concatenating results...")
    if prediction_list[0].device.type == 'cuda':
        for i, prediction in enumerate(prediction_list):
            prediction_list[i] = prediction.detach().cpu()
        prediction = torch.cat(prediction_list, dim=0).detach().cpu()
        del prediction_list
        gc.collect()
        torch.cuda.empty_cache()
        if sample and samples_list:
            for i, samples in enumerate(samples_list):
                samples_list[i] = samples.detach().cpu()
    else:
        for i, prediction in enumerate(prediction_list):
            prediction_list[i] = prediction.detach().to('cuda:0')
        prediction = torch.cat(prediction_list, dim=0).detach().cpu()
        del prediction_list
        gc.collect()
        torch.cuda.empty_cache()
        # if sample and samples_list:
        #     for i, samples in enumerate(samples_list):
        #         samples_list[i] = samples.detach().to('cuda:0')
    if not predictions_only:
        target = torch.cat(target_list, dim=0)
        del target_list
        gc.collect()
        torch.cuda.empty_cache()
        inputs = torch.cat(inputs_list, dim=0)
        del inputs_list
        gc.collect()
        torch.cuda.empty_cache()
        result = {'prediction': prediction, 'target': target, 'inputs': inputs}
    else:
        result = {'prediction': prediction}
    
    if sample and samples_list:
        samples = torch.cat(samples_list, dim=0).detach().cpu()
        result['samples'] = samples
        del samples_list
        gc.collect()
        torch.cuda.empty_cache()
    return result

def load_runner(configs, first_only=False, batch_size_limit=50, predictions_only=False, _format='.pkl', save_pred=True):
    """
    Load and process runners with memory optimization.
    
    Args:
        batch_size_limit: Process this many batches before garbage collection
    """
    for dataset in configs.keys():
        for model in configs[dataset].keys():
            for dist in configs[dataset][model].keys():
                for random_state in configs[dataset][model][dist].keys():
                    
                    cfg = load_cfg(configs[dataset][model][dist][random_state]['cfg'])
                    cfg['DATASET']['TYPE'] = TimeSeriesForecastingDataset
                    ckpt = configs[dataset][model][dist][random_state]['ckpt']
                    
                    ckpt_dir = cfg['TRAIN'].get('CKPT_SAVE_DIR', None)
                    ckpt_dir = ckpt_dir.replace("/home", "/work")
                    
                    if os.path.isfile(f"{ckpt_dir}/model_return.pkl.gz") or os.path.isfile(f"{ckpt_dir}/model_return.pt.gz"):
                        print(f'Model return already available: {ckpt_dir}/model_return.pkl')
                        continue
                    
                    print(f'Processing: {dataset}/{model}/{dist}/{random_state}')
                    
                    # Initialize runner
                    runner = cfg['RUNNER'](cfg)
                    
                    # Setup graph if needed
                    if runner.need_setup_graph:
                        runner.setup_graph(cfg=cfg, train=False)
                    
                    print(f'Loading model checkpoint from {ckpt}')
                    runner.load_model(ckpt_path=ckpt, strict=True)
                    
                    # Get predictions with memory optimization
                    returns_all = get_predictions(runner, cfg, first_only=first_only, batch_size_limit=batch_size_limit, predictions_only=predictions_only)
                    if 'runner' in locals():
                        del runner
                        gc.collect()
                        torch.cuda.empty_cache()
                    
                    print(f"Prediction shape: {returns_all['prediction'].shape}")
                    if not predictions_only:
                        print(f"Target shape: {returns_all['target'].shape}")
                        print(f"Inputs shape: {returns_all['inputs'].shape}")
                    if 'samples' in returns_all:
                        print(f"Samples shape: {returns_all['samples'].shape}")

                    
                    # Save inputs and targets once (if not already saved)
                    base_save_dir = f"/work/kreffert/Probabilistic_LTSF/BasicTS/final_weights/{dataset}"
                    os.makedirs(base_save_dir, exist_ok=True)
                    
                    if not predictions_only:
                        if not os.path.isfile(f"{base_save_dir}/inputs.pkl"):
                            print(f"Saving inputs to {base_save_dir}/inputs.pkl")
                            with open(f'{base_save_dir}/inputs.pkl', 'wb') as f:
                                pickle.dump(returns_all['inputs'], f)
                        
                        if not os.path.isfile(f"{base_save_dir}/target.pkl"):
                            print(f"Saving target to {base_save_dir}/target.pkl")
                            with open(f'{base_save_dir}/target.pkl', 'wb') as f:
                                pickle.dump(returns_all['target'], f)
                    

                    # Save predictions and samples
                    if save_pred:
                        results_dict = {"prediction": returns_all["prediction"]}
                        if "samples" in returns_all:
                            results_dict["samples"] = returns_all["samples"]
                    else:
                        results_dict = {"samples": returns_all["samples"]}
                    
                    os.makedirs(ckpt_dir, exist_ok=True)
                    output_path = f'{ckpt_dir}/model_return{_format}.gz'
                    print(f'Saving results to {output_path}')

                    if _format == '.pkl':
                        # Save using pickle with gzip
                        with gzip.open(output_path, 'wb') as f:
                            pickle.dump(results_dict, f)
                        # Save as pickle file
                        # with open(output_path, 'wb') as f:
                        #     pickle.dump(results_dict, f)
                    elif _format == '.pt':
                        # Save as PyTorch file
                        # torch.save(results_dict, output_path)
                        with gzip.open(output_path, 'wb') as f:
                            torch.save(results_dict, f)
                    
                    print(f"Successfully processed {dataset}/{model}/{dist}/{random_state}")
                    
                    
                    # Cleanup
                    if 'runner' in locals():
                        del runner
                    if 'cfg' in locals():
                        del cfg
                    if 'returns_all' in locals():
                        del returns_all
                    if 'results_dict' in locals():
                        del results_dict
                    
                    # Force garbage collection and clear GPU cache
                    gc.collect()
                    torch.cuda.empty_cache()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run evaluation for ETTm1 dataset')
    parser.add_argument('--models', nargs='+', default=None, help='List of models to evaluate')
    parser.add_argument('--dists', nargs='+', default=None, help='List of distributions')
    parser.add_argument('--seeds', nargs='+', default=None, help='List of distributions')
    parser.add_argument('--dataset', nargs='+', default=None, help='Dataset name')

    args = parser.parse_args()

    prefix = "/home/kreffert/"
    with open(f'{prefix}Probabilistic_LTSF/notebooks/Final plots/weights.yaml', 'r') as file:
        _configs = yaml.safe_load(file)
    
    if args.dataset is not None:
        _dataset = args.dataset
    else:
        _dataset = ['ETTm1']

    if args.models is not None:
        _models = args.models
    else:
        _models = ['DLinear', 'DeepAR', 'PatchTST']

    if args.dists is not None:
        _dists = args.dists
    else:
        _dists = ['u', 'q', 'iq', 'm']

    if args.seeds is not None:
        _seeds = args.seeds
        _seeds = [int(seed) for seed in _seeds] 
    else:
        _seeds = [0, 1, 2, 3, 4]
    
    print(f"Using dataset: {_dataset}, models: {_models}, distributions: {_dists}, seeds: {_seeds}")
    # _configs = reconstruct_paths(_configs, _dataset=_dataset, _models=['DLinear'], _dists=['m'])
    _configs = reconstruct_paths(_configs, _dataset=_dataset, _models=_models, _dists=_dists, _seeds=_seeds)
    load_runner(_configs, first_only=False, batch_size_limit=None, predictions_only=True, _format='.pt', save_pred=False)