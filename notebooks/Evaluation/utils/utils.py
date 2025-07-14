import numpy as np
import matplotlib.pyplot as plt
import torch
import logging
from tqdm import tqdm
from scipy.stats import norm
import time
from torch.nn.parallel import DistributedDataParallel as DDP

__all__ = ["init", "get_predictions"]

def init(model, dataset, path, no_logging=False, cfg_path=None, set_rescale=False):
    if no_logging:
        logging.disable(logging.CRITICAL)  # Temporarily disable all logging
    #print("This will print, but logs are disabled")

    # Run a baseline model in BasicTS framework.
    # pylint: disable=wrong-import-position
    import os
    import sys
    from argparse import ArgumentParser

    try:
        __file__ = "/home/kreffert/Probabilistic_LTSF/BasicTS/experiments/train.py"
        sys.path.append(os.path.abspath(__file__ + '/../..'))
        os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        sys.argv = [__file__]
        prefix = "/home/kreffert"
    except FileNotFoundError:
        __file__ = "/pfs/data6/home/ma/ma_ma/ma_kreffert/Probabilistic_LTSF/BasicTS/experiments/train.py"
        sys.path.append(os.path.abspath(__file__ + '/../..'))
        os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        sys.argv = [__file__]
        prefix = "/pfs/data6/home/ma/ma_ma/ma_kreffert"
    
    import torch
    import basicts
    from easytorch.utils import set_visible_devices, get_logger
    from easytorch.config import init_cfg
    
    torch.set_num_threads(4) # avoid high cpu avg usage

    if cfg_path is None:
        cfg_path = f'baselines/{model}/{dataset}.py'
    
    def parse_args():
        parser = ArgumentParser(description='Run time series forecasting model in BasicTS framework!')
        parser.add_argument('-c', '--cfg', default=cfg_path, help='training config')
        parser.add_argument('-g', '--gpus', default='0', help='visible gpus')
        return parser.parse_args()
    
    args = parse_args()

    cfg = init_cfg(args.cfg, save=True)
    cfg['TRAIN']['RESUME_TRAINING'] = True
    if set_rescale:
        cfg['RESCALE'] = True
        cfg['SCALER']['PARAM']['rescale'] = True
    
    logger = get_logger('easytorch-launcher')
    logger.info('Initializing runner "{}"'.format(cfg['RUNNER']))
    runner = cfg['RUNNER'](cfg)

    distribution_type = cfg['MODEL']['PARAM'].get('distribution_type', None)
    seq_len = cfg['MODEL']['PARAM'].get('seq_len', None)
    pred_len = cfg['MODEL']['PARAM'].get('pred_len', None)
    data_name = cfg['DATASET'].get('NAME', None)
    num_epochs = cfg['TRAIN'].get('NUM_EPOCHS', None)
    
        


    
    # runner.ckpt_save_dir = save_dir
    runner.init_logger(logger_name='easytorch-training', log_file_name=None)



    runner.init_training(cfg)
    if no_logging:
        logging.disable(logging.NOTSET)  # Re-enable logging
        
    if "/" in path:
        save_path = f'{prefix}/{path}'
    else:    
        save_path = f'{prefix}/Probabilistic_LTSF/BasicTS/checkpoints/{distribution_type}_{model}/{data_name}_{num_epochs}_{seq_len}_{pred_len}/{path}'
        
    checkpoint = torch.load(save_path)
    if isinstance(runner.model, DDP):
        runner.model.module.load_state_dict(checkpoint['model_state_dict'], strict=True)
    else:
        runner.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print(checkpoint.keys())
    return runner, cfg

@torch.no_grad()
def validate(runner, cfg= None, train_epoch = None):
    """Validate model.

    Args:
        cfg (Dict, optional): config
        train_epoch (int, optional): current epoch if in training process.
    """

    # init validation if not in training process
    # if train_epoch is None:
    #     self.init_validation(cfg)

    runner.logger.info('Start validation.')

    runner.on_validating_start(train_epoch)

    val_start_time = time.time()
    runner.model.eval()
    
    #tqdm process bar
    data_iter = tqdm(runner.val_data_loader)

    # val loop
    for iter_index, data in enumerate(data_iter):
        runner.val_iters(iter_index, data)

    val_end_time = time.time()
    runner.update_epoch_meter('val/time', val_end_time - val_start_time)
    
    # Now log validation averages to wandb
    if (runner.use_wandb) and (train_epoch is not None):
        # Calculate proper global step at end of validation
        global_step = train_epoch * runner.iter_per_epoch
        
        # Log all validation metrics at once
        wandb_dict = {}            
        # Add metrics from meter pool
        for meter_name in runner.meter_pool._pool.keys():
            if 'val' in meter_name:
                meter_value = runner.meter_pool._pool[meter_name]['meter'].avg
                wandb_dict[f'epoch_summary/{meter_name}'] = meter_value
                wandb_dict[f'{meter_name}'] = meter_value
        wandb.log(wandb_dict, step=global_step)
    
    # print val meters
    runner.print_epoch_meters('val')
    if train_epoch is not None:
        # tensorboard plt meters
        runner.plt_epoch_meters('val', train_epoch // runner.val_interval)

    runner.on_validating_end(train_epoch)

def get_predictions(runner, data_loader="test"):
    if data_loader == "test":
        dataloader = runner.test_data_loader
    elif data_loader == "val":
        dataloader = runner.val_data_loader
    else:
        dataloader = runner.train_data_loader
    model = runner.model
    model.eval()  # Ensure model is in eval mode
    device = 'cuda:0'
    model.to(device)  # Move to appropriate device
    predictions = []
    past_actuals = []
    future_actuals = []
    
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader)):
            history = batch['inputs'][..., :1]     # B, L, N -> i.e. ignore the remaining features
            targets = batch['target'][..., :1]
            
            # TODO scaler
            model_return = runner.forward(batch, epoch=None, iter_num=batch_idx, train=False)
            #forecasts = model(batch['inputs'][..., :1].to(device), batch['target'].to(device), batch_seen=0, epoch=0, train=False)
            history = model_return['inputs']
            targets = model_return['target']
            forecasts = model_return['prediction']
            
            history, targets, forecasts = history.to('cpu'), targets.to('cpu'), forecasts.to('cpu') 
            future_actuals.append(targets)
            past_actuals.append(history)
            predictions.append(forecasts)
    
    # Convert to numpy if needed
    predictions = torch.cat(predictions, dim=0).numpy()
    past_actuals = torch.cat(past_actuals, dim=0).numpy()
    future_actuals = torch.cat(future_actuals, dim=0).numpy()
    return predictions, past_actuals, future_actuals