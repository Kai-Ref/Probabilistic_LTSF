import os
import traceback
from typing import Dict, Optional, Union

import easytorch
from easytorch.config import init_cfg
from easytorch.device import set_device_type
from easytorch.utils import get_logger, set_visible_devices
import wandb
os.environ["WANDB_API_KEY"] = "fc6544bc618ba7ae3008cf7c53d9650e1668bf12"
import yaml
import datetime
from basicts.scaler import ZScoreScaler, MinMaxScaler
from copy import deepcopy
from basicts.utils import get_regular_settings


def evaluation_func(cfg: Dict,
                    ckpt_path: str = None,
                    batch_size: Optional[int] = None,
                    strict: bool = True) -> None:
    """
    Starts the evaluation process.

    This function performs the following steps:
    1. Initializes the runner specified in the configuration (`cfg`).
    2. Sets up logging for the evaluation process.
    3. Loads the model checkpoint.
    4. Executes the test pipeline using the initialized runner.

    Args:
        cfg (Dict): EasyTorch configuration dictionary.
        ckpt_path (str): Path to the model checkpoint. If not provided, the best model checkpoint is loaded automatically.
        batch_size (Optional[int]): Batch size for evaluation. If not specified, 
                                    it should be defined in the config. Defaults to None.
        strict (bool): Enforces that the checkpoint keys match the model. Defaults to True.

    Raises:
        Exception: Catches any exception, logs the traceback, and re-raises it.
    """

    # initialize the runner
    logger = get_logger('easytorch-launcher')
    logger.info(f"Initializing runner '{cfg['RUNNER']}'")
    runner = cfg['RUNNER'](cfg)

    # initialize the logger for the runner
    runner.init_logger(logger_name='easytorch-evaluation', log_file_name='evaluation_log')

    # setup the graph if needed
    if runner.need_setup_graph:
        runner.setup_graph(cfg=cfg, train=False)

    try:
        # set batch size if provided
        if batch_size is not None:
            cfg.TEST.DATA.BATCH_SIZE = batch_size
        else:
            assert 'BATCH_SIZE' in cfg.TEST.DATA, 'Batch size must be specified either in the config or as an argument.'

        # load the model checkpoint
        if ckpt_path is None or not os.path.exists(ckpt_path):
            ckpt_path_auto = os.path.join(runner.ckpt_save_dir, '{}_best_val_{}.pt'.format(runner.model_name, runner.target_metrics.replace('/', '_')))
            logger.info(f'Checkpoint file not found at {ckpt_path}. Loading the best model checkpoint `{ckpt_path_auto}` automatically.')
            if not os.path.exists(ckpt_path_auto):
                raise FileNotFoundError(f'Checkpoint file not found at {ckpt_path}')
            runner.load_model(ckpt_path=ckpt_path_auto, strict=strict)
        else:
            logger.info(f'Loading model checkpoint from {ckpt_path}')
            runner.load_model(ckpt_path=ckpt_path, strict=strict)

        # start the evaluation pipeline
        runner.test_pipeline(cfg=cfg, save_metrics=True, save_results=True)

    except BaseException as e:
        # log the exception and re-raise it
        runner.logger.error(traceback.format_exc())
        raise e

def launch_evaluation(cfg: Union[Dict, str],
                      ckpt_path: str,
                      device_type: str = 'gpu',
                      gpus: Optional[str] = None,
                      batch_size: Optional[int] = None) -> None:
    """
    Launches the evaluation process using EasyTorch.

    Args:
        cfg (Union[Dict, str]): EasyTorch configuration as a dictionary or a path to a config file.
        ckpt_path (str): Path to the model checkpoint.
        device_type (str, optional): Device type to use ('cpu' or 'gpu'). Defaults to 'gpu'.
        gpus (Optional[str]): GPU device IDs to use. Defaults to None (use all available GPUs).
        batch_size (Optional[int]): Batch size for evaluation. Defaults to None (use value from config).

    Raises:
        AssertionError: If the batch size is not specified in either the config or as an argument.
    """

    logger = get_logger('easytorch-launcher')
    logger.info('Launching EasyTorch evaluation.')

    # check params
    # cfg path which start with dot will crash the easytorch, just remove dot
    while isinstance(cfg, str) and cfg.startswith(('./','.\\')):
        cfg = cfg[2:]
    while ckpt_path.startswith(('./','.\\')):
        ckpt_path = ckpt_path[2:]

    # initialize the configuration
    cfg = init_cfg(cfg, save=True)

    # set the device type (CPU, GPU, or MLU)
    set_device_type(device_type)

    # set the visible GPUs if the device type is not CPU
    if device_type != 'cpu':
        set_visible_devices(gpus)

    # run the evaluation process
    evaluation_func(cfg, ckpt_path, batch_size)

def launch_training(cfg: Union[Dict, str],
                    gpus: Optional[str] = None,
                    node_rank: int = 0) -> None:
    """
    Launches the training process using EasyTorch.

    Args:
        cfg (Union[Dict, str]): EasyTorch configuration as a dictionary or a path to a config file.
        gpus (Optional[str]): GPU device IDs to use. Defaults to None (use all available GPUs).
        node_rank (int, optional): Rank of the current node in distributed training. Defaults to 0.
    """

    # placeholder for potential pre-processing steps (e.g., model registration, config validation)

    # cfg path which start with dot will crash the easytorch, just remove dot
    while isinstance(cfg, str) and cfg.startswith(('./','.\\')):
        cfg = cfg[2:]

    # launch the training process
    easytorch.launch_training(cfg=cfg, devices=gpus, node_rank=node_rank)

def training_func(cfg: Dict, gpus: Optional[str] = None, node_rank:int = 0, wandb_run=None) -> Dict:
    """
    Starts the training process.

    This function performs the following steps:
    1. Initializes the runner specified in the configuration (`cfg`).
    2. Sets up training according to the configuration.
    3. Executes the training pipeline.
    4. Returns the best metrics if available.

    Args:
        cfg (Dict): EasyTorch configuration dictionary.
        wandb_run: The wandb run object if being used with wandb sweeps.

    Returns:
        Dict: Best metrics from training (if available).

    Raises:
        Exception: Catches any exception, logs the traceback, and re-raises it.
    """
    
    # launch the training process
    def count_devices(devices_str):
        if not devices_str:
            return 0
        # Split by comma and filter out empty strings
        devices_list = [d.strip() for d in devices_str.split(',') if d.strip()]
        return len(devices_list)

    device_count = count_devices(gpus)
    if device_count > cfg['GPU_NUM']:
        print(f"Switching in training_func from {cfg['GPU_NUM']} gpus to {device_count}")
        cfg['GPU_NUM'] = device_count
    easytorch.launch_training(cfg=cfg, devices=gpus, node_rank=node_rank)

    # initialize the runner
    # logger = get_logger('easytorch-launcher')
    # logger.info(f"Initializing runner '{cfg['RUNNER']}'")
    # runner = cfg['RUNNER'](cfg)
    
    # If a wandb run is passed, store it in the runner for use during training
    # if wandb_run is not None:
    #     runner.wandb_run = wandb_run

    # # initialize the logger for the runner
    # runner.init_logger(logger_name='easytorch-training', log_file_name='training_log')

    # # setup the graph if needed
    # if runner.need_setup_graph:
    #     runner.setup_graph(cfg=cfg, train=True)
    
    # best_metrics = {}
    # try:
    #     # start the training pipeline
    #     best_metrics = runner.train(cfg=cfg)
    # except BaseException as e:
    #     # log the exception and re-raise it
    #     runner.logger.error(traceback.format_exc())
    #     raise e
    # return best_metrics

def fill_dependencies(override_config, model_name='PatchTST', data_name='ETTh1'):
    """
    Expands an override config by filling in all dependent keys.
    
    Args:
        override_config: Dictionary with dot-notation keys to override
        model_name: Name of model (currently only 'PatchTST' supported)
        
    Returns:
        Expanded dictionary with all dependent keys filled in
    """
    # Create expanded config
    expanded_config = override_config.copy()

    if "DATA_NAME" in expanded_config.keys():
        data_name = expanded_config['DATA_NAME']
    if "MODEL.NAME" in expanded_config.keys():
        model_name = expanded_config['MODEL.NAME']
    
    regular_settings = get_regular_settings(data_name)
    # Define all dependencies for PatchTST
    if model_name == 'PatchTST':
        dependency_dict = {
            # Core parameters and their dependencies
            'DATA_NAME': [
                'DATASET.NAME',
                'DATASET.PARAM.dataset_name',
                'SCALER.PARAM.dataset_name'
            ],
            'INPUT_LEN': [
                'MODEL.PARAM.seq_len',
                'DATASET.PARAM.input_len'
            ],
            'OUTPUT_LEN': [
                'MODEL.PARAM.pred_len',
                'DATASET.PARAM.output_len'
            ],
            # 'TRAIN_VAL_TEST_RATIO': [
            #     'DATASET.PARAM.train_val_test_ratio',
            #     'SCALER.PARAM.train_ratio'
            # ],
            'NUM_NODES': ['MODEL.PARAM.enc_in'],
            'NUM_EPOCHS': ['TRAIN.NUM_EPOCHS'],
            'NORM_EACH_CHANNEL': ['SCALER.PARAM.norm_each_channel'],
            'RESCALE': ['SCALER.PARAM.rescale'],
            # 'NULL_VAL': ['METRICS.NULL_VAL'],
        }
    elif model_name == 'iTransformer':
        dependency_dict = {
            # Core parameters and their dependencies
            'DATA_NAME': [
                'DATASET.NAME',
                'DATASET.PARAM.dataset_name',
                'SCALER.PARAM.dataset_name'
            ],
            # 'INPUT_LEN': [
            #     'MODEL.PARAM.seq_len',
            #     # 'MODEL.PARAM.label_len',#/2
            #     'DATASET.PARAM.input_len'
            # ],
            'OUTPUT_LEN': [
                'MODEL.PARAM.pred_len',
                'DATASET.PARAM.output_len'
            ],
            # 'TRAIN_VAL_TEST_RATIO': [
            #     'DATASET.PARAM.train_val_test_ratio',
            #     'SCALER.PARAM.train_ratio'
            # ],
            'NUM_NODES': ['MODEL.PARAM.enc_in', 'MODEL.PARAM.dec_in', 'MODEL.PARAM.c_out'],
            'NUM_EPOCHS': ['TRAIN.NUM_EPOCHS'],
            'NORM_EACH_CHANNEL': ['SCALER.PARAM.norm_each_channel'],
            'RESCALE': ['SCALER.PARAM.rescale'],
            # 'NULL_VAL': ['METRICS.NULL_VAL'],
        }
    elif model_name == 'DeepAR':
        dependency_dict = {
            # Core parameters and their dependencies
            'DATA_NAME': [
                'DATASET.NAME',
                'DATASET.PARAM.dataset_name',
                'SCALER.PARAM.dataset_name'
            ],
            'INPUT_LEN': [
                'DATASET.PARAM.input_len'
            ],
            'OUTPUT_LEN': [
                'DATASET.PARAM.output_len'
            ],
            # 'TRAIN_VAL_TEST_RATIO': [
            #     'DATASET.PARAM.train_val_test_ratio',
            #     'SCALER.PARAM.train_ratio'
            # ],
            # 'NUM_NODES': ['MODEL.PARAM.enc_in'],
            'NUM_EPOCHS': ['TRAIN.NUM_EPOCHS'],
            'NORM_EACH_CHANNEL': ['SCALER.PARAM.norm_each_channel'],
            'RESCALE': ['SCALER.PARAM.rescale'],
            # 'NULL_VAL': ['METRICS.NULL_VAL'],
        }
    elif model_name == 'DLinear':
        dependency_dict = {
            # Core parameters and their dependencies
            'DATA_NAME': [
                'DATASET.NAME',
                'DATASET.PARAM.dataset_name',
                'SCALER.PARAM.dataset_name'
            ],
            'INPUT_LEN': [
                'DATASET.PARAM.input_len',
                'MODEL.PARAM.seq_len',
            ],
            'OUTPUT_LEN': [
                'DATASET.PARAM.output_len',
                'MODEL.PARAM.pred_len',
            ],
            # 'TRAIN_VAL_TEST_RATIO': [
            #     'DATASET.PARAM.train_val_test_ratio',
            #     'SCALER.PARAM.train_ratio'
            # ],
            # 'NUM_NODES': ['MODEL.PARAM.enc_in'],
            'NUM_EPOCHS': ['TRAIN.NUM_EPOCHS'],
            'NORM_EACH_CHANNEL': ['SCALER.PARAM.norm_each_channel'],
            'RESCALE': ['SCALER.PARAM.rescale'],
            # 'NULL_VAL': ['METRICS.NULL_VAL'],
        }
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # First pass: handle direct dependencies
    for key in list(override_config.keys()):
        if key in dependency_dict:
            for dep_key in dependency_dict[key]:
                if dep_key not in expanded_config:
                    expanded_config[dep_key] = override_config[key]

    if model_name == 'iTransformer': # handle the list attribute
        # Handle p_hidden_dims based on p_hidden_layers
        if 'MODEL.PARAM.p_hidden_layers' in expanded_config:
            p_layers = expanded_config['MODEL.PARAM.p_hidden_layers']
            # If p_hidden_dims is not explicitly provided
            layer_size = expanded_config.get('MODEL.PARAM.p_hidden_dim_size', 128)
            # Create a list with the same size for each layer
            expanded_config['MODEL.PARAM.p_hidden_dims'] = [layer_size] * p_layers
    if model_name == 'DeepAR':
        number_cov_features = expanded_config['MODEL.PARAM.cov_feat_size']
        expanded_config['MODEL.FORWARD_FEATURES'] = list(range(number_cov_features + 1))
        if 'MODEL.PARAM.prob_args.fixed_qe' in expanded_config:
            expanded_config['MODEL.PARAM.prob_args.fixed_qe'] = int(expanded_config['MODEL.PARAM.hidden_size'])

    if model_name == 'DLinear':
        if 'MODEL.PARAM.prob_args.fixed_qe' in expanded_config:
            if expanded_config['MODEL.PARAM.prob_individual'] is True:
                expanded_config['MODEL.PARAM.prob_args.fixed_qe'] = 96
            else:
                expanded_config['MODEL.PARAM.prob_args.fixed_qe'] = 336

    # specialized overrides
    # CKPT path
    keys_to_check = ['MODEL.PARAM.distribution_type', 'INPUT_LEN', 'OUTPUT_LEN'] 
    if any(key in expanded_config for key in keys_to_check):
        INPUT_LEN = expanded_config['INPUT_LEN'] if 'INPUT_LEN' in expanded_config.keys() else regular_settings['INPUT_LEN']
        OUTPUT_LEN = expanded_config['OUTPUT_LEN'] if 'OUTPUT_LEN' in expanded_config.keys() else regular_settings['OUTPUT_LEN']

        expanded_config['TRAIN.CKPT_SAVE_DIR'] = os.path.join(
                    'checkpoints', f'{expanded_config["MODEL.PARAM.distribution_type"]}_{model_name}',
                    '_'.join([data_name, str(INPUT_LEN), str(OUTPUT_LEN)])
                )
    return expanded_config

def merge_configs(base_config, override_config):
    # First create a deep copy of the base config to avoid modifying it
    merged_config = deepcopy(base_config)

    for key_path, value in override_config.items():
        # Split the dot-notation path into parts
        keys = key_path.split('.')
        
        # Navigate to the correct position in the merged config
        current_level = merged_config
        for key in keys[:-1]:  # all keys except the last one
            if key not in current_level:
                current_level[key] = {}  # create nested dict if it doesn't exist
            current_level = current_level[key]
        
        # Set the value at the final key
        current_level[keys[-1]] = value 
        print(f"Update {keys[-1]} to {value}")   
    return merged_config

def sweep_agent_function(cfg: Dict, gpus: Optional[str] = None, node_rank:int = 0) -> Dict:
    """
    Agent function that runs a single trial for wandb sweep.
    
    Args:
        cfg (Dict): Base EasyTorch configuration dictionary.
        wandb_config (Dict): Configuration from wandb sweep controller.
        
    Returns:
        Dict: Best metrics from training.
    """
    wandb_config = dict(wandb.config)
    logger = get_logger('easytorch-sweep-agent')
    logger.info('Starting sweep agent trial with configuration:')

    if 'SCALER.TYPE' in wandb_config.keys():
        if wandb_config['SCALER.TYPE'] == 'ZScoreScaler':
            wandb_config['SCALER.TYPE'] = ZScoreScaler
        else:
            wandb_config['SCALER.TYPE'] = MinMaxScaler

    # print(cfg)
    # print("-------------------")
    wandb_config = fill_dependencies(wandb_config)
    # print(wandb_config)
    # Update the config with the sweep parameters
    merged_config = merge_configs(cfg, wandb_config)
    # print("-------------------")
    print(merged_config)
    
    logger.info(f"Updated configuration with sweep parameters.")

    training_func(merged_config, gpus=gpus, node_rank=node_rank, wandb_run=wandb.run)
    
    # # Log the best metrics
    # for metric_name, metric_value in best_metrics.items():
    #     wandb.log({f"best_{metric_name}": metric_value})
        
    # return best_metrics

def launch_sweep(cfg: Union[Dict, str],
                 path: str,
                sweep_config: str,
                gpus: Optional[str] = None, 
                node_rank: int = 0) -> None:
    """
    Launches a wandb sweep for hyperparameter optimization.
    
    Args:
        cfg (Union[Dict, str]): Base EasyTorch configuration as a dictionary or a path to a config file.
        project_name (str): Name of the wandb project.
        entity (Optional[str]): wandb entity (username or team name). Defaults to None.
        device_type (str): Device type to use ('cpu' or 'gpu'). Defaults to 'gpu'.
        gpus (Optional[str]): GPU device IDs to use. Defaults to None (use all available GPUs).
        count (int): Number of sweep runs to execute. Defaults to 1.
        
    Returns:
        str: The sweep ID of the created sweep.
    """
    logger = get_logger('easytorch-sweep')
    logger.info('Launching EasyTorch HPO with wandb sweeps.')

    # Load and initialize configuration
    if isinstance(cfg, str):
        # cfg path which start with dot will crash the easytorch, just remove dot
        while cfg.startswith(('./','.\\')):
            cfg = cfg[2:]
        cfg = init_cfg(cfg, save=True)

    entity="kai-reffert-university-mannheim"
    project = "Prob_LTSF"
    if path is not None:
        with open(f"{path}", "r") as f:
            sweep_config = yaml.safe_load(f)
        sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
        logger.info(f"Created sweep with ID: {sweep_id}")
    else: # as soon as you have a sweep in which you want to try out more runs, replace the last sweep_id below
        sweep_id = f'kai-reffert-university-mannheim/Prob_LTSF/{sweep_config}'
    
    # Create a function that will be passed to wandb.agent
    def agent_function():
        # Initialize a wandb run which will create a new run and populate wandb.config
        timestamp = datetime.datetime.now().strftime("%b_%d_%H_%M")
        with wandb.init(project=project, group='HPO', entity=entity, name=f"{str(cfg.MODEL.NAME)}_{timestamp}") as run:
            # Use the config that wandb populated automatically
            return sweep_agent_function(cfg, gpus=gpus, node_rank=node_rank)
    
    # Start the sweep agententity="kai-reffert-university-mannheim",
    wandb.agent(sweep_id, function=agent_function, project=project, entity=entity) #count=count)



# def training_func(cfg: Dict, gpus: Optional[str] = None, node_rank: int = 0, wandb_run=None) -> Dict:
#     import os
#     import torch.distributed as dist
#     from easytorch.launcher.dist_wrap import dist_wrap
#     from easytorch.utils import get_logger

#     def count_devices(devices_str):
#         if not devices_str:
#             return 0
#         return len([d.strip() for d in devices_str.split(',') if d.strip()])

#     device_count = count_devices(gpus)
#     if device_count > 0 and device_count != cfg.get('GPU_NUM', 0):
#         logger = get_logger('easytorch-sweep-agent')
#         logger.info(f"Switching in training_func from {cfg.get('GPU_NUM', 0)} gpus to {device_count}")
#         cfg['GPU_NUM'] = device_count

#     from functools import partial

#     world_size = cfg.get('GPU_NUM', 1)

#     dist_train = dist_wrap(
#         partial(wrapped_training_func, world_size=world_size, cfg=cfg, wandb_run=wandb_run),
#         node_num=cfg.get('DIST_NODE_NUM', 1),
#         device_num=world_size,
#         node_rank=node_rank,
#         dist_backend=cfg.get('DIST_BACKEND', 'nccl'),
#         init_method=cfg.get('DIST_INIT_METHOD', None)
#     )

#     return dist_train(cfg)



# # At the top level of the file (same as training_func)
# def wrapped_training_func(rank: int, world_size: int, cfg: Dict, wandb_run):
#     import wandb
#     import os
#     from easytorch.utils import get_logger

#     # Only initialize wandb in the main process (rank 0)
#     if rank == 0 and cfg.get('USE_WANDB', False):
#         os.environ["WANDB_SPAWN_METHOD"] = "thread"
#         if wandb_run is None and not wandb.run:
#             wandb.init(
#                 project=cfg.get('WANDB_PROJECT', 'Prob_LTSF'),
#                 config=cfg,
#                 reinit=True
#             )
#         elif wandb_run is not None and not wandb.run:
#             wandb.init(
#                 id=wandb_run.id,
#                 resume="allow",
#                 project=cfg.get('WANDB_PROJECT', 'Prob_LTSF'),
#                 config=cfg
#             )

#     logger = get_logger('easytorch-launcher')
#     logger.info(f"Initializing runner '{cfg['RUNNER']}'")
#     runner = cfg['RUNNER'](cfg)
#     runner.init_logger(logger_name='easytorch-training', log_file_name='training_log')
#     runner.rank = rank
#     print(f"MY RUNNERS RANK {runner.rank}")

#     if hasattr(runner, 'need_setup_graph') and runner.need_setup_graph:
#         runner.setup_graph(cfg=cfg, train=True)

#     if rank == 0 and cfg.get('USE_WANDB', False):
#         runner.wandb_run = wandb.run

#     best_metrics = {}
#     try:
#         best_metrics = runner.train(cfg=cfg)
#         if rank == 0 and cfg.get('USE_WANDB', False) and wandb.run:
#             for metric_name, metric_value in best_metrics.items():
#                 wandb.log({f"best_{metric_name}": metric_value})
#     except Exception as e:
#         import traceback
#         runner.logger.error(traceback.format_exc())
#         raise e

#     return best_metrics

