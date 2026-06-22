import jittor as jt
jt.flags.use_cuda = 1

from omegaconf import OmegaConf
from tqdm import tqdm
from typing import Dict, List

import argparse
import numpy as np
import os
import random

from src.data.asset import Asset, Exporter
from src.data.dataset import DatasetConfig, DatasetConfig, PCDatasetModule
from src.data.transform import Transform
from src.model.parse import get_model
from src.system.parse import get_system, get_writer

def load(task: str, path: str) -> Dict:
    if path.endswith('.yaml'):
        path = path.removesuffix('.yaml')
    path += '.yaml'
    print(f"\033[92mload {task} config: {path}\033[0m")
    return OmegaConf.to_container(OmegaConf.load(path)) # type: ignore

def preflight_train_data(train_dataset_config) -> None:
    """Fail fast with a clear message when training meshes are missing."""
    if train_dataset_config is None:
        return
    dp = train_dataset_config.datapath
    root = dp.input_dataset_dir
    data_name = dp.data_name or ""
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"Training data directory not found: {os.path.abspath(root)}\n"
            "Download A榜训练集 dataset_train.tar.gz and extract under the project root:\n"
            "  tar xzf /path/to/dataset_train.tar.gz\n"
            "Expected layout: dataset_train/shapenet/<synset>/<model_id>/models/model_normalized.obj\n"
            "Or set input_dataset_dir in configs/data/train_*.yaml to your extracted path."
        )
    # probe a few entries from the datalist
    checked = 0
    missing = 0
    for rel in dp.filepaths[:20]:
        mesh_path = os.path.join(root, rel, data_name) if data_name else os.path.join(root, rel)
        checked += 1
        if not os.path.isfile(mesh_path):
            missing += 1
    if missing == checked:
        sample = os.path.join(root, dp.filepaths[0], data_name) if dp.filepaths else root
        raise FileNotFoundError(
            f"No mesh files found under {os.path.abspath(root)} "
            f"(checked {checked} samples, e.g. {sample}).\n"
            "Ensure dataset_train is fully extracted, not only datalist/*.txt."
        )
    if missing > 0:
        print(f"\033[33mWarning: {missing}/{checked} sampled train paths missing under {root}\033[0m")


def debug_fn(data: PCDatasetModule):
    train_dataloader = data.train_dataloader()
    assert train_dataloader is not None, "train_dataloader is None, cannot debug"
    for batch in tqdm(train_dataloader):
        batch: List[Asset]
        # for asset in batch:
        #     Exporter.export_obj(asset.sampled_vertices, "debug.obj")
        #     Exporter.export_obj(asset.sampled_vertices_noisy, "debug_noisy.obj")
        #     exit()

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--seed", type=int, required=False, default=None)
    args = parser.parse_args()
    
    task = load('task', args.task)
    seed = args.seed if args.seed is not None else task.get("seed", 123)
    jt.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"\033[92mseed: {seed}\033[0m")
    mode = task['mode']
    assert mode in ['train', 'predict', 'debug', 'validate']
    components = task['components']
    
    # get train/validate/predict data
    data_config = load('data', os.path.join('configs/data', components['data']))
    
    # get train dataset
    _train_dataset_config = data_config.get('train_dataset', None)
    if _train_dataset_config is not None:
        train_dataset_config = DatasetConfig.parse(**_train_dataset_config)
    else:
        train_dataset_config = None
    
    # get validate dataset
    _validate_dataset_config = data_config.get('validate_dataset', None)
    if _validate_dataset_config is not None:
        validate_dataset_config = DatasetConfig.parse(**_validate_dataset_config).split_by_cls()
    else:
        validate_dataset_config = None
        
    # get predict dataset
    _predict_dataset_config = data_config.get('predict_dataset', None)
    if _predict_dataset_config is not None:
        predict_dataset_config = DatasetConfig.parse(**_predict_dataset_config).split_by_cls()
    else:
        predict_dataset_config = None
    
    # get transform
    transform_config = load('transform', os.path.join('configs/transform', components['transform']))

    # get model
    model_config = components.get('model', None)
    if model_config is None:
        model = None
    else:
        model_config = load('model', os.path.join('configs/model', model_config))
        # stage-wise pretrained weights (task yaml overrides model yaml)
        for key in ('vm_ckpt', 'cvm_ckpt'):
            if task.get(key):
                model_config[key] = task[key]
                print(f"\033[92muse {key}: {task[key]}\033[0m")
        model = get_model(model_config=model_config, transform_config=transform_config)
    
    train_transform = (Transform.parse(**transform_config.get('train_transform', {}))) if model is None else model.get_train_transform()
    validate_transform = (Transform.parse(**transform_config.get('validate_transform', {}))) if model is None else model.get_validate_transform()
    predict_transform = (Transform.parse(**transform_config.get('predict_transform', {}))) if model is None else model.get_predict_transform()
    dataset_module = PCDatasetModule(
        process_fn=None if model is None else model._process_fn,
        train_dataset_config=train_dataset_config,
        validate_dataset_config=validate_dataset_config,
        predict_dataset_config=predict_dataset_config,
        train_transform=train_transform,
        validate_transform=validate_transform,
        predict_transform=predict_transform,
        debug=task.get('debug', False),
    )
    
    optimizer_config = task.get('optimizer', None)
    loss_config = task.get('loss', None)
    trainer_config = task.get('trainer', None)
    
    # load ckpt
    load_ckpt = task.get('load_ckpt', None)
    
    if load_ckpt is not None and model is not None:
        model.load(load_ckpt)
    
    # get writer
    writer_config = task.get('writer', None)
    
    # get system
    system_config = components.get('system', None)
    if system_config is not None:
        system_config = load('system', os.path.join('configs/system', system_config))
        system = get_system(
            dataset_module=dataset_module,
            model=model,
            optimizer_config=optimizer_config,
            loss_config=loss_config,
            trainer_config=trainer_config,
            writer=get_writer(**writer_config) if writer_config is not None else None,
            **system_config,
        )
    else:
        system = None
    
    if mode == 'debug':
        preflight_train_data(train_dataset_config)
        debug_fn(data=dataset_module)
    elif mode == 'train':
        preflight_train_data(train_dataset_config)
        assert system is not None, "system is None, cannot train"
        system.train()
    elif mode == 'predict':
        assert system is not None, "system is None, cannot predict"
        system.predict()
    else:
        assert 0