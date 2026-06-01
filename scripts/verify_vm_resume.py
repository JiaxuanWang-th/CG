"""Preflight check: load checkpoint_99 and measure train patch loss on 10 batches."""
import os
import sys

CG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(CG)
sys.path.insert(0, CG)

import jittor as jt

jt.flags.use_cuda = 1

from omegaconf import OmegaConf

from src.data.dataset import DatasetConfig, PCDatasetModule
from src.data.transform import Transform
from src.model.parse import get_model
from src.system.spec import _get_item

CKPT = os.path.join(CG, "experiments/vm_max/checkpoint_99.pkl")
NUM_BATCHES = 10


def load_yaml(path: str):
    if not path.endswith(".yaml"):
        path += ".yaml"
    return OmegaConf.to_container(OmegaConf.load(path))


def main():
    if not os.path.isfile(CKPT):
        print(f"checkpoint missing: {CKPT}")
        sys.exit(1)

    task = load_yaml("configs/task/train_vm_max_resume.yaml")
    components = task["components"]
    data_config = load_yaml(os.path.join("configs/data", components["data"]))
    transform_config = load_yaml(os.path.join("configs/transform", components["transform"]))
    model_config = load_yaml(os.path.join("configs/model", components["model"]))

    train_dataset_config = DatasetConfig.parse(**data_config["train_dataset"])
    model = get_model(model_config=model_config, transform_config=transform_config)
    train_transform = model.get_train_transform()
    dataset_module = PCDatasetModule(
        process_fn=model._process_fn,
        train_dataset_config=train_dataset_config,
        validate_dataset_config=None,
        predict_dataset_config=None,
        train_transform=train_transform,
        validate_transform=None,
        predict_transform=None,
        debug=False,
    )

    print(f"loading checkpoint: {CKPT}")
    model.load(CKPT)
    model.set_predict(False)
    model.train()

    loss_config = task["loss"]
    train_dataloader = dataset_module.train_dataloader()
    assert train_dataloader is not None

    losses = []
    for i, batch in enumerate(train_dataloader):
        if i >= NUM_BATCHES:
            break
        with jt.no_grad():
            loss_dict = model.training_step(batch)
        loss_sum = 0.0
        for name in loss_dict:
            if loss_config.get(name, 0) > 0:
                loss_sum += loss_config[name] * _get_item(loss_dict[name])
        losses.append(float(loss_sum))
        print(f"  batch {i}: {losses[-1]:.6f}")

    mean = sum(losses) / len(losses)
    print(f"[after load] {len(losses)} batches mean: {mean:.6f}")
    if 0.17 <= mean <= 0.22:
        print("PASS (expected 0.17~0.22)")
    else:
        print("WARN: mean outside 0.17~0.22 — check env/deps before training")
        sys.exit(2)


if __name__ == "__main__":
    main()
