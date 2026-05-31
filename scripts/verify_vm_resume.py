#!/usr/bin/env python3
"""Verify vm_max checkpoint_99 loads and produces sane patch-loss."""
import os
import sys

import jittor as jt
import numpy as np

jt.flags.use_cuda = 1

CG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CG)

CKPT = os.path.join(CG, "experiments/vm_max/checkpoint_99.pkl")
TASK = os.path.join(CG, "configs/task/train_vm_max_resume.yaml")


def main():
    print("=" * 60)
    print("vm_max resume verification")
    print("CG:", CG)
    print("CKPT:", CKPT)
    print("jittor:", jt.__version__, "device:", jt.flags.device)
    print("=" * 60)

    if not os.path.isfile(CKPT):
        print("FAIL: checkpoint missing:", CKPT)
        sys.exit(1)
    print("ckpt size bytes:", os.path.getsize(CKPT))

    try:
        obj = jt.load(CKPT)
        print("jt.load OK, type:", type(obj))
    except Exception as e:
        print("FAIL: jt.load:", e)
        sys.exit(1)

    from omegaconf import OmegaConf
    from src.data.dataset import PCDatasetModule, DatasetConfig
    from src.model.parse import get_model

    task = OmegaConf.to_container(OmegaConf.load(TASK), resolve=True)
    components = task["components"]
    model_cfg = OmegaConf.to_container(
        OmegaConf.load(os.path.join(CG, "configs/model", components["model"] + ".yaml")),
        resolve=True,
    )
    transform_cfg = OmegaConf.to_container(
        OmegaConf.load(os.path.join(CG, "configs/transform", components["transform"] + ".yaml")),
        resolve=True,
    )
    data_cfg = OmegaConf.to_container(
        OmegaConf.load(os.path.join(CG, "configs/data", components["data"] + ".yaml")),
        resolve=True,
    )

    print("model edge_aggr:", model_cfg.get("edge_aggr"))
    print("num_workers (data yaml):", data_cfg["train_dataset"].get("num_workers"))

    model = get_model(model_cfg, transform_cfg)
    model.train()
    model.set_predict(False)

    train_cfg = DatasetConfig.parse(**data_cfg["train_dataset"])
    dm = PCDatasetModule(
        process_fn=model._process_fn,
        train_dataset_config=train_cfg,
        validate_dataset_config=None,
        predict_dataset_config=None,
        train_transform=model.get_train_transform(),
    )
    loader = dm.train_dataloader()
    assert loader is not None

    def batch_loss(m):
        batch = None
        for batch in loader:
            break
        assert batch is not None
        pc_mix = batch["pc_mix"]
        std = float(np.std(pc_mix.numpy() if hasattr(pc_mix, "numpy") else pc_mix))
        out = m.training_step(batch)
        loss = float(out["loss"].item() if hasattr(out["loss"], "item") else out["loss"])
        return loss, std

    loss_random, std = batch_loss(model)
    print(f"\n[random init] first-batch loss={loss_random:.6f}, pc_mix std={std:.6f}")

    model.load(CKPT)
    model.train()
    loss_loaded, std2 = batch_loss(model)
    print(f"[after load]  first-batch loss={loss_loaded:.6f}, pc_mix std={std2:.6f}")

    losses = []
    for i, batch in enumerate(loader):
        if i >= 10:
            break
        out = model.training_step(batch)
        losses.append(float(out["loss"].item()))
    print(f"\n[after load] 10 batches: min={min(losses):.6f} max={max(losses):.6f} mean={sum(losses)/len(losses):.6f}")

    print("\n--- verdict ---")
    mean = sum(losses) / len(losses)
    if 0.15 <= mean <= 0.25:
        print("OK: loss range matches school epoch-99 (~0.19). Weights look fine.")
        print("If tqdm showed zeros, set num_workers=0 and restart.")
    elif mean < 0.001:
        print("BAD: near-zero loss — check edge_aggr, data path, code version.")
    elif mean > 1.0:
        print("BAD: high loss — checkpoint likely NOT loaded. Check load_ckpt path.")
    else:
        print(f"UNCLEAR: mean={mean:.6f}. School baseline ~0.189.")


if __name__ == "__main__":
    main()
