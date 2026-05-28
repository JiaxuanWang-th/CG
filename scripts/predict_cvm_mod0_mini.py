#!/usr/bin/env python3
"""
可选消融脚本（不影响 run.py 主流程）。
CVM module-0 only + VM 4-step；5 万点 + Jittor FPS 可能很慢，可跳过。
主流程请用: python run.py --task configs/task/predict_cvm_mini.yaml
"""
import sys
import time
from pathlib import Path

import jittor as jt
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model.denoise_utils import patch_based_denoise
from src.model.vm import VelocityModule

MODEL_CFG = dict(
    frame_knn=32,
    num_train_points=128,
    feat_embedding_dim=256,
    decoder_hidden_dim=64,
    dsm_sigma=0.01,
    patch_size=1000,
    seed_k=6,
    seed_k_alpha=1,
    niters=1,
)
TRANSFORM_CFG = {"predict_transform": {"augments": [{"__target__": "normalize_noisy_pc"}]}}


def normalize_noisy(pc: np.ndarray) -> np.ndarray:
    c = pc.mean(0)
    pc = pc - c
    r = np.linalg.norm(pc, axis=1).max()
    return pc / (r + 1e-8)


def mean_nn(pc: np.ndarray) -> float:
    d, _ = cKDTree(pc).query(pc, k=7)
    return float(d[:, 1:].mean())


def load_mod0_as_vm(cvm_ckpt: Path) -> VelocityModule:
    """Load CVM module-0 weights into a VelocityModule (same inference as VM mini)."""
    vm = VelocityModule(MODEL_CFG, TRANSFORM_CFG)
    state = jt.load(str(cvm_ckpt))
    mod0 = {
        k[len("velocity_nets.0.") :]: v
        for k, v in state.items()
        if k.startswith("velocity_nets.0.")
    }
    vm.load_parameters(mod0)
    return vm


def main():
    cvm_ckpt = ROOT / "experiments/cvm/checkpoint_39.pkl"
    print(f"Loading module 0 from {cvm_ckpt} ...", flush=True)
    t0 = time.time()
    vm = load_mod0_as_vm(cvm_ckpt)
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

    noisy_root = Path("/home/cslab/dataset_test_noisy")
    out_root = ROOT / "results_cvm_mod0_mini"
    list_path = ROOT / "datalist/test_mini.txt"

    for rel in open(list_path):
        rel = rel.strip()
        if not rel:
            continue
        print(f"\n[{rel}] N=50000 denoise ...", flush=True)
        t1 = time.time()
        noisy = np.load(noisy_root / rel / "noisy.npy").astype(np.float32)
        noisy_n = normalize_noisy(noisy)
        pc = jt.array(noisy_n)
        den = patch_based_denoise(vm, pc, patch_size=1000, seed_k=6).numpy()
        print(f"  done in {time.time() - t1:.1f}s", flush=True)

        out_p = out_root / rel / "denoised.npy"
        out_p.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_p, den)

        disp = np.linalg.norm(den - noisy_n, axis=1).mean()
        print(f"  mod0-only  mean_nn={mean_nn(den):.6f}  mean|disp|={disp:.6f}", flush=True)

    print(f"\nSaved -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
