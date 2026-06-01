#!/usr/bin/env python3
"""对比 noisy / VM-mini / StraightPCF 的点间距与位移（诊断聚类）。"""
import argparse
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree


def mean_nn_dist(pc: np.ndarray, k: int = 6) -> float:
    d, _ = cKDTree(pc).query(pc, k=k + 1)
    return float(d[:, 1:].mean())


def stats(name: str, pc: np.ndarray, ref=None):
    delta = pc - ref if ref is not None else None
    line = (
        f"{name:12s}  N={pc.shape[0]:5d}  "
        f"mean_nn={mean_nn_dist(pc):.6f}  "
        f"mean|r|={np.linalg.norm(pc, axis=1).mean():.4f}"
    )
    if delta is not None:
        line += f"  mean|disp|={np.linalg.norm(delta, axis=1).mean():.6f}"
    print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", default="datalist/test_mini.txt")
    parser.add_argument("--noisy-root", default="/home/cslab/dataset_test_noisy")
    parser.add_argument("--vm-root", default="results_vm_mini")
    parser.add_argument("--spcf-root", default="results_spcf_mini")
    args = parser.parse_args()

    for rel in open(args.list):
        rel = rel.strip()
        if not rel:
            continue
        noisy_p = Path(args.noisy_root) / rel / "noisy.npy"
        vm_p = Path(args.vm_root) / rel / "denoised.npy"
        spcf_p = Path(args.spcf_root) / rel / "denoised.npy"

        print("=" * 72)
        print(rel)
        noisy = np.load(noisy_p)
        stats("noisy", noisy)
        if vm_p.exists():
            vm = np.load(vm_p)
            stats("VM_stage1", vm, noisy)
            tree = cKDTree(vm)
            d, _ = tree.query(vm, k=2)
            print(f"  VM frac_nn<1e-5: {(d[:,1]<1e-5).mean():.4f}")
        else:
            print("  VM_stage1:  (missing)", vm_p)
        if spcf_p.exists():
            sp = np.load(spcf_p)
            stats("StraightPCF", sp, noisy)
            tree = cKDTree(sp)
            d, _ = tree.query(sp, k=2)
            print(f"  SPCF frac_nn<1e-5: {(d[:,1]<1e-5).mean():.4f}")
        else:
            print("  StraightPCF: (missing)", spcf_p)


if __name__ == "__main__":
    main()
