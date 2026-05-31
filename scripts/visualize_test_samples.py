#!/usr/bin/env python3
"""Visualize test-set noisy vs CVM denoised (side-by-side PNG + PLY)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SAMPLES = [
    "shapenet/04379243/d78d509ada047f34e1a714ee619465a",  # table (test 主类)
    "shapenet/02691156/46d4d453ceac2f5c3c3b254d8683a766",  # airplane
    "shapenet/04256520/150da8f39b055ad0b827fae7748988f",  # sofa
    "shapenet/03642806/689667fca044e15b941a99145cf33e72",  # knife
]


def load_npy(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float64)


def subsample(pc: np.ndarray, n: int, seed: int) -> np.ndarray:
    if pc.shape[0] <= n:
        return pc
    rng = np.random.default_rng(seed)
    idx = rng.choice(pc.shape[0], size=n, replace=False)
    return pc[idx]


def save_ply(path: Path, pc: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pc.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for x, y, z in pc:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def plot_pair(noisy: np.ndarray, denoised: np.ndarray, title: str, out_png: Path, n_vis: int) -> None:
    n1 = subsample(noisy, n_vis, 0)
    n2 = subsample(denoised, n_vis, 0)
    fig = plt.figure(figsize=(12, 5))
    for i, (pc, name) in enumerate([(n1, "noisy"), (n2, "denoised (CVM max e39)")], start=1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=0.2, c=pc[:, 2], cmap="viridis", linewidths=0)
        ax.set_title(name)
        ax.set_axis_off()
        lim = np.max(np.abs(pc)) * 1.05
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noisy-root", type=Path, default=Path("/home/cslab/dataset_test_noisy"))
    parser.add_argument("--pred-root", type=Path, default=ROOT / "results_cvm_max_test")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "viz_cvm_max_test_epoch39")
    parser.add_argument("--samples", nargs="*", default=DEFAULT_SAMPLES)
    parser.add_argument("--n-vis", type=int, default=8000)
    parser.add_argument("--n-ply", type=int, default=20000)
    args = parser.parse_args()

    for rel in args.samples:
        rel = rel.strip("/")
        noisy_path = args.noisy_root / rel / "noisy.npy"
        den_path = args.pred_root / rel / "denoised.npy"
        if not noisy_path.is_file():
            raise FileNotFoundError(noisy_path)
        if not den_path.is_file():
            raise FileNotFoundError(den_path)

        noisy = load_npy(noisy_path)
        denoised = load_npy(den_path)
        tag = rel.replace("/", "_")
        out_base = args.out_dir / tag
        save_ply(out_base.with_name(tag + "_noisy.ply"), subsample(noisy, args.n_ply, 1))
        save_ply(out_base.with_name(tag + "_denoised.ply"), subsample(denoised, args.n_ply, 2))
        plot_pair(noisy, denoised, rel, out_base.with_suffix(".png"), args.n_vis)
        print(f"OK {rel}  noisy={noisy.shape} denoised={denoised.shape} -> {out_base.with_suffix('.png')}")

    print(f"\nOutputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
