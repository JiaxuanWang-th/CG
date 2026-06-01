#!/usr/bin/env python3
"""从 validate.txt 网格导出本地评测集（clean.npy + noisy.npy）。

与 A 榜测试集一致：50000 点、Laplace 噪声。
mesh 使用 ShapeNet model_normalized.obj（已在单位球内），采样后不再二次归一化，否则 P2S 与 mesh 坐标系错位。
每个样本用固定随机种子，便于可重复对比不同 checkpoint。

输出目录结构（供 evaluate.py 使用）：
  <out_dir>/shapenet/<synset>/<model_id>/clean.npy
  <out_dir>/shapenet/<synset>/<model_id>/noisy.npy
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.utils import sample_vertex_groups  # noqa: E402


def _seed_from_rel(rel: str, base_seed: int) -> int:
    h = hashlib.md5(f"{base_seed}:{rel}".encode()).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)


def _export_one(
    rel: str,
    mesh_root: Path,
    out_root: Path,
    num_samples: int,
    num_vertex_samples: int,
    noise_std_min: float,
    noise_std_max: float,
    base_seed: int,
) -> tuple[str, str | None]:
    """Returns (rel, error_message)."""
    mesh_path = mesh_root / rel / "models" / "model_normalized.obj"
    if not mesh_path.is_file():
        return rel, f"missing mesh: {mesh_path}"

    rng = np.random.default_rng(_seed_from_rel(rel, base_seed))
    np.random.seed(_seed_from_rel(rel, base_seed))

    mesh = trimesh.load(str(mesh_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    vertices = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int64)

    clean, _, _, _ = sample_vertex_groups(
        vertices=vertices,
        faces=faces,
        num_samples=num_samples,
        num_vertex_samples=num_vertex_samples,
    )
    # model_normalized.obj 已在单位球内；勿再 normalize，否则 evaluate 的 P2S 与 mesh 不对齐

    noise_std = float(rng.uniform(noise_std_min, noise_std_max))
    noise = rng.laplace(0.0, noise_std, size=clean.shape)
    noisy = (clean + noise).astype(np.float32)

    out_dir = out_root / rel
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "clean.npy", clean.astype(np.float32))
    np.save(out_dir / "noisy.npy", noisy)
    return rel, None


def main():
    parser = argparse.ArgumentParser(description="Export validation-set local benchmark")
    parser.add_argument(
        "--list",
        type=Path,
        default=ROOT / "datalist/validate.txt",
        help="Datalist (one shapenet/... path per line)",
    )
    parser.add_argument(
        "--mesh-root",
        type=Path,
        default=Path("/home/dataset_train"),
        help="Training meshes root (dataset_train)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "benchmark_val",
        help="Output root (contains shapenet/.../clean.npy & noisy.npy)",
    )
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--num-vertex-samples", type=int, default=1024)
    parser.add_argument("--noise-std-min", type=float, default=0.005)
    parser.add_argument("--noise-std-max", type=float, default=0.020)
    parser.add_argument(
        "--fixed-noise-std",
        type=float,
        default=None,
        help="If set, use fixed noise std (e.g. 0.015) instead of uniform range",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if args.fixed_noise_std is not None:
        args.noise_std_min = args.noise_std_max = args.fixed_noise_std

    rels = [ln.strip() for ln in args.list.read_text().splitlines() if ln.strip()]
    if not rels:
        print("Empty datalist.", file=sys.stderr)
        sys.exit(1)

    todo = []
    for rel in rels:
        out_clean = args.out / rel / "clean.npy"
        if args.skip_existing and out_clean.is_file():
            continue
        todo.append(rel)

    print(
        f"Export {len(todo)}/{len(rels)} samples -> {args.out}\n"
        f"  points={args.num_samples}, noise=[{args.noise_std_min}, {args.noise_std_max}], "
        f"laplace, seed={args.seed}"
    )

    job_args = (
        args.mesh_root,
        args.out,
        args.num_samples,
        args.num_vertex_samples,
        args.noise_std_min,
        args.noise_std_max,
        args.seed,
    )

    failed = []
    if args.workers <= 1:
        for rel in todo:
            _, err = _export_one(rel, *job_args)
            if err:
                failed.append((rel, err))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_export_one, rel, *job_args): rel for rel in todo}
            for fut in as_completed(futs):
                rel, err = fut.result()
                if err:
                    failed.append((rel, err))

    if failed:
        print(f"\nFailed {len(failed)}:")
        for rel, err in failed[:10]:
            print(f"  {rel}: {err}")
        sys.exit(1)

    print(f"Done. {len(rels)} samples under {args.out}")
    print(
        "\nLocal score:\n"
        f"  python evaluate.py \\\n"
        f"    --pred_dir <your_denoised_dir> \\\n"
        f"    --gt_dir {args.out} \\\n"
        f"    --noisy_dir {args.out} \\\n"
        f"    --mesh_dir {args.mesh_root} \\\n"
        f"    --gt_filename clean.npy --noisy_filename noisy.npy \\\n"
        f"    --workers 8 --verbose"
    )


if __name__ == "__main__":
    main()
