"""Patch-based denoising utilities (StraightPCF inference pipeline)."""
from math import ceil
from typing import Protocol

import jittor as jt


class DenoiseDynamics(Protocol):
    def denoise_langevin_dynamics(self, pcl_noisy, num_steps: int = 4):
        ...


def farthest_point_sampling(pcls, num_pnts):
    """pcls: (B, N, 3) -> sampled (B, num_pnts, 3), indices (B, num_pnts)"""
    B, N, _ = pcls.shape
    sampled = []
    indices = []
    for b in range(B):
        pts = pcls[b]
        selected = []
        dist = jt.ones((N,)) * 1e10
        farthest = 0
        for _ in range(num_pnts):
            selected.append(farthest)
            centroid = pts[farthest]
            d = ((pts - centroid) ** 2).sum(dim=1)
            dist = jt.minimum(dist, d)
            farthest, _ = jt.argmax(dist, dim=-1)
            farthest = farthest.item()
        idx = jt.array(selected).int32()
        sampled.append(pts[idx][None, ...])
        indices.append(idx[None, ...])
    return jt.concat(sampled, dim=0), jt.concat(indices, dim=0)


def knn_points(x, y, k):
    """x: (B, P, 3), y: (B, N, 3) -> dist (B,P,k), idx (B,P,k), nn (B,P,k,3)"""
    dist = ((x.unsqueeze(2) - y.unsqueeze(1)) ** 2).sum(-1)
    dist_k, idx = jt.topk(dist, k=k, dim=-1, largest=False)
    B = x.shape[0]
    nn = jt.stack([y[b][idx[b]] for b in range(B)], dim=0)
    return dist_k, idx, nn


def patch_based_denoise(
    model: DenoiseDynamics,
    pcl_noisy,
    patch_size: int = 1000,
    seed_k: int = 6,
    seed_k_alpha: int = 1,
    preserve_point_count: bool = True,
) -> jt.Var:
    """
    FPS seeds + KNN patches + 加权拼接（与官方 StraightPCF 一致）。
    pcl_noisy: (N, 3)，应为单位球归一化坐标。
    """
    assert len(pcl_noisy.shape) == 2
    N, _ = pcl_noisy.shape
    num_patches = int(seed_k * N / patch_size)
    pcl_batch = pcl_noisy.unsqueeze(0)

    seed_pnts, _ = farthest_point_sampling(pcl_batch, num_patches)
    patch_dists, point_idxs, patches = knn_points(seed_pnts, pcl_batch, patch_size)

    patches = patches[0]
    patch_dists = patch_dists[0]
    point_idxs = point_idxs[0]

    seed_expand = seed_pnts.squeeze().unsqueeze(1).broadcast(patches.shape)
    patches = patches - seed_expand

    patch_dists = patch_dists / (patch_dists[:, -1:].broadcast(patch_dists.shape) + 1e-8)

    all_dists = jt.ones((num_patches, N)) * 1e10
    for i in range(num_patches):
        all_dists[i][point_idxs[i]] = patch_dists[i]

    weights = jt.exp(-all_dists)
    best_weights_idx, _ = jt.argmax(weights, dim=0)

    patches_denoised = []
    i = 0
    patch_step = int(ceil(N / (seed_k_alpha * patch_size)))
    assert patch_step > 0
    while i < num_patches:
        curr = patches[i : i + patch_step]
        out, _ = model.denoise_langevin_dynamics(curr)
        patches_denoised.append(out)
        i += patch_step

    patches_denoised = jt.concat(patches_denoised, dim=0)
    patches_denoised = patches_denoised + seed_expand

    pcl_out = []
    for pidx in range(N):
        patch_id = best_weights_idx[pidx].item()
        mask = point_idxs[patch_id] == pidx
        if int(mask.sum().item()) > 0:
            pcl_out.append(patches_denoised[patch_id][mask])
        elif preserve_point_count:
            pcl_out.append(pcl_batch[0, pidx : pidx + 1, :])
        else:
            raise RuntimeError(f"Point {pidx} not covered by any patch")
    pcl_out = jt.concat(pcl_out, dim=0)
    assert pcl_out.shape[0] == N
    return pcl_out
