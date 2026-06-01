from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from scipy.spatial import cKDTree
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .asset import Asset
from .spec import ConfigSpec
from .utils import random_euler_rotation, sample_vertex_groups, normalize_unit_sphere

@dataclass(frozen=True)
class Augment(ConfigSpec):
    
    @classmethod
    @abstractmethod
    def parse(cls, **kwags) -> 'Augment':
        pass
    
    @abstractmethod
    def apply(self, asset: Asset, **kwargs):
        pass

@dataclass(frozen=True)
class AugmentSample(Augment):
    
    num_samples: int # total number of vertices on the face to be sampled
    
    num_vertex_samples: int=0 # number of vertices to be chosen
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentSample':
        cls.check_keys(kwargs)
        return AugmentSample(**kwargs)
    
    def apply(self, asset: Asset, **kwargs):
        assert asset.vertices is not None
        assert asset.faces is not None
        sampled_vertices, sampled_normals, sampled_vertex_groups, hidden_states = sample_vertex_groups(
            vertices=asset.vertices,
            faces=asset.faces,
            num_samples=self.num_samples,
            num_vertex_samples=self.num_vertex_samples,
        )
        asset.sampled_vertices = sampled_vertices

@dataclass(frozen=True)
class AugmentNormalizePC(Augment):
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentNormalizePC':
        cls.check_keys(kwargs)
        return AugmentNormalizePC(**kwargs)
    
    def apply(self, asset: Asset, **kwargs):
        pc = asset.sampled_vertices
        assert pc is not None, "sampled_vertices is None, cannot apply AugmentNormalizePC"
        pc, center, scale = normalize_unit_sphere(pc)
        asset.sampled_vertices = pc

@dataclass(frozen=True)
class AugmentNormalizeNoisyPC(Augment):
    """推理：对 noisy.npy 做单位球归一化，并记录 center/scale 供反归一化。"""

    @classmethod
    def parse(cls, **kwargs) -> 'AugmentNormalizeNoisyPC':
        cls.check_keys(kwargs)
        return AugmentNormalizeNoisyPC(**kwargs)

    def apply(self, asset: Asset, **kwargs):
        pc = asset.sampled_vertices_noisy
        assert pc is not None, "sampled_vertices_noisy is None, cannot apply AugmentNormalizeNoisyPC"
        pc, center, scale = normalize_unit_sphere(pc)
        asset.sampled_vertices_noisy = pc
        asset.norm_center = center
        asset.norm_scale = float(scale)

@dataclass(frozen=True)
class AugmentAddNoise(Augment):
    
    noise_std_min: float
    
    noise_std_max: float

    # official StraightPCF uses Gaussian; competition baseline uses laplace
    distribution: str = "laplace"
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentAddNoise':
        cls.check_keys(kwargs)
        return AugmentAddNoise(**kwargs)

    def _sample_noise(self, std: float, shape):
        if self.distribution == "gaussian":
            return np.random.normal(0, std, size=shape)
        if self.distribution == "laplace":
            return np.random.laplace(0, std, size=shape)
        raise ValueError(f"unknown noise distribution: {self.distribution}")
    
    def apply(self, asset: Asset, **kwargs):
        pc = asset.sampled_vertices
        assert pc is not None, "sampled_vertices is None, cannot apply AugmentAddNoise"
        noise_std = np.random.uniform(self.noise_std_min, self.noise_std_max)
        noise = self._sample_noise(noise_std, pc.shape)
        asset.sampled_vertices_noisy = pc + noise
        # L1/L2 noise levels for StraightPCF CVM & distance-head training
        noise_l1 = self._sample_noise(self.noise_std_min, pc.shape)
        noise_l2 = self._sample_noise(self.noise_std_max, pc.shape)
        asset.sampled_vertices_noisy_l1 = pc + noise_l1
        asset.sampled_vertices_noisy_l2 = pc + noise_l2

@dataclass(frozen=True)
class AugmentLinear(Augment):
    
    scale: Tuple[float, float]=(1.0, 1.0)
    
    rotate_x_range: Tuple[float, float]=(0.0, 0.0)
    
    rotate_y_range: Tuple[float, float]=(0.0, 0.0)
    
    rotate_z_range: Tuple[float, float]=(0.0, 0.0)
    
    scale_p: float=0.0
    
    rotate_p: float=0.0
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentLinear':
        cls.check_keys(kwargs)
        return AugmentLinear(**kwargs)
    
    def apply(self, asset: Asset, **kwargs):
        trans_vertex = np.eye(4, dtype=np.float32)
        if np.random.rand() < self.rotate_p:
            r = random_euler_rotation(
                1,
                x_range=self.rotate_x_range,
                y_range=self.rotate_y_range,
                z_range=self.rotate_z_range,
            )[0]
            trans_vertex = r @ trans_vertex
        if np.random.rand() < self.scale_p:
            scale = np.zeros((4, 4), dtype=np.float32)
            scale[0, 0] = np.random.uniform(self.scale[0], self.scale[1])
            scale[1, 1] = np.random.uniform(self.scale[0], self.scale[1])
            scale[2, 2] = np.random.uniform(self.scale[0], self.scale[1])
            scale[3, 3] = 1.0
            trans_vertex = scale @ trans_vertex
        asset.transform(trans_vertex)

@dataclass(frozen=True)
class AugmentPatch(Augment):
    
    patch_size: int
    
    num_patches: int
    
    train_cvm_network: bool
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentPatch':
        cls.check_keys(kwargs)
        return AugmentPatch(**kwargs)
    
    def apply(self, asset: Asset, **kwargs):
        pc = asset.sampled_vertices
        assert pc is not None

        # StraightPCF uses L2 (max noise) for patch extraction; fallback to single noise level
        pc_l2 = getattr(asset, "sampled_vertices_noisy_l2", None)
        if pc_l2 is None:
            pc_l2 = asset.sampled_vertices_noisy
        assert pc_l2 is not None

        N = pc_l2.shape[0]
        seed_idx = np.random.permutation(N)[: self.num_patches]
        seed_points = pc_l2[seed_idx]

        tree = cKDTree(pc_l2)
        _, nn_idx = tree.query(seed_points, k=self.patch_size)

        pat_a = pc_l2[nn_idx]
        pat_b = pc[nn_idx]

        if asset.meta is None:
            asset.meta = {}

        if not self.train_cvm_network:
            l1, l2 = 1e-8, 1.0
            t = np.random.rand(self.num_patches, self.patch_size, 1)
            t = (l2 - l1) * t + l1

            pat_t = t * pat_b + (1 - t) * pat_a
            seed_points_t = (
                t[:, 0:1, :] * pc[seed_idx][:, None, :]
                + (1 - t[:, 0:1, :]) * pc_l2[seed_idx][:, None, :]
            )

            pat_a = pat_a - seed_points_t
            pat_b = pat_b - seed_points_t
            pat_t = pat_t - seed_points_t

            asset.meta["pc_noisy"] = pat_a
            asset.meta["pc_clean"] = pat_b
            asset.meta["pc_mix"] = pat_t
        else:
            # CVM / StraightPCF: scalar time step per patch (same as official StraightPCF)
            t_scalar = np.random.rand() * (1.0 - 1e-8) + 1e-8
            t = np.full((self.num_patches, self.patch_size, 1), t_scalar)
            # (P, M, 3) — per-point interpolated seed, not (P, 1, 3)
            seed_points_t = (
                t * pc[seed_idx][:, None, :]
                + (1 - t) * pc_l2[seed_idx][:, None, :]
            )

            asset.meta["pc_noisy_l2"] = pat_a
            asset.meta["pc_clean"] = pat_b
            asset.meta["seed_points_t"] = seed_points_t
            asset.meta["original_time_step"] = np.full(
                (self.num_patches,), t_scalar, dtype=np.float32
            )

def get_augments(*args) -> List[Augment]:
    MAP = {
        "sample": AugmentSample,
        "normalize_pc": AugmentNormalizePC,
        "normalize_noisy_pc": AugmentNormalizeNoisyPC,
        "add_noise": AugmentAddNoise,
        "linear": AugmentLinear,
        "patch": AugmentPatch,
    }
    MAP: Dict[str, type[Augment]]
    augments = []
    for (i, config) in enumerate(args):
        __target__ = config.get('__target__')
        assert __target__ is not None, f"do not find `__target__` in augment of position {i}"
        c = deepcopy(config)
        del c['__target__']
        augments.append(MAP[__target__].parse(**c))
    return augments