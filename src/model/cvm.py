"""Coupled Velocity Modules (CVM) — StraightPCF stage 2."""
from typing import Dict, List, Optional

import jittor as jt
from jittor import nn

from .denoise_utils import patch_based_denoise
from .spec import ModelSpec
from .vm import VelocityModule, get_random_indices


class CoupledVMArch(ModelSpec):
    """Stack of coupled VelocityModules with consistency loss."""

    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        cfg = self.model_config

        self.frame_knn = cfg["frame_knn"]
        self.num_train_points = cfg["num_train_points"]
        self.dsm_sigma = cfg["dsm_sigma"]
        self.tot_its = cfg.get("tot_its", 3)
        self.num_modules = cfg.get("num_modules", 2)

        vm_ckpt = cfg.get("vm_ckpt", None)
        self.velocity_nets = nn.ModuleList()
        for i in range(self.num_modules):
            vm = VelocityModule(model_config, transform_config)
            if vm_ckpt is not None:
                vm.load(vm_ckpt)
            self.velocity_nets.append(vm)

    def get_supervised_loss(
        self,
        pc_clean,
        pc_noisy_l2,
        seed_points_t,
        original_time_step,
    ):
        B, N, d = pc_noisy_l2.shape
        grad_target = pc_clean - pc_noisy_l2

        total_dir_loss = jt.array(0.0)
        total_consistency_loss = jt.array(0.0)

        curr_step = (original_time_step * (self.num_modules - 0) + 0) / self.num_modules
        curr_step = curr_step.reshape(B, 1, 1)
        pc_noisy = curr_step * pc_clean + (1 - curr_step) * pc_noisy_l2
        pc_noisy = pc_noisy - seed_points_t

        for mod in range(self.num_modules):
            feat = self.velocity_nets[mod].encoder(pc_noisy)
            F_dim = feat.shape[2]
            pred_dir = self.velocity_nets[mod].decoder(
                c=feat.reshape(-1, F_dim)
            ).reshape(B, N, d)
            dir_loss = ((pred_dir - grad_target) ** 2).sum(dim=-1).mean()
            total_dir_loss = total_dir_loss + dir_loss

            step_scale = (1.0 - original_time_step.reshape(B, 1, 1)) / self.num_modules
            pc_noisy = pc_noisy + step_scale * pred_dir

            if mod < self.num_modules - 1:
                curr_step_plus_1 = (
                    original_time_step * (self.num_modules - (mod + 1)) + (mod + 1)
                ) / self.num_modules
                curr_step_plus_1 = curr_step_plus_1.reshape(B, 1, 1)
                pc_interp = curr_step_plus_1 * pc_clean + (1 - curr_step_plus_1) * pc_noisy_l2
                pc_interp = pc_interp - seed_points_t
                consistency_loss = ((pc_interp - pc_noisy) ** 2).sum(dim=-1).mean()
                total_consistency_loss = total_consistency_loss + consistency_loss

        return (total_dir_loss + 10 * total_consistency_loss) / self.dsm_sigma

    def denoise_langevin_dynamics(self, pcl_noisy, num_steps: int = 4):
        B, N, d = pcl_noisy.shape
        with jt.no_grad():
            pcl_next = pcl_noisy.clone()
            for _ in range(self.tot_its):
                for mod in range(self.num_modules):
                    feat = self.velocity_nets[mod].encoder(pcl_next)
                    F_dim = feat.shape[2]
                    pred_dir = self.velocity_nets[mod].decoder(
                        c=feat.reshape(-1, F_dim)
                    ).reshape(B, N, d)
                    pcl_next = pcl_next + (1.0 / self.tot_its) * (1.0 / self.num_modules) * pred_dir
        return pcl_next, None

    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch["pc_noisy_l2"].shape[-2]
        return {
            "loss": self.get_supervised_loss(
                pc_clean=batch["pc_clean"].reshape(-1, patch_size, 3),
                pc_noisy_l2=batch["pc_noisy_l2"].reshape(-1, patch_size, 3),
                seed_points_t=batch["seed_points_t"].reshape(-1, patch_size, 3),
                original_time_step=batch["original_time_step"].reshape(-1),
            )
        }

    def execute(self, **kwargs) -> Dict:
        return self.training_step(**kwargs)

    @jt.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        cfg = self.model_config
        patch_size = cfg.get("patch_size", 1000)
        seed_k = cfg.get("seed_k", 6)
        seed_k_alpha = cfg.get("seed_k_alpha", 1)
        niters = cfg.get("niters", 1)

        res = []
        for pc_noisy in batch["pc_noisy"]:
            pc_next = pc_noisy
            for _ in range(niters):
                pc_next = patch_based_denoise(
                    self,
                    pc_next,
                    patch_size=patch_size,
                    seed_k=seed_k,
                    seed_k_alpha=seed_k_alpha,
                )
            res.append({"pc_denoised": pc_next.detach().numpy()})
        return res

    def process_fn(self, batch: List) -> List[Dict]:
        res = []
        for b in batch:
            if not self.is_predict():
                assert b.meta is not None
                res.append({
                    "pc_noisy_l2": b.meta["pc_noisy_l2"],
                    "pc_clean": b.meta["pc_clean"],
                    "seed_points_t": b.meta["seed_points_t"],
                    "original_time_step": b.meta["original_time_step"],
                })
            else:
                d = {"pc_noisy": b.sampled_vertices_noisy}
                if b.sampled_vertices is not None:
                    d["pc_clean"] = b.sampled_vertices
                res.append(d)
        return res
