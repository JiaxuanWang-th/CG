"""StraightPCF — distance-scaled coupled velocity denoising (stage 3)."""
from typing import Dict, List, Optional

import jittor as jt
from jittor import nn

from .cvm import CoupledVMArch
from .denoise_utils import patch_based_denoise
from .feature import Decoder, FeatureExtraction
from .spec import ModelSpec
from .vm import get_random_indices


class StraightPCF(ModelSpec):
    """
    Frozen CVM velocity nets + trainable distance estimation head.
    Inference: pred_d scales each VM step; sequential module updates.
    """

    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        cfg = self.model_config

        self.frame_knn = cfg["frame_knn"]
        self.num_train_points = cfg["num_train_points"]
        self.dsm_sigma = cfg["dsm_sigma"]
        self.tot_its = cfg.get("tot_its", 3)
        self.distance_estimation = cfg.get("distance_estimation", True)

        cvm_ckpt = cfg.get("cvm_ckpt", None)
        self.num_modules = cfg.get("num_modules", 2)
        # velocity_nets 结构与 CVM 阶段一致（256），与距离头 encoder（128）分开
        cvm_cfg = {k: v for k, v in cfg.items() if k not in ("vm_ckpt", "cvm_ckpt")}
        cvm_cfg["feat_embedding_dim"] = cfg.get(
            "cvm_feat_embedding_dim", cfg.get("vm_feat_embedding_dim", 256)
        )
        cvm = CoupledVMArch(cvm_cfg, transform_config)
        if cvm_ckpt is not None:
            cvm.load(cvm_ckpt)
        self.velocity_nets = cvm.velocity_nets
        self.num_modules = len(self.velocity_nets)

        self.encoder = FeatureExtraction(
            k=self.frame_knn,
            input_dim=3,
            embedding_dim=cfg["feat_embedding_dim"],
            distance_estimation=self.distance_estimation,
        )
        self.decoder = Decoder(
            z_dim=self.encoder.embedding_dim,
            dim=3,
            out_dim=1,
            hidden_size=cfg["decoder_hidden_dim"],
        )

        # 默认冻结 CVM：只训练距离头，显存/内存约为全量微调的 1/3
        self.freeze_velocity_nets = cfg.get("freeze_velocity_nets", True)
        if self.freeze_velocity_nets:
            self._set_velocity_nets_trainable(False)

    def _set_velocity_nets_trainable(self, trainable: bool):
        for mod in self.velocity_nets:
            mod.train() if trainable else mod.eval()
            for p in mod.parameters():
                if trainable:
                    p.start_grad()
                else:
                    p.stop_grad()

    def get_trainable_parameters(self):
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        if not self.freeze_velocity_nets:
            for mod in self.velocity_nets:
                params += list(mod.parameters())
        return params

    def get_supervised_loss(
        self,
        pc_clean,
        pc_noisy_l2,
        seed_points_t,
        original_time_step,
    ):
        B, N, d = pc_noisy_l2.shape

        curr_step = original_time_step.reshape(B, 1, 1)
        pc_noisy = curr_step * pc_clean + (1 - curr_step) * pc_noisy_l2

        num = jt.sqrt(((pc_clean - pc_noisy) ** 2).sum(dim=-1))
        den = jt.sqrt(((pc_clean - pc_noisy_l2) ** 2).sum(dim=-1))
        ratio = num[:, 0] / (den[:, 0] + 1e-8)

        pc_clean_c = pc_clean - seed_points_t
        pc_noisy_c = pc_noisy - seed_points_t

        feat_d = self.encoder(pc_noisy_c)
        F_d = feat_d.shape[2]
        pred_d = self.decoder(c=feat_d.reshape(-1, F_d), B=B, N=N).reshape(B)

        dist_loss = ((pred_d - ratio) ** 2).mean()

        for mod in range(self.num_modules):
            if self.freeze_velocity_nets:
                with jt.no_grad():
                    feat = self.velocity_nets[mod].encoder(pc_noisy_c)
                    F_dim = feat.shape[2]
                    pred_dir = self.velocity_nets[mod].decoder(
                        c=feat.reshape(-1, F_dim)
                    ).reshape(B, N, d)
            else:
                feat = self.velocity_nets[mod].encoder(pc_noisy_c)
                F_dim = feat.shape[2]
                pred_dir = self.velocity_nets[mod].decoder(
                    c=feat.reshape(-1, F_dim)
                ).reshape(B, N, d)
            pc_noisy_c = pc_noisy_c + (1.0 / self.num_modules) * pred_d.reshape(B, 1, 1) * pred_dir

        finetune_loss = 2e2 * ((pc_clean_c - pc_noisy_c) ** 2).sum(dim=-1).mean()
        return (dist_loss + finetune_loss) / self.dsm_sigma

    def denoise_langevin_dynamics(self, pcl_noisy, num_steps: int = 4):
        B, N, d = pcl_noisy.shape
        with jt.no_grad():
            pcl_next = pcl_noisy.clone()

            feat_d = self.encoder(pcl_next)
            F_d = feat_d.shape[2]
            pred_d = self.decoder(c=feat_d.reshape(-1, F_d), B=B, N=N).reshape(B, 1, 1)

            for _ in range(self.tot_its):
                for mod in range(self.num_modules):
                    feat = self.velocity_nets[mod].encoder(pcl_next)
                    F_dim = feat.shape[2]
                    pred_dir = self.velocity_nets[mod].decoder(
                        c=feat.reshape(-1, F_dim)
                    ).reshape(B, N, d)
                    pcl_next = pcl_next + (1.0 / self.tot_its) * (1.0 / self.num_modules) * pred_d * pred_dir
        return pcl_next, None

    def training_step(self, batch: Dict) -> Dict:
        if self.freeze_velocity_nets:
            for mod in self.velocity_nets:
                mod.eval()
        self.encoder.train()
        self.decoder.train()

        patch_size = batch["pc_noisy_l2"].shape[-2]
        loss = self.get_supervised_loss(
            pc_clean=batch["pc_clean"].reshape(-1, patch_size, 3),
            pc_noisy_l2=batch["pc_noisy_l2"].reshape(-1, patch_size, 3),
            seed_points_t=batch["seed_points_t"].reshape(-1, patch_size, 3),
            original_time_step=batch["original_time_step"].reshape(-1),
        )
        return {"loss": loss}

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
