from typing import Dict, List

import jittor as jt
import numpy as np

from .denoise_utils import patch_based_denoise
from .feature import Decoder, FeatureExtraction
from .spec import ModelSpec

from ..data.asset import Asset


def get_random_indices(n, m):
    assert m < n
    idx = np.random.permutation(n)[:m]
    return jt.array(idx).int32()


class VelocityModule(ModelSpec):

    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)

        cfg = self.model_config
        self.frame_knn = cfg["frame_knn"]
        self.num_train_points = cfg["num_train_points"]
        self.dsm_sigma = cfg["dsm_sigma"]

        self.encoder = FeatureExtraction(
            k=self.frame_knn,
            input_dim=3,
            embedding_dim=cfg["feat_embedding_dim"],
            edge_aggr=cfg.get("edge_aggr", "mean"),
        )
        self.decoder = Decoder(
            z_dim=self.encoder.embedding_dim,
            dim=3,
            out_dim=3,
            hidden_size=cfg["decoder_hidden_dim"],
        )

    def get_supervised_loss(self, pc_noisy, pc_mix, pc_clean):
        """Denoising score matching on interpolated patches."""
        B, N_noisy, d = pc_mix.shape

        pnt_idx = get_random_indices(N_noisy, self.num_train_points)

        feat = self.encoder(pc_mix)
        F_dim = feat.shape[2]

        feat = feat[:, pnt_idx, :]
        pc_noisy = pc_noisy[:, pnt_idx, :]
        pc_clean = pc_clean[:, pnt_idx, :]

        grad_dir_t_target = pc_clean - pc_noisy

        pred_dir = self.decoder(c=feat.reshape(-1, F_dim)).reshape(B, len(pnt_idx), d)

        loss = (((pred_dir - grad_dir_t_target) ** 2.0) / self.dsm_sigma).sum(dim=-1).mean()
        return loss

    def denoise_langevin_dynamics(self, pcl_noisy, num_steps: int = 4):
        B, N, d = pcl_noisy.shape
        with jt.no_grad():
            pcl_next = pcl_noisy.clone()
            for _ in range(num_steps):
                feat = self.encoder(pcl_next)
                F_dim = feat.shape[2]
                pred_dir = self.decoder(c=feat.reshape(-1, F_dim)).reshape(B, N, d)
                pcl_next = pcl_next + (1.0 / num_steps) * pred_dir
        return pcl_next, None

    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch["pc_noisy"].shape[-2]
        pc_noisy = batch["pc_noisy"].reshape(-1, patch_size, 3)
        pc_mix = batch["pc_mix"].reshape(-1, patch_size, 3)
        pc_clean = batch["pc_clean"].reshape(-1, patch_size, 3)
        loss = self.get_supervised_loss(pc_noisy=pc_noisy, pc_mix=pc_mix, pc_clean=pc_clean)
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

    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        res = []
        for b in batch:
            if not self.is_predict():
                assert b.meta is not None
                res.append({
                    "pc_noisy": b.meta["pc_noisy"],
                    "pc_clean": b.meta["pc_clean"],
                    "pc_mix": b.meta["pc_mix"],
                })
            else:
                d = {"pc_noisy": b.sampled_vertices_noisy}
                if b.sampled_vertices is not None:
                    d["pc_clean"] = b.sampled_vertices
                res.append(d)
        return res
