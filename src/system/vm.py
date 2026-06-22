from typing import List, Dict, Optional

import numpy as np
import os

from .cvm import CVMSystem
from .spec import DummyWriter
from ..data.asset import Asset, Exporter
from ..data.utils import denormalize_unit_sphere

class VMWriter(DummyWriter):
    
    def __init__(self, save_dir: str="tmp_predict", save_name: str="predict", output_format: str="npy"):
        super().__init__()
        self.save_dir = save_dir
        self.save_name = save_name
        self.output_format = output_format
    
    @staticmethod
    def _submission_relpath(npy_path: str) -> str:
        """.../shapenet/<synset>/<id>/noisy.npy -> shapenet/<synset>/<id>"""
        norm = npy_path.replace("\\", "/")
        parts = norm.split("/")
        if "shapenet" in parts:
            i = parts.index("shapenet")
            return os.path.join(*parts[i:-1])
        return os.path.dirname(norm)

    def write(self, batch, prediction: List[Dict], dataset_module=None):
        for i, asset in enumerate(batch['asset']):
            path = asset.path
            assert path is not None, "asset path is None"
            dirname = os.path.join(self.save_dir, self._submission_relpath(path))
            os.makedirs(dirname, exist_ok=True)
            denoised = prediction[i]['pc_denoised']
            if isinstance(denoised, np.ndarray):
                denoised_np = denoised
            else:
                denoised_np = denoised.numpy()
            if asset.norm_center is not None and asset.norm_scale is not None:
                denoised_np = denormalize_unit_sphere(
                    denoised_np, asset.norm_center, np.float32(asset.norm_scale)
                )
            if self.output_format == 'npy':
                np.save(os.path.join(dirname, f"{self.save_name}.npy"), denoised_np.astype(np.float32))
            else:
                Exporter.export_obj(denoised_np, os.path.join(dirname, f"{self.save_name}.obj"))

class VMSystem(CVMSystem):
    """VM epoch training + val patch loss + full-cloud Chamfer (same loop as CVMSystem)."""