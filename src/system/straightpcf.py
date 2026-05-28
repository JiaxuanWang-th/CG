"""StraightPCF training loop aligned with official train_straightpcf.py."""
import itertools
import os
from collections import defaultdict
from typing import Dict, List, Optional

import jittor as jt
from jittor import optim
from tqdm import tqdm

from ..data.asset import Asset
from ..data.augment import get_augments
from ..data.dataset import DatasetConfig, PCDataset, PCDatasetModule
from ..data.transform import Transform
from ..model.denoise_utils import patch_based_denoise
from ..model.metrics import chamfer_distance_unit_sphere
from ..model.spec import ModelSpec
from .spec import DummyWriter, _get_item


def _grad_norm(model: ModelSpec, optimizer) -> float:
    """Jittor stores grads on params via opt_grad(optimizer), not .grad."""
    total = 0.0
    for p in model.parameters():
        g = p.opt_grad(optimizer)
        if g is None:
            continue
        total += float((g * g).sum().item())
    return total ** 0.5


class StraightPCFSystem:
    """
    Iteration-based training (max_iters) + Chamfer validation (val_freq),
    matching official StraightPCF stage 3.
    """

    def __init__(
        self,
        dataset_module: PCDatasetModule,
        model: ModelSpec,
        loss_config=None,
        optimizer_config=None,
        trainer_config=None,
        writer: Optional[DummyWriter] = None,
        ckpt_save_dir: str = "experiments",
        ckpt_save_name: str = "checkpoint",
    ):
        self.dataset_module = dataset_module
        self.model = model
        self.loss_config = loss_config or {}
        self.ckpt_save_dir = ckpt_save_dir
        self.ckpt_save_name = ckpt_save_name
        self.writer = writer
        trainer_config = trainer_config or {}

        self.max_iters = trainer_config.get("max_iters", 90000)
        self.val_freq = trainer_config.get("val_freq", 10000)
        self.start_iter = trainer_config.get("start_iter", 0)
        self.log_interval = trainer_config.get("log_interval", 100)
        self.max_grad_norm = trainer_config.get("max_grad_norm", float("inf"))
        # epoch fallback when max_iters is None
        self.epochs = trainer_config.get("epochs", None)
        self.save_every = trainer_config.get("save_every", 5)
        self.start_epoch = trainer_config.get("start_epoch", 0)

        opt_cfg = dict(optimizer_config) if optimizer_config else None
        if opt_cfg is not None and model is not None:
            target = opt_cfg.pop("__target__", "adam")
            if target != "adam":
                raise ValueError(f"official StraightPCF uses Adam, got {target}")
            # official: Adam(model.parameters(), lr=1e-4, weight_decay=0)
            self.optimizer = optim.Adam(model.parameters(), **opt_cfg)
        else:
            self.optimizer = None

        self._best_chamfer = float("inf")

    def _train_one_iter(self, batch) -> tuple:
        self.optimizer.zero_grad()
        loss_dict = self.model.training_step(batch)
        loss = loss_dict["loss"]
        for name in loss_dict:
            if name != "loss" and name in self.loss_config:
                loss = loss + self.loss_config[name] * loss_dict[name]
        self.optimizer.backward(loss)
        gn = _grad_norm(self.model, self.optimizer)
        if self.max_grad_norm < 1e8:
            if hasattr(jt, "clip_grad_norm_"):
                jt.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return _get_item(loss), gn

    @jt.no_grad()
    def validate_chamfer(self) -> float:
        """Full-cloud denoise + unit-sphere Chamfer (official validate())."""
        cfg = self.model.model_config
        patch_size = cfg.get("patch_size", 1000)
        seed_k = cfg.get("seed_k", 6)
        seed_k_alpha = cfg.get("seed_k_alpha", 1)

        chamfer_cfg = self.model.transform_config.get("validate_chamfer_transform")
        if chamfer_cfg is None:
            raise ValueError("validate_chamfer_transform missing in transform config")

        val_transform = Transform(augments=get_augments(*chamfer_cfg["augments"]))
        val_cfgs = self.dataset_module.validate_dataset_config
        assert val_cfgs is not None

        was_predict = self.model.is_predict()
        self.model.set_predict(True)
        self.model.eval()

        cds = []
        for cls, ds_cfg in val_cfgs.items():
            datapath = ds_cfg.datapath
            dataset = PCDataset(
                data=datapath.get_data(),
                transform=val_transform,
                name=f"validate-chamfer-{cls}",
                process_fn=self.model._process_fn,
            )
            for idx in tqdm(range(len(dataset)), desc=f"Chamfer val ({cls})", leave=False):
                asset = dataset[idx]
                proc = self.model.process_fn([asset])
                for item in proc:
                    pc_noisy = jt.array(item["pc_noisy"])
                    pc_clean = item.get("pc_clean")
                    if pc_clean is None:
                        continue
                    pc_denoised = patch_based_denoise(
                        self.model,
                        pc_noisy,
                        patch_size=patch_size,
                        seed_k=seed_k,
                        seed_k_alpha=seed_k_alpha,
                    )
                    cd = chamfer_distance_unit_sphere(
                        pc_denoised.numpy(), pc_clean
                    )
                    cds.append(cd)

        self.model.set_predict(was_predict)
        if not cds:
            return float("nan")
        return float(sum(cds) / len(cds))

    def _save_checkpoint(self, tag: str):
        path = os.path.join(self.ckpt_save_dir, f"{self.ckpt_save_name}_{tag}.pkl")
        os.makedirs(self.ckpt_save_dir, exist_ok=True)
        self.model.save(path)
        return path

    def train(self):
        assert self.optimizer is not None
        self.model.set_predict(False)

        if self.max_iters is not None:
            self._train_iters()
        else:
            self._train_epochs()

    def _train_iters(self):
        train_loader = self.dataset_module.train_dataloader()
        assert train_loader is not None
        train_iter = itertools.cycle(train_loader)

        loss_acc = 0.0
        loss_cnt = 0

        pbar = tqdm(range(self.start_iter + 1, self.max_iters + 1), desc="Train")
        for it in pbar:
            self.model.train()
            batch = next(train_iter)
            loss_val, gn = self._train_one_iter(batch)
            loss_acc += loss_val
            loss_cnt += 1
            last_gn = gn

            if it % self.log_interval == 0:
                avg = loss_acc / max(loss_cnt, 1)
                pbar.set_description(
                    f"Iter {it} | loss {avg:.6f} | grad {last_gn:.4f}"
                )
                loss_acc = 0.0
                loss_cnt = 0

            if it % self.val_freq == 0:
                avg_cd = self.validate_chamfer()
                print(f"[Val] Iter {it:05d} | Chamfer {avg_cd:.6f}")
                if avg_cd < self._best_chamfer:
                    self._best_chamfer = avg_cd
                    self._save_checkpoint(f"best_cd{avg_cd:.4f}_iter{it}")
                self._save_checkpoint(f"iter{it}")

            jt.gc()

        self._save_checkpoint(f"iter{self.max_iters}")

    def _train_epochs(self):
        """Fallback epoch loop (legacy DummySystem behaviour)."""
        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()
            train_loader = self.dataset_module.train_dataloader()
            pbar = tqdm(train_loader, total=len(train_loader) // train_loader.batch_size)
            epoch_loss = []
            for batch in pbar:
                loss_val, _ = self._train_one_iter(batch)
                epoch_loss.append(loss_val)
                pbar.set_description(
                    f"Epoch {epoch} | loss {sum(epoch_loss) / len(epoch_loss):.6f}"
                )
            jt.gc()
            avg_cd = self.validate_chamfer()
            print(f"[Val] Epoch {epoch} | Chamfer {avg_cd:.6f}")
            epochs_in_run = epoch - self.start_epoch + 1
            if epochs_in_run % self.save_every == 0 or epoch + 1 == self.epochs:
                self._save_checkpoint(str(epoch))

    def predict(self):
        raise NotImplementedError("use predict task yaml")
