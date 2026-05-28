"""CVM training with patch val loss + full-cloud Chamfer validation."""
from collections import defaultdict
import os
from typing import Optional

import jittor as jt
from tqdm import tqdm

from ..data.augment import get_augments
from ..data.dataset import PCDataset
from ..data.transform import Transform
from ..model.denoise_utils import patch_based_denoise
from ..model.metrics import chamfer_distance_unit_sphere
from .spec import DummySystem, DummyWriter, _get_item


class CVMSystem(DummySystem):
    """Epoch training; logs patch supervised loss and Chamfer (denoising quality)."""

    def __init__(
        self,
        dataset_module,
        model,
        loss_config=None,
        optimizer_config=None,
        trainer_config=None,
        writer: Optional[DummyWriter] = None,
        ckpt_save_dir: str = "experiments",
        ckpt_save_name: str = "checkpoint",
    ):
        super().__init__(
            dataset_module=dataset_module,
            model=model,
            loss_config=loss_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
            writer=writer,
            ckpt_save_dir=ckpt_save_dir,
            ckpt_save_name=ckpt_save_name,
        )
        trainer_config = trainer_config or {}
        self.val_chamfer_every = int(trainer_config.get("val_chamfer_every", 1))
        self.val_chamfer_max_samples = trainer_config.get("val_chamfer_max_samples", None)
        if self.val_chamfer_max_samples is not None:
            self.val_chamfer_max_samples = int(self.val_chamfer_max_samples)
        self._best_chamfer = float("inf")
        self._epoch_val_patch_loss = None
        self._epoch_val_cd = None

    def _mean_val_patch_loss(self) -> Optional[float]:
        keys = [k for k in self._validation_loss if k.endswith("_loss_sum")]
        if not keys:
            return None
        vals = self._validation_loss[keys[0]]
        return float(sum(vals) / len(vals)) if vals else None

    @jt.no_grad()
    def validate_chamfer(self) -> float:
        """Full-cloud patch_based_denoise + unit-sphere Chamfer (official CVM validate)."""
        cfg = self.model.model_config
        patch_size = cfg.get("patch_size", 1000)
        seed_k = cfg.get("seed_k", 6)
        seed_k_alpha = cfg.get("seed_k_alpha", 1)

        chamfer_cfg = self.model.transform_config.get("validate_chamfer_transform")
        if chamfer_cfg is None:
            raise ValueError(
                "validate_chamfer_transform missing in transform config (e.g. configs/transform/cvm.yaml)"
            )

        val_transform = Transform(augments=get_augments(*chamfer_cfg["augments"]))
        val_cfgs = self.dataset_module.validate_dataset_config
        assert val_cfgs is not None

        was_predict = self.model.is_predict()
        self.model.set_predict(True)
        self.model.eval()

        cds = []
        for cls, ds_cfg in val_cfgs.items():
            dataset = PCDataset(
                data=ds_cfg.datapath.get_data(),
                transform=val_transform,
                name=f"validate-chamfer-{cls}",
                process_fn=self.model._process_fn,
            )
            n = len(dataset)
            if self.val_chamfer_max_samples is not None:
                n = min(n, self.val_chamfer_max_samples)
            for idx in tqdm(range(n), desc=f"Chamfer CD ({cls})", leave=False):
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
                    cds.append(
                        chamfer_distance_unit_sphere(pc_denoised.numpy(), pc_clean)
                    )

        self.model.set_predict(was_predict)
        if not cds:
            return float("nan")
        return float(sum(cds) / len(cds))

    def _save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.model.save(path)

    def train(self):
        assert self.optimizer is not None, "optimizer is None, cannot train"
        self.model.set_predict(False)

        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()
            self.on_train_epoch_start()
            train_dataloader = self.dataset_module.train_dataloader()
            assert train_dataloader is not None
            pbar = tqdm(
                train_dataloader,
                total=len(train_dataloader) // train_dataloader.batch_size,
            )
            train_loss_acc = 0.0
            train_steps = 0
            for batch in pbar:
                self.on_train_batch_start()
                loss = self.training_step(batch)
                self.optimizer.zero_grad()
                self.optimizer.backward(loss)
                lv = _get_item(loss)
                train_loss_acc += lv
                train_steps += 1
                pbar.set_description(f"Epoch {epoch}, Train patch-loss: {lv:.6f}")
                self.on_before_optimizer_step(self.optimizer)
                self.optimizer.step()
                self.on_train_batch_end()
            train_loss_mean = train_loss_acc / max(train_steps, 1)
            self.on_train_epoch_end()
            jt.gc()

            self._epoch_val_patch_loss = None
            self._epoch_val_cd = None

            validate_dataloader = self.dataset_module.validate_dataloader()
            if validate_dataloader is not None:
                self.model.eval()
                self.on_validation_epoch_start()
                if isinstance(validate_dataloader, dict):
                    for name, dataloader in validate_dataloader.items():
                        vbar = tqdm(
                            dataloader,
                            total=len(dataloader) // dataloader.batch_size,
                        )
                        for batch in vbar:
                            self.on_validation_batch_start()
                            loss = self.validation_step(batch)
                            vbar.set_description(
                                f"Epoch {epoch}, Val patch-loss ({name}): {_get_item(loss):.6f}"
                            )
                            self.on_validation_batch_end()
                self.on_validation_epoch_end()
                self._epoch_val_patch_loss = self._mean_val_patch_loss()

            run_cd = (
                validate_dataloader is not None
                and self.val_chamfer_every > 0
                and (epoch + 1) % self.val_chamfer_every == 0
            )
            if run_cd:
                self._epoch_val_cd = self.validate_chamfer()
                if self._epoch_val_cd < self._best_chamfer:
                    self._best_chamfer = self._epoch_val_cd
                    tag = f"best_cd{self._epoch_val_cd:.4f}_epoch{epoch}"
                    best_path = os.path.join(
                        self.ckpt_save_dir, f"{self.ckpt_save_name}_{tag}.pkl"
                    )
                    self._save_checkpoint(best_path)

            patch_s = (
                f"{self._epoch_val_patch_loss:.6f}"
                if self._epoch_val_patch_loss is not None
                else "n/a"
            )
            cd_s = f"{self._epoch_val_cd:.6f}" if self._epoch_val_cd is not None else "n/a"
            best_s = (
                f"{self._best_chamfer:.6f}"
                if self._best_chamfer < float("inf")
                else "n/a"
            )
            print(
                f"[Epoch {epoch}] train_patch_loss={train_loss_mean:.6f} | "
                f"val_patch_loss={patch_s} | val_CD={cd_s} | best_CD={best_s}",
                flush=True,
            )

            epochs_in_this_run = epoch - self.start_epoch + 1
            should_save = (
                epochs_in_this_run % self.save_every == 0
                or epoch + 1 == self.epochs
            )
            if should_save:
                path = os.path.join(
                    self.ckpt_save_dir, f"{self.ckpt_save_name}_{epoch}.pkl"
                )
                self._save_checkpoint(path)
