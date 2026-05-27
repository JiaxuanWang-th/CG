# StraightPCF Jittor 移植说明

在竞赛 Baseline（`/home/cslab/CG`）上，将 [StraightPCF](https://github.com/...)（PyTorch 官方实现：`/home/cslab/StraightPCF`）的三阶段流程用 **纯 Jittor** 复现，并保持 `run.py` + YAML 配置框架不变。

## 三阶段训练流程

| 阶段 | 模型 | 配置 | 说明 |
|------|------|------|------|
| 1 | `VelocityModule` | `configs/task/train_vm.yaml` | 单 VM，DSM 位移回归 |
| 2 | `CoupledVMArch` | `configs/task/train_cvm.yaml` | 多 VM 耦合 + 一致性损失 |
| 3 | `StraightPCF` | `configs/task/train_straightpcf.yaml` | 冻结式 CVM + 可训练距离头 |

推理（推荐提交用）：

```bash
python run.py --task configs/task/predict_straightpcf.yaml
```

在 task yaml 中设置 `load_ckpt` 与 `configs/model/straightpcf.yaml` 里的 `niters` / `tot_its`。

## 已实现模块（相对官方 StraightPCF）

| 模块 | 状态 | 文件 |
|------|------|------|
| Dynamic EdgeConv + 3 层特征提取 | ✅ | `src/model/feature.py` |
| Decoder（out_dim=3 与 out_dim=1） | ✅ | `src/model/feature.py` |
| VelocityModule | ✅ | `src/model/vm.py` |
| CoupledVMArch | ✅ | `src/model/cvm.py` |
| StraightPCF（距离比损失 + finetune） | ✅ | `src/model/straightpcf.py` |
| patch_based_denoise（FPS+KNN+加权融合） | ✅ | `src/model/denoise_utils.py` |
| L1/L2 噪声 + CVM patch 数据 | ✅ | `src/data/augment.py` |
| 推理 `niters` 外循环 | ✅ | 模型 config：`niters` |
| 点数守恒（patch 未覆盖回退） | ✅ | `preserve_point_count=True`（相对官方的增强） |

## 明确取舍（未实现或简化）

| 项目 | 说明 |
|------|------|
| **PUNet / PCNet / Kinect 数据集** | 竞赛仅 ShapeNet mesh/npy，沿用 Baseline 数据管线 |
| **多分辨率训练列表** | 官方 `resolutions=['10k','30k','50k']`；本仓库单次 mesh 采样 32768 点 |
| **Graph Laplacian 精修 (`glr`)** | 官方代码中已注释，未移植 |
| **`denoise_large_pointcloud`（KMeans 分块）** | B 榜大规模可后续加；A 榜单样本 ~5 万点 patch 流程足够 |
| **迭代训练步数 `max_iters`** | 官方按 iteration；本仓库按 **epoch**（`trainer.epochs`），需自行换算 |
| **CVM 默认 4 模块** | 官方 `train_cvm` 默认 `num_modules=4`；默认配置为 **2**（与 StraightPCF 阶段一致，省显存） |
| **frame_knn** | 官方 32；原 Baseline VM 为 16；新配置 **32**，旧 VM checkpoint 需重训或改回 16 |
| **高斯噪声 L1/L2** | 官方 AddNoise 用 Gaussian；竞赛 Baseline 用 **Laplace**，与评测噪声更一致 |
| **Chamfer 在线验证** | 未接入训练 loop，请用 `evaluate.py` 离线评测 |
| **TensorBoard / iteration checkpoint** | 未实现，仅 epoch 存盘 |

## 预训练权重衔接

在 `configs/model/cvm.yaml` 的 task 中通过 `load_ckpt` 加载 VM；或在 model yaml 设置：

```yaml
vm_ckpt: experiments/vm/checkpoint_99.pkl
```

StraightPCF 阶段在 model yaml 或构造前设置：

```yaml
cvm_ckpt: experiments/cvm/checkpoint_99.pkl
```

## 超参建议（对齐官方 test）

`configs/model/straightpcf.yaml`：

- `patch_size: 1000`, `seed_k: 6`, `seed_k_alpha: 1`
- `tot_its: 2~3`, `niters: 1~3`（高噪声可增大 `niters`）
- `feat_embedding_dim: 128`（距离头）；`cvm_feat_embedding_dim: 256`（velocity_nets，须与 CVM ckpt 一致）

## 规则合规

- **仅 Jittor**：无 PyTorch 依赖
- **仅提供数据集**：训练仍从 `dataset_train` mesh 采样
- **提交点数一致**：`preserve_point_count` 保证与 `noisy.npy` 同 N
