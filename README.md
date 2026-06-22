# Point Cloud Denoising

三阶段点云降噪管线：**VM → CVM → SPCF**。训练阶段对每样本随机采样 10k / 30k / 50k 点（multires），推理阶段对测试集全图 50k 点降噪。

---

## Quick Start（推理 + 提交）

```bash
conda activate jittor   # 见下方环境配置

# 1. 解压测试集到项目根目录 -> ./dataset_test_noisy/
unzip dataset_test_noisy.zip

# 2. 推理并打包
bash scripts/pack_submission.sh
# 产出: output/predictions/ 与 submission.zip
```

等价命令：

```bash
python run.py --task configs/task/predict.yaml --seed 123
cd output/predictions && zip -r ../../submission.zip shapenet/
```

默认加载 `checkpoints/spcf/checkpoint_best.pkl`（线上全流程 **75.54** 分）。换权重只需改 `configs/task/predict.yaml` 中的 `load_ckpt`，或：

```bash
CKPT=checkpoints/spcf/checkpoint_best.pkl bash scripts/pack_submission.sh
```

---

## 环境配置

**依赖**：Python 3.9 · GCC/G++ ≤ 10 · CUDA · [Jittor](https://github.com/Jittor/jittor)

```bash
conda create -n jittor python=3.9 -y
conda activate jittor
conda install -c conda-forge gcc=10 gxx=10 libgomp -y

cd /path/to/CG
python -m pip install -r requirements.txt
pip install point-cloud-utils
```

---

## 数据准备

路径默认为项目根目录下的相对路径，可按需修改 `configs/data/` 中对应 yaml。

| 用途 | 默认路径 | 配置文件 |
|------|----------|----------|
| 训练 | `./dataset_train/` | `configs/data/train_{vm,cvm,spcf}.yaml` |
| 推理 | `./dataset_test_noisy/` | `configs/data/predict.yaml` |

**训练集**目录结构：

```
dataset_train/shapenet/<synset_id>/<model_id>/models/model_normalized.obj
```

**测试集**目录结构：

```
dataset_test_noisy/shapenet/<synset_id>/<model_id>/noisy.npy
```

数据列表：`datalist/train.txt`、`validate.txt`、`test.txt`。

---

## 预训练权重

```
checkpoints/
├── vm/checkpoint_99.pkl          # VM 续训起点（可选）
├── vm/checkpoint_299.pkl         # Stage 1 输出
├── cvm/checkpoint_best.pkl       # Stage 2 输出
└── spcf/checkpoint_best.pkl      # Stage 3 定稿（推理默认）
```

| 阶段 | 文件 | 说明 |
|------|------|------|
| VM | `checkpoints/vm/checkpoint_299.pkl` | Stage 1 |
| CVM | `checkpoints/cvm/checkpoint_best.pkl` | 依赖 VM |
| **SPCF** | `checkpoints/spcf/checkpoint_best.pkl` | 全流程定稿 |

---

## 训练（可选）

统一设置：`export CUDA_VISIBLE_DEVICES=0`，`--seed 123`。

三阶段按顺序执行；后一阶段 yaml 中的 `vm_ckpt` / `cvm_ckpt` 指向前一阶段输出。

```bash
python run.py --task configs/task/train_vm.yaml --seed 123
python run.py --task configs/task/train_cvm.yaml --seed 123
python run.py --task configs/task/train_spcf.yaml --seed 123
```

| 阶段 | Task | 输出目录 |
|------|------|----------|
| VM | `configs/task/train_vm.yaml` | `checkpoints/vm/` |
| CVM | `configs/task/train_cvm.yaml` | `checkpoints/cvm/` |
| SPCF | `configs/task/train_spcf.yaml` | `checkpoints/spcf/` |

**VM 续训说明**：默认从 `checkpoint_99.pkl` 续到 epoch 299。若要从头训练 VM，删除 `load_ckpt` 并将 `trainer.start_epoch` 设为 `0`。

---

## 提交格式

```
submission.zip
└── shapenet/
    └── <synset_id>/
        └── <model_id>/
            └── denoised.npy    # float32, shape (N, 3)
```

---

## 定稿结果

| 指标 | 分数 |
|------|------|
| **总分** | **75.54** |
| Chamfer Distance | 62.46 |
| Point-to-Surface | 88.63 |

---

## 目录结构

```
CG/
├── run.py
├── checkpoints/              # 预训练权重
├── configs/
│   ├── task/                 # train_{vm,cvm,spcf}.yaml, predict.yaml
│   ├── data/
│   ├── model/
│   ├── system/               # vm / cvm / spcf（训练）, infer（推理）
│   └── transform/
├── datalist/
├── scripts/pack_submission.sh
└── src/
```

### 配置说明

- **`configs/system/vm.yaml`**：VM 训练循环与 checkpoint 保存路径。
- **`configs/system/infer.yaml`**：推理专用 runtime（predict 模式），与 VM 训练配置分离，避免歧义。
- **`configs/task/predict.yaml`**：默认推理入口，指向 `checkpoints/spcf/checkpoint_best.pkl`。

---

## 常见问题

**Q: 没有 GPU / 没装训练数据，能复现提交吗？**  
可以。只需测试集 + `predict.yaml`（或 `pack_submission.sh`），使用仓库内 `checkpoints/spcf/checkpoint_best.pkl`。

**Q: 训练报找不到 mesh？**  
确认 `dataset_train/` 已解压，或修改 `configs/data/train_*.yaml` 中的 `input_dataset_dir`。

**Q: 推理报找不到 noisy.npy？**  
确认 `dataset_test_noisy/` 已解压，或修改 `configs/data/predict.yaml`。

**Q: 随机种子？**  
定稿实验使用 `--seed 123`。
