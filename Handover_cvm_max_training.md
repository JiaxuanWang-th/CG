# 云端交接文档：vm_max epoch128 上训练 CVM max（Stage 2）

**用途**：在云端 `/home/ubuntu/CG` 上，用 **vm_max epoch128** 作为预训练底座，**从头训练** CoupledVM（CVM max），避免与学校配置漂移。  
**学校对照实验**：学校机器续训 `cvm_max`（仍绑 `vm_max epoch89`），见本文 **第十一节**。  
**日期**：2026-06-02  

---

## 一、实验目标与边界

| 项目 | 说明 |
|------|------|
| **要做什么** | Stage 2：`CoupledVMArch`，4×VelocityModule + consistency loss |
| **预训练** | 仅加载 **vm_max** 权重（`vm_ckpt`）；**不要**加载已有 cvm_max |
| **不要做什么** | 不要改 `niters>1`；不要在本实验里训 SPCF Stage3 |
| **成功标准** | test 线上或 `benchmark_val`（**n1 推理**）超过学校 CVM97 线上 **72.60** |
| **本地 val 注意** | val 分与线上可差 4–6 分；**勿**用 val 绝对分或 sweep 参数下注 |

**与学校的唯一 intentional 差异**：

```yaml
# 学校 train_cvm_max.yaml
vm_ckpt: experiments/vm_max/checkpoint_best_cd0.0002_epoch89.pkl

# 云端本实验（示例，以实际 128 ckpt 文件名为准）
vm_ckpt: experiments/vm_max/checkpoint_best_cd0.0002_epoch128.pkl
```

其余 task / model / data / transform / system **应与学校一致**（见下文全文）。

---

## 二、前置检查（开训前必做）

```bash
cd /home/ubuntu/CG
git pull   # 与学校 CG 仓库同步

source ~/miniconda3/etc/profile.d/conda.sh   # 或 miniconda3 实际路径
conda activate jittor

nvidia-smi
python -c "import jittor as jt; print('jittor', jt.__version__, 'cuda', jt.flags.use_cuda)"
```

### 2.1 训练数据

| 项 | 路径 |
|----|------|
| Mesh 根目录 | `/home/dataset_train` |
| 训练列表 | `./datalist/train.txt` |
| 验证列表 | `./datalist/validate.txt` |
| 单样本 mesh | `{root}/shapenet/<synset>/<id>/models/model_normalized.obj` |

```bash
test -f /home/dataset_train/shapenet/03642806/2134ad3fc25a6284193a4c984002ed32/models/model_normalized.obj \
  && echo "dataset OK" || echo "dataset MISSING"
```

若学校路径不同，**只改** `configs/data/train_cvm_max.yaml` 的 `input_dataset_dir`，两处（train/validate）保持一致。

### 2.2 vm_max epoch128 权重（关键）

```bash
ls -lh experiments/vm_max/checkpoint_*128*.pkl
ls -lh experiments/vm_max/checkpoint_best_cd*_epoch128.pkl
```

- 必须是 **`vm_max`**（`VelocityModule`，`edge_aggr: max`），**不是** `cvm_max`。
- 文件约 **~1MB 量级**（单 VM）；CVM ckpt 约 **~3.7–4MB**（4 modules）。
- 将实际文件名写入 `train_cvm_max.yaml` 的 `vm_ckpt`（下文 3.1）。

**禁止**：

- 用 `experiments/cvm_max/*` 当 `vm_ckpt`
- 用 `experiments/vm/*`（非 max EdgeConv 的旧 VM）

### 2.3 代码必须存在

```bash
test -f run.py && test -f src/model/cvm.py && test -f src/system/cvm.py \
  && test -f configs/transform/cvm.yaml && echo "code OK"
```

---

## 三、配置文件（云端对照学校，建议逐项 diff）

> 以下内容为 **2026-06-02 学校仓库快照**。云端若已有同名文件，用 `diff` 核对；**仅允许改 `vm_ckpt` 与数据路径**。

### 3.1 `configs/task/train_cvm_max.yaml`（云端必改 vm_ckpt）

```yaml
mode: train
debug: false

# ★ 云端：改为 vm_max epoch128 的实际路径
vm_ckpt: experiments/vm_max/checkpoint_best_cd0.0002_epoch128.pkl

# ★ 从头训 CVM：不要设 load_ckpt
# load_ckpt: ...

components:
  data: train_cvm_max
  transform: cvm
  system: cvm_max
  model: cvm_max

loss:
  loss: 1.0

optimizer:
  __target__: adam
  lr: 0.0001

trainer:
  epochs: 100
  save_every: 10
  start_epoch: 0
  val_chamfer_every: 1
  val_chamfer_max_samples: 20
```

**`vm_ckpt` 注入方式**（`run.py`）：task 里的 `vm_ckpt` 会写入 model config，在 `CoupledVMArch.__init__` 里对 **4 个** `VelocityModule` 各 `load` 一次。

**不要**同时设 `load_ckpt`（除非做 CVM 断点续训，见第十一节）。

---

### 3.2 `configs/model/cvm_max.yaml`

```yaml
__target__: CoupledVMArch
edge_aggr: max
frame_knn: 32
patch_size: 1000
seed_k: 6
seed_k_alpha: 1
num_train_points: 128
feat_embedding_dim: 256
decoder_hidden_dim: 64
dsm_sigma: 0.01
num_modules: 4
tot_its: 3
consistency_weight: 10.0
# vm_ckpt: 由 configs/task/train_cvm_max.yaml 注入，勿写死
```

**注意**：

- **不要**写 `niters`（推理参数；训练不用）。
- `num_modules: 4`、`consistency_weight: 10.0` 与官方 StraightPCF CVM 对齐。
- `tot_its: 3` 用于 **patch 内** Langevin（`denoise_langevin_dynamics`），与推理默认一致。

---

### 3.3 `configs/data/train_cvm_max.yaml`

```yaml
data_name: &data_name models/model_normalized.obj
loader: &loader obj

train_dataset:
  shuffle: True
  batch_size: 8
  num_workers: 0
  datapath:
    input_dataset_dir: /home/dataset_train
    use_prob: True
    num_files: 10000
    loader: *loader
    data_name: *data_name
    ignore_check: false
    data_path:
      shapenet:
        - [./datalist/train.txt, 1.0]

validate_dataset:
  shuffle: False
  batch_size: 1
  num_workers: 0
  datapath:
    input_dataset_dir: /home/dataset_train
    use_prob: False
    loader: *loader
    data_name: *data_name
    ignore_check: false
    data_path:
      shapenet:
        - [./datalist/validate.txt, 1.0]
```

| 字段 | 值 | 说明 |
|------|-----|------|
| `batch_size` | **8** | CVM 比 VM（16）小，防 OOM |
| `num_workers` | **0** | 学校如此；云可保持 0 避免偏差 |
| `num_files` | 10000 | 每 epoch 随机采 10000 个训练样本 |

---

### 3.4 `configs/transform/cvm.yaml`

```yaml
train_transform:
  augments:
    - __target__: sample
      num_samples: 32768
      num_vertex_samples: 1024
    - __target__: normalize_pc
    - __target__: add_noise
      noise_std_min: 0.005
      noise_std_max: 0.020
    - __target__: linear
      scale: [0.8, 1.2]
      rotate_x_range: [-180, 180]
      rotate_y_range: [-180, 180]
      rotate_z_range: [-180, 180]
      scale_p: 0.5
      rotate_p: 0.5
    - __target__: patch
      patch_size: 1000
      num_patches: 1
      train_cvm_network: true

validate_transform: &cvm_validate
  augments:
    - __target__: sample
      num_samples: 32768
      num_vertex_samples: 1024
    - __target__: normalize_pc
    - __target__: add_noise
      noise_std_min: 0.005
      noise_std_max: 0.020
    - __target__: patch
      patch_size: 1000
      num_patches: 1
      train_cvm_network: true

validate_chamfer_transform:
  augments:
    - __target__: sample
      num_samples: 32768
      num_vertex_samples: 1024
    - __target__: normalize_pc
    - __target__: add_noise
      noise_std_min: 0.015
      noise_std_max: 0.015
      distribution: laplace

predict_transform: *cvm_validate
```

**训练内 val_CD 与 patch val 的区别**：

- **patch val loss**：`validate_transform`，随机噪声 [0.005, 0.020]。
- **val_CD（选 best）**：`validate_chamfer_transform`，**固定** Laplace σ=0.015，整云 `patch_based_denoise` **一次**（等价推理 **niters=1**），见 `src/system/cvm.py::validate_chamfer`。

---

### 3.5 `configs/system/cvm_max.yaml`

```yaml
__target__: cvm

ckpt_save_dir: experiments/cvm_max
ckpt_save_name: checkpoint
```

产出目录：

```
experiments/cvm_max/checkpoint_{epoch}.pkl          # 每 save_every epoch
experiments/cvm_max/checkpoint_best_cd{cd}_epoch{N}.pkl  # val_CD 刷新时
```

---

## 四、训练命令与日志

```bash
cd /home/ubuntu/CG
mkdir -p log

screen -dmS train_cvm_max bash -lc '
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate jittor
  export CUDA_VISIBLE_DEVICES=0
  cd /home/ubuntu/CG
  python run.py --task configs/task/train_cvm_max.yaml --seed 123 \
    2>&1 | tee log/train_cvm_max.log
'

# 查看
tail -f log/train_cvm_max.log
screen -r train_cvm_max
```

**开训首行应出现**：

```
use vm_ckpt: experiments/vm_max/checkpoint_best_cd0.0002_epoch128.pkl
```

若 vm_ckpt 路径错或未注入，**立刻停训**。

**每个 epoch 末尾一行**（示例）：

```
[Epoch 42] train_patch_loss=0.190000 | val_patch_loss=0.185000 | val_CD=0.000210 | best_CD=0.000180
```

| 字段 | 含义 |
|------|------|
| `val_CD` | 20 个 validate 样本整云 Chamfer（越小越好） |
| `best_CD` | 历史最小 val_CD |
| best 文件 | `checkpoint_best_cd0.0002_epoch42.pkl` 这类名字 |

**耗时粗估**：~15–20 min/epoch（4090，batch 8）→ 100 epoch ≈ **1–1.5 天**。

---

## 五、权重加载逻辑（防偏差）

```
run.py
  ├─ 读 task.yaml → vm_ckpt 写入 model_config
  ├─ CoupledVMArch.__init__
  │     └─ for i in 4 modules: VelocityModule.load(vm_ckpt)  # 四个模块同一份 VM 初始化
  ├─ （可选）task.load_ckpt → model.load()  # 仅 CVM 断点续训时用
  └─ CVMSystem.train()  # 只训练 CVM 全部参数，不冻结 VM
```

**云端新训**：只设 `vm_ckpt`，**不设** `load_ckpt`。

---

## 六、推理 / 打包 / 评测（训完后）

**固定推理默认值（与训练对齐）**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `niters` | **1**（或不写） | **禁止 2** |
| `tot_its` | 3 | model 默认 |
| `seed_k` | 6 | patch 密度 |
| `seed` | 123 | `run.py --seed 123` |

### 6.1 本地 benchmark_val（100 样本，有 GT）

```bash
# 1) 若无 benchmark_val，在学校或云执行一次：
python scripts/export_val_benchmark.py \
  --mesh-root /home/dataset_train \
  --out benchmark_val --workers 8 --skip-existing

# 2) predict（改 load_ckpt 与 save_dir）
python run.py --task configs/task/predict_val_cvm_max.yaml --seed 123

# 3) evaluate
python evaluate.py \
  --pred_dir results_val_cvm_max \
  --gt_dir benchmark_val \
  --noisy_dir benchmark_val \
  --mesh_dir /home/dataset_train \
  --gt_filename clean.npy \
  --noisy_filename noisy.npy \
  --pred_filename denoised.npy \
  --workers 8 --verbose
```

`predict_val_cvm_max.yaml` 模板：

```yaml
mode: predict
debug: false

load_ckpt: experiments/cvm_max/checkpoint_best_cd0.0002_epochXX.pkl  # 换成实际 best

components:
  data: predict_val
  transform: predict
  system: vm
  model: cvm_max

writer:
  __target__: vm
  save_dir: results_val_cvm_max
  save_name: denoised
```

### 6.2 test 打包提交

```yaml
# configs/task/predict_cvm_max_test.yaml
mode: predict
debug: false
load_ckpt: experiments/cvm_max/checkpoint_best_cd0.0002_epochXX.pkl
components:
  data: predict
  transform: predict
  system: vm
  model: cvm_max
writer:
  __target__: vm
  save_dir: results_cvm_max_test
  save_name: denoised
```

```bash
python run.py --task configs/task/predict_cvm_max_test.yaml --seed 123
cd results_cvm_max_test && zip -r ../result_cvm_max_epochXX.zip shapenet/
# 200 个 denoised.npy，float32 (50000,3)
```

`configs/data/predict.yaml` 中 `input_dataset_dir` 指向 **`dataset_test_noisy`**（云需 scp 或共享盘）。

---

## 七、与 vm_max 配置的对应关系

CVM 的 4 个 velocity encoder/decoder **结构**须与预训练 VM 一致：

| 字段 | vm_max | cvm_max |
|------|--------|---------|
| `edge_aggr` | max | max |
| `frame_knn` | 32 | 32 |
| `patch_size` | 1000 | 1000 |
| `feat_embedding_dim` | 256 | 256 |
| `decoder_hidden_dim` | 64 | 64 |
| `dsm_sigma` | 0.01 | 0.01 |
| `num_modules` | — | **4**（CVM 独有） |

vm_max 完整 model yaml（供 diff）：

```yaml
__target__: VelocityModule
edge_aggr: max
frame_knn: 32
patch_size: 1000
seed_k: 6
seed_k_alpha: 1
niters: 1
num_train_points: 128
feat_embedding_dim: 256
decoder_hidden_dim: 64
dsm_sigma: 0.01
```

---

## 八、常见错误排查

| 现象 | 可能原因 |
|------|----------|
| 开训无 `use vm_ckpt:` 行 | task yaml 未写 `vm_ckpt` 或路径 typo |
| OOM | batch_size 改成 8 以外；或同时跑其它 GPU 任务 |
| val_CD 一直 nan | validate 数据路径错；或 `validate_chamfer_transform` 缺失 |
| load shape mismatch | vm_ckpt 不是 max VM；或用了 cvm ckpt 当 vm_ckpt |
| benchmark 虚高、线上崩 | 用了 **niters=2** 或 sweep 参数 |
| best epoch 与线上一致性差 | 正常；以 **线上 test** 为准 |

---

## 九、学校已跑结果（对照，非云目标）

| 项目 | 学校 vm89 → cvm_max |
|------|---------------------|
| vm_ckpt | `checkpoint_best_cd0.0002_epoch89.pkl` |
| 训练 epoch | 0–99 已完成 |
| best val_CD | ~**0.000174**（epoch 89 附近） |
| benchmark val best | epoch89 **78.25**（n1） |
| 线上 best | CVM97 **72.60**（非 epoch89） |

云实验目的：验证 **vm128 底座** 能否打破 CVM97 线上上限。

---

## 十、文件清单（训练 cvm_max 全链路）

```
configs/task/train_cvm_max.yaml      # vm_ckpt、trainer
configs/model/cvm_max.yaml           # 网络结构
configs/data/train_cvm_max.yaml      # 数据路径、batch
configs/transform/cvm.yaml           # 增广 + validate_chamfer
configs/system/cvm_max.yaml          # ckpt 目录

src/model/cvm.py                     # CoupledVMArch、predict_step
src/system/cvm.py                    # CVMSystem、validate_chamfer
run.py                               # vm_ckpt 注入、train 入口

datalist/train.txt
datalist/validate.txt
/home/dataset_train/...              # mesh

experiments/vm_max/<128 ckpt>        # 输入
experiments/cvm_max/                 # 输出
log/train_cvm_max.log                # 建议 tee
```

**推理/评测额外**：

```
configs/data/predict_val.yaml
configs/data/predict.yaml
configs/transform/predict.yaml
configs/task/predict_val_cvm_max.yaml
configs/task/predict_cvm_max_test.yaml
evaluate.py
scripts/export_val_benchmark.py
benchmark_val/                       # 本地评测集
dataset_test_noisy/                  # test predict
```

---

## 十一、学校机器续训 cvm_max（并行实验，可选）

与云 **独立**：仍用 **vm89**，从 epoch 99 往后训。

```yaml
# configs/task/train_cvm_max.yaml 学校续训示例
vm_ckpt: experiments/vm_max/checkpoint_best_cd0.0002_epoch89.pkl
load_ckpt: experiments/cvm_max/checkpoint_99.pkl   # 或 checkpoint_best_cd...
trainer:
  epochs: 150
  start_epoch: 100
  save_every: 10
  val_chamfer_every: 1
  val_chamfer_max_samples: 20
```

**续训逻辑**：`load_ckpt` 在 `run.py` 里于 model 构建后 `model.load()`，**覆盖** init 时的 vm 权重；此时 `vm_ckpt` 仅首次 init 用，最终以 `load_ckpt` 为准。续训时保留 `vm_ckpt` 与学校一致即可。

```bash
screen -dmS train_cvm_max_resume bash -lc '
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate jittor
  cd /home/cslab/CG
  python run.py --task configs/task/train_cvm_max.yaml --seed 123 \
    2>&1 | tee -a log/train_cvm_max.log
'
```

---

## 十二、交付回传

训完请回传：

1. `log/train_cvm_max.log` 末尾 20 行（含 best_CD 轨迹）
2. `ls -lh experiments/cvm_max/checkpoint_best*.pkl`
3. 若有 benchmark：`grep 最终得分 log/eval_*_score.log`
4. 实际使用的 `vm_ckpt` 完整路径

---

## 十三、禁止事项（血泪教训）

1. **禁止** `niters: 2` 或任何推理 sweep（本地 val 83 ≠ 线上 51）。
2. **禁止** 用 benchmark_val 排名直接代替线上 test。
3. **禁止** 把 `cvm_max` ckpt 当作 `vm_ckpt`。
4. **禁止** 在未核对 yaml 的情况下改 `num_modules`、`consistency_weight`、`edge_aggr`。
5. 云、校 **各改各的** task yaml；不要在学校仓库提交云专用路径。

---

*文档生成自学校 `/home/cslab/CG` 仓库配置快照；若 git 有更新，开训前请 `diff` 核对 `configs/{task,model,data,transform,system}/` 下 cvm_max 相关文件。*
