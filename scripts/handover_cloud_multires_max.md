# 云端交接文档：multires 全链路 VM → CVM → SPCF max

> **文件路径**：`scripts/handover_cloud_multires_max.md`（仓库根目录下 `scripts/`，已纳入 git；`HANDOVER*` 在 `.gitignore` 中，请勿只建根目录副本。）

**交接人**：Jiaxuan  
**执行机器**：`/home/ubuntu/CG`（RTX 4090/5090，conda 环境 `jittor`）  
**Git 分支**：`dev/multires-cloud`（**不要**在学校主线 `dev/straightpcf-cvm` 上改 multires 配置）  
**日期**：2026-06-05  

---

## 一、实验目标与边界

| 项目 | 说明 |
|------|------|
| **要做什么** | 官方 StraightPCF 式 **多分辨率训练**：每个 train sample 随机 **10k / 30k / 50k** 点；**VM → CVM → SPCF** 三阶段 **从头重训** |
| **与学校 32768 线的关系** | 学校继续 `dev/straightpcf-cvm`（固定 32768 + ep125 SPCF 等）；本实验 **独立分支、独立权重目录**，互不混用 ckpt |
| **成功标准** | 线上 test **超过** 学校当前 SPCF best **73.53**（或 CVM/SPCF 分阶段有明确线上增益再提交） |
| **失败也 OK** | val 无提升或线上 ≤ 73.53 → 归档，不必占学校提交槽 |

**不要做什么**：

- **不要**加载 `experiments/vm_max/`、`cvm_max/`、`straightpcf_max/`（32768 单分辨率线）的 ckpt 做本实验预训练或续训  
- **不要**改推理 `niters>1`（平台 n2 已翻车）；固定 **n1, tot_its=3, seed_k=6**  
- **不要**对 multires SPCF 做 250k 式长续训（学校已证 200k/250k 线上更差）  
- **不要**用本地 val_CD / benchmark 绝对分代替 **线上 test** 选提交  

---

## 二、multires 与单分辨率差在哪

| 项 | 学校单分辨率 (`vm.yaml` 等) | 本实验 (`*_multires.yaml`) |
|----|------------------------------|----------------------------|
| **Train 采样** | 固定 `num_samples: 32768` | `sample_multires` → `[10000, 30000, 50000]` 随机 |
| **Val / val_CD** | 32768 | **仍 32768**（与 max 线可比） |
| **Test 推理** | 读平台 `noisy.npy`（50000 点） | **相同**，不改 `predict.yaml` |
| **权重目录** | `experiments/*_max/` | `experiments/*_max_multires/` |

训练进 GPU 的仍是 **patch (1, 1000, 3)**，batch 8/16 一般可保持不变。

---

## 三、前置检查（开训前必做）

```bash
cd /home/ubuntu/CG
git fetch origin
git checkout dev/multires-cloud
git pull origin dev/multires-cloud

source ~/miniconda3/etc/profile.d/conda.sh   # 路径按机器实际调整
conda activate jittor

nvidia-smi
python -c "import jittor as jt; print('jittor', jt.__version__, 'cuda', jt.flags.use_cuda)"
```

### 3.1 代码与 augment

```bash
grep -q sample_multires src/data/augment.py && echo "AugmentSampleMultires OK"
test -f configs/transform/vm_multires.yaml && test -f configs/task/train_vm_max_multires.yaml && echo "multires configs OK"
```

### 3.2 训练数据

| 项 | 路径 |
|----|------|
| Mesh 根目录 | `/home/dataset_train` |
| 训练列表 | `./datalist/train.txt` |
| 验证列表 | `./datalist/validate.txt` |

```bash
test -f /home/dataset_train/datalist/train.txt && echo "datalist OK"
```

### 3.3 磁盘

三阶段 ckpt + log 粗估 **< 5GB**；predict 打包每份 zip ~200 样本，预留 **10GB+** 空闲。

---

## 四、三阶段配置一览

| Stage | Task 入口 | Transform | System（ckpt 目录） | 训练量 |
|-------|-----------|-----------|---------------------|--------|
| 1 VM | `configs/task/train_vm_max_multires.yaml` | `vm_multires` | `experiments/vm_max_multires/` | **100 epoch** |
| 2 CVM | `configs/task/train_cvm_max_multires.yaml` | `cvm_multires` | `experiments/cvm_max_multires/` | **150 epoch** |
| 3 SPCF | `configs/task/train_straightpcf_max_multires.yaml` | `straightpcf_official_multires` | `experiments/straightpcf_max_multires/` | **90000 iter** |

**Data（云路径）**：`train_*_max_multires_cloud.yaml` → `input_dataset_dir: /home/dataset_train`

**Model**：与学校 max 栈相同（`vm_max` / `cvm_max` / `straightpcf_max` yaml，4 modules、max EdgeConv、Stage3 全参）。

---

## 五、启动命令（screen 后台）

### Stage 1 — VM max multires（从零，无 load_ckpt）

```bash
mkdir -p log
screen -dmS vm_multires bash -c '
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate jittor &&
  cd /home/ubuntu/CG &&
  python run.py --task configs/task/train_vm_max_multires.yaml --seed 123 \
    2>&1 | tee log/train_vm_max_multires.log
'
screen -r vm_multires   # 查看；Ctrl+A D  detach
```

**日志关注**：每 epoch 末尾 `val_CD` / `best_CD`；best 写入  
`experiments/vm_max_multires/checkpoint_best_cd*_epoch*.pkl`

**耗时粗估**：~10–15 min/epoch × 100 ≈ **17–25 h**

---

### Stage 2 — CVM max multires

Stage 1 结束后，**改 task 里的 `vm_ckpt`** 为 Stage 1 的 **best**（勿用占位 `epoch99` 若 best 在其他 epoch）：

```bash
ls -lh experiments/vm_max_multires/checkpoint_best*.pkl
# 编辑 configs/task/train_cvm_max_multires.yaml → vm_ckpt: <上一步 best 路径>
```

```bash
screen -dmS cvm_multires bash -c '
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate jittor &&
  cd /home/ubuntu/CG &&
  python run.py --task configs/task/train_cvm_max_multires.yaml --seed 123 \
    2>&1 | tee log/train_cvm_max_multires.log
'
```

- **不要**设 `load_ckpt`（除非 CVM 中断续训同一 multires run）  
- **不要**用 `experiments/cvm_max/` 旧权重  

**耗时粗估**：~15–20 min/epoch × 150 ≈ **1.5–2 天**

**CVM 中断续训**（可选）：

```yaml
# train_cvm_max_multires.yaml
load_ckpt: experiments/cvm_max_multires/checkpoint_80.pkl
trainer:
  start_epoch: 81
  epochs: 150
```

---

### Stage 3 — SPCF max multires

Stage 2 结束后，**改 `cvm_ckpt`** 为 CVM multires **best**（yaml 里占位 `epoch149` 仅示例，以实际 best 文件为准）：

```bash
ls -lh experiments/cvm_max_multires/checkpoint_best*.pkl
# 编辑 configs/task/train_straightpcf_max_multires.yaml → cvm_ckpt: <best>
```

```bash
screen -dmS spcf_multires bash -c '
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate jittor &&
  cd /home/ubuntu/CG &&
  python run.py --task configs/task/train_straightpcf_max_multires.yaml --seed 123 \
    2>&1 | tee log/train_straightpcf_max_multires.log
'
```

**耗时粗估**：90000 iter ≈ **~1 天**（与学校 SPCF max 同量级）

**SPCF 线上选 ckpt 经验（32768 线）**：best 常在 **iter 75000** 附近，不必迷信训满；训完后打包 **60000 / 75000 / 90000** 各测一次（占提交槽）。

---

## 六、训练内验证说明

| 指标 | 含义 | 能否直接选提交 |
|------|------|----------------|
| `val_patch_loss` | patch 训练 loss | 仅看趋势 |
| **`val_CD`** | 20 样本整云 Chamfer（`validate_chamfer_transform`，32768 点） | **粗筛 ckpt** |
| 线上 test | 200 样本，50000 点 | **最终依据** |

学校经验：**val 第一名 ≠ 线上第一名**（例：CVM ep125 线上 > ep97；SPCF 75k > 200k）。

---

## 七、训完 — 打包 test 提交

### 7.1 仅 CVM multires（Stage 2 完成后可先测 CVM）

复制学校 `predict_cvm_max_test.yaml`，改 `load_ckpt` 与 `save_dir`，例如：

```yaml
# configs/task/predict_cvm_max_multires_test_epochXX.yaml
mode: predict
debug: false
load_ckpt: experiments/cvm_max_multires/checkpoint_best_cd0.0002_epochXX.pkl
components:
  data: predict
  transform: predict
  system: vm
  model: cvm_max
writer:
  __target__: vm
  save_dir: results_cvm_max_multires_test_epochXX
  save_name: denoised
```

```bash
python run.py --task configs/task/predict_cvm_max_multires_test_epochXX.yaml --seed 123 \
  2>&1 | tee log/predict_cvm_max_multires_test_epochXX.log

cd results_cvm_max_multires_test_epochXX && zip -r ../result_cvm_max_multires_epochXX.zip shapenet/
# 上传 result_cvm_max_multires_epochXX.zip
```

### 7.2 SPCF multires（Stage 3）

```yaml
# configs/task/predict_spcf_max_multires_test_iter75000.yaml
load_ckpt: experiments/straightpcf_max_multires/checkpoint_best_cd0.0001_iter75000.pkl
# model: straightpcf_max（niters: 1 已在 model yaml）
writer:
  save_dir: results_spcf_max_multires_test_iter75000
```

```bash
python run.py --task configs/task/predict_spcf_max_multires_test_iter75000.yaml --seed 123
cd results_spcf_max_multires_test_iter75000 && zip -r ../result_spcf_max_multires_iter75000.zip shapenet/
```

**推理参数**：确认 `configs/model/straightpcf_max.yaml` 中 `niters: 1`，**不要**改 n2。

---

## 八、与学校机器分工

| 机器 | 分支 | 任务 |
|------|------|------|
| **学校** | `dev/straightpcf-cvm` | ep149 / ep125 CVM 选型、标准 SPCF 90k、日常提交槽 |
| **云** | `dev/multires-cloud` | 本 multires 全链路；有 ckpt 再线上测，不必和学校抢同一实验 |

同步方式：云 `git pull` 即可；**不要**把 multires 合并进学校主线，除非 Jiaxuan 明确要合并。

---

## 九、常见问题

| 现象 | 处理 |
|------|------|
| `sample_multires` KeyError | 未在 `dev/multires-cloud`；或 `augment.py` 过旧，`git pull` |
| CVM load vm 失败 shape mismatch | `vm_ckpt` 用了 `vm_max` 而非 `vm_max_multires` |
| SPCF load cvm 失败 | `cvm_ckpt` 用了 `cvm_max` 而非 `cvm_max_multires` |
| val_CD 长期不降 | 先查数据路径；multires 收敛可能慢 10–20 epoch，属正常 |
| OOM | 优先 `num_workers: 0`（已设）；仍 OOM 则将 data yaml 的 `batch_size` 8→4 |
| 50k 采样 epoch 变慢 | 预期行为；50k 样本 CPU 采样 + KD-tree 更慢 |

---

## 十、完成后反馈学校（请复制发 Jiaxuan）

1. 各阶段 **best ckpt 路径** + 对应 **val_CD**（或 SPCF iter + val Chamfer）  
2. **线上 test 分数**（若已提交）  
3. 三份 log 末尾 30 行：`train_vm_max_multires.log`、`train_cvm_max_multires.log`、`train_straightpcf_max_multires.log`  
4. `ls -lh experiments/*_multires/checkpoint_best*.pkl`  

---

## 十一、文件清单（本分支新增，便于 diff）

```
src/data/augment.py                          # AugmentSampleMultires
configs/transform/vm_multires.yaml
configs/transform/cvm_multires.yaml
configs/transform/straightpcf_official_multires.yaml
configs/data/train_vm_max_multires_cloud.yaml
configs/data/train_cvm_max_multires_cloud.yaml
configs/data/train_spcf_max_multires_cloud.yaml
configs/system/vm_max_multires.yaml
configs/system/cvm_max_multires.yaml
configs/system/straightpcf_max_multires.yaml
configs/task/train_vm_max_multires.yaml
configs/task/train_cvm_max_multires.yaml
configs/task/train_straightpcf_max_multires.yaml
```

---

*文档基于学校 `/home/cslab/CG` 仓库 `dev/multires-cloud` @ 2026-06-05。开训前请 `git pull` 核对 commit 是否含 CVM **150 epoch** 配置。*
