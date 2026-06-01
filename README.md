# 点云降噪赛题 Baseline

## 环境安装
```bash
# 安装计图
conda create -n jittor python=3.9 -y
conda activate jittor
conda install -c conda-forge gcc=10 gxx=10 -y # 确保gcc、g++版本不高于10
conda install -c conda-forge libgomp -y # 确保OpenMP runtime存在

# 安装依赖
python -m pip install -r requirements.txt
pip install jittor numpy trimesh scipy omegaconf point-cloud-utils
```

## 数据准备
1. 将训练数据 `dataset_train.tar.gz` 解压（本机数据在 `/home/dataset_train`）：
   ```bash
   # 例如解压到 /home/dataset_train
   tar xzf dataset_train.tar.gz -C /home
   ```
   解压后目录：`/home/dataset_train/shapenet/<synset_id>/<model_id>/models/model_normalized.obj`  
   若路径不同，请修改 `configs/data/train.yaml` 中的 `input_dataset_dir`。

2. 将测试数据 `dataset_test_noisy.zip` 解压到本目录下：
   ```bash
   unzip dataset_test_noisy.zip
   ```
   解压后目录：`dataset_test_noisy/shapenet/<synset_id>/<model_id>/noisy.npy`

## 训练

### Baseline：单 VelocityModule
```bash
python run.py --task configs/task/train_vm.yaml
```

### StraightPCF 三阶段（推荐冲榜）
```bash
# 1) VM
python run.py --task configs/task/train_vm.yaml

# 2) CVM：修改 configs/task/train_cvm.yaml 中的 vm_ckpt 指向 VM 权重
python run.py --task configs/task/train_cvm.yaml

# 3) StraightPCF：修改 configs/task/train_straightpcf.yaml 中的 cvm_ckpt 指向 CVM 权重
python run.py --task configs/task/train_straightpcf.yaml
```

详见 [docs/STRAIGHTPCF_PORT.md](docs/STRAIGHTPCF_PORT.md)（模块清单与取舍说明）。

训练权重保存在 `experiments/` 目录下。

## 推理（生成提交文件）

StraightPCF 提交：
```bash
# 修改 configs/task/predict_straightpcf.yaml 的 load_ckpt
python run.py --task configs/task/predict_straightpcf.yaml
```

Baseline VM：
修改 `configs/task/predict_vm.yaml` 中的 `load_ckpt` 为你的最佳权重路径，然后运行：
```bash
python run.py --task configs/task/predict_vm.yaml
```
降噪结果保存在 `results/` 目录下，格式为 `.npy` (float32, shape (N,3))。

## 打包提交
```bash
cd results/dataset_test_noisy
zip -r ../../result.zip shapenet/
```

## 提交格式
每个测试样本一个 `denoised.npy`，目录结构与测试集一致，打包为 `result.zip`：
```
result.zip
  shapenet/
    <synset_id>/
      <model_id>/
        denoised.npy    # np.float32, shape (N, 3)
```

## 本地评测（需要 GT 数据，仅组委会持有）
```bash
python evaluate.py \
    --pred_dir ./results/dataset_test_noisy \
    --gt_dir ./test_gt \
    --noisy_dir ./dataset_test_noisy \
    --mesh_dir /home/dataset_train \
    --workers 8
```
