#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/CG
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate jittor
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDA_VISIBLE_DEVICES=0

CKPT="experiments/vm_max_multires/checkpoint_99.pkl"
OUT_DIR="results_vm_max_multires_test_epoch99"
ZIP="result_vm_max_multires_epoch99.zip"
LOG="log/predict_vm_max_multires_test_epoch99.log"
TASK="configs/task/predict_vm_max_multires_test_epoch99.yaml"

mkdir -p log
log() { echo "[$(date '+%F %T')] $*"; }

log "PREDICT test: ckpt=${CKPT}"
python run.py --task "$TASK" --seed 123 2>&1 | tee "$LOG"

log "PACK zip -> ${ZIP}"
rm -f "$ZIP"
(cd "$OUT_DIR" && zip -r "../$ZIP" shapenet/)

n=$(find "$OUT_DIR" -name 'denoised.npy' | wc -l)
log "DONE: $(ls -lh $ZIP)"
log "denoised.npy count: ${n}"
