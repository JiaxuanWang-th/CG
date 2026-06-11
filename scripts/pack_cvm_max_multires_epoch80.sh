#!/usr/bin/env bash
# CVM max multires：打包 test zip（当前 best @ ep80）
set -euo pipefail

ROOT="/home/cslab/CG"
cd "$ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate jittor
export CUDA_VISIBLE_DEVICES=0

EPOCH="${EPOCH:-80}"
CKPT="experiments/cvm_max_multires/checkpoint_best_cd0.0002_epoch${EPOCH}.pkl"
if [[ ! -f "$CKPT" ]]; then
  CKPT="experiments/cvm_max_multires/checkpoint_${EPOCH}.pkl"
fi

OUT_DIR="results_cvm_max_multires_test_epoch${EPOCH}"
ZIP="result_cvm_max_multires_epoch${EPOCH}.zip"
LOG="log/predict_cvm_max_multires_test_epoch${EPOCH}.log"
TASK="configs/task/predict_cvm_max_multires_test_epoch${EPOCH}.yaml"
SUMMARY="log/pack_cvm_max_multires_epoch${EPOCH}_summary.txt"

mkdir -p log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$SUMMARY"; }

: > "$SUMMARY"
log "CVM max multires pack | epoch=${EPOCH} ckpt=${CKPT}"
log "reference_online: CVM125=72.72 | VM multires ep99=67.36"

cat > "$TASK" <<EOF
mode: predict
debug: false

load_ckpt: ${CKPT}

components:
  data: predict
  transform: predict
  system: vm
  model: cvm_max

writer:
  __target__: vm
  save_dir: ${OUT_DIR}
  save_name: denoised
EOF

log "PREDICT test (niters=1 tot_its=3 seed_k=6) -> ${OUT_DIR}"
python run.py --task "$TASK" --seed 123 2>&1 | tee "$LOG"

log "ZIP -> ${ZIP}"
rm -f "$ZIP"
(cd "$OUT_DIR" && zip -r "../$ZIP" shapenet/)

n="$(find "$OUT_DIR" -name 'denoised.npy' | wc -l)"
log "DONE: zip=$(ls -lh "$ZIP" | awk '{print $5}') denoised.npy=${n}"
