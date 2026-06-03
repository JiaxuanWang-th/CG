#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/cslab/CG"
cd "$ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate jittor
export CUDA_VISIBLE_DEVICES=0

CKPT="experiments/straightpcf_max/checkpoint_best_cd0.0001_iter75000.pkl"
OUT_DIR="results_spcf_max_test_iter75000"
ZIP="result_spcf_max_iter75000.zip"
LOG="log/predict_spcf_max_test_iter75000.log"

mkdir -p log
log() { echo "[$(date '+%F %T')] $*"; }

cat > configs/task/predict_spcf_max_test_iter75000.yaml <<EOF
mode: predict
debug: false

load_ckpt: ${CKPT}

components:
  data: predict
  transform: predict
  system: vm
  model: straightpcf_max

writer:
  __target__: vm
  save_dir: ${OUT_DIR}
  save_name: denoised
EOF

log "PREDICT test: ckpt=${CKPT} niters=1 tot_its=3 seed_k=6"
python run.py --task configs/task/predict_spcf_max_test_iter75000.yaml --seed 123 \
  2>&1 | tee "$LOG"

log "PACK zip -> ${ZIP}"
rm -f "$ZIP"
(cd "$OUT_DIR" && zip -r "../$ZIP" shapenet/)

log "DONE: $(ls -lh $ZIP)"
find "$OUT_DIR" -name 'denoised.npy' | wc -l | xargs -I{} echo "denoised.npy count: {}"
