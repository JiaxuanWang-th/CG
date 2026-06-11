#!/usr/bin/env bash
# SPCF max multires：按 iter 或 ckpt 路径打包 test zip
set -euo pipefail

ROOT="/home/cslab/CG"
cd "$ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate jittor
export CUDA_VISIBLE_DEVICES=0

CKPT_DIR="experiments/straightpcf_max_multires"
DATA_COMPONENT="${DATA_COMPONENT:-predict_fast}"

if [[ -n "${CKPT:-}" ]]; then
  pick="$CKPT"
elif [[ -n "${ITER:-}" ]]; then
  best="${CKPT_DIR}/checkpoint_best_cd0.0001_iter${ITER}.pkl"
  periodic="${CKPT_DIR}/checkpoint_iter${ITER}.pkl"
  if [[ -f "$best" ]]; then pick="$best"
  elif [[ -f "$periodic" ]]; then pick="$periodic"
  else echo "ERROR: no ckpt for iter ${ITER}" >&2; exit 1; fi
else
  echo "ERROR: set CKPT=... or ITER=..." >&2
  exit 1
fi

TAG="$(python3 - <<PY
import os, re
base = os.path.basename("$pick")
m = re.search(r"iter(\d+)", base)
print(f"iter{m.group(1)}" if m else "custom")
PY
)"

OUT_DIR="results_spcf_max_multires_test_${TAG}"
ZIP="result_spcf_max_multires_${TAG}.zip"
PRED_LOG="log/predict_spcf_max_multires_test_${TAG}.log"
TASK="configs/task/predict_spcf_max_multires_test_${TAG}.yaml"
SUMMARY="${SUMMARY:-log/pack_spcf_max_multires_${TAG}_summary.txt}"

mkdir -p log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$SUMMARY"; }

: > "$SUMMARY"
log "PACK SPCF multires | ckpt=${pick} | reference: CVM237=73.53"

cat > "$TASK" <<EOF
mode: predict
debug: false

load_ckpt: ${pick}

components:
  data: ${DATA_COMPONENT}
  transform: predict
  system: vm
  model: straightpcf_max

writer:
  __target__: vm
  save_dir: ${OUT_DIR}
  save_name: denoised
EOF

log "PREDICT test -> ${OUT_DIR}"
python run.py --task "$TASK" --seed 123 2>&1 | tee "$PRED_LOG"

log "ZIP -> ${ZIP} (-1 fast)"
rm -f "$ZIP"
(cd "$OUT_DIR" && zip -r -1 "../$ZIP" shapenet/)

n="$(find "$OUT_DIR" -name 'denoised.npy' | wc -l)"
log "DONE: zip=$(ls -lh "$ZIP" | awk '{print $5}') denoised.npy=${n}"
echo "$ZIP"
