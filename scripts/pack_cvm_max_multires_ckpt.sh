#!/usr/bin/env bash
# CVM max multires：按 ckpt 路径打包 test zip（用于 best 提交 / SPCF 前验证）
set -euo pipefail

ROOT="/home/cslab/CG"
cd "$ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate jittor
export CUDA_VISIBLE_DEVICES=0

CKPT="${CKPT:?set CKPT=experiments/cvm_max_multires/checkpoint_best_....pkl}"
DATA_COMPONENT="${DATA_COMPONENT:-predict_fast}"

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: ckpt not found: $CKPT" >&2
  exit 1
fi

TAG="$(python3 - <<PY
import os, re
base = os.path.basename("$CKPT")
m = re.search(r"epoch(\d+)", base) or re.search(r"checkpoint_(\d+)\.pkl", base)
ep = m.group(1) if m else "unknown"
m2 = re.search(r"cd([\d.]+)_epoch", base)
cd = m2.group(1) if m2 else "unknown"
label = os.environ.get("PACK_LABEL", "")
suffix = f"_{label}" if label else ""
print(f"epoch{ep}_cd{cd}{suffix}")
PY
)"

OUT_DIR="results_cvm_max_multires_test_${TAG}"
ZIP="result_cvm_max_multires_${TAG}.zip"
PRED_LOG="log/predict_cvm_max_multires_test_${TAG}.log"
TASK="configs/task/predict_cvm_max_multires_test_${TAG}.yaml"
SUMMARY="log/pack_cvm_max_multires_${TAG}_summary.txt"

mkdir -p log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$SUMMARY"; }

: > "$SUMMARY"
log "PACK CVM multires | ckpt=${CKPT} tag=${TAG}"

cat > "$TASK" <<EOF
mode: predict
debug: false

load_ckpt: ${CKPT}

components:
  data: ${DATA_COMPONENT}
  transform: predict
  system: vm
  model: cvm_max

writer:
  __target__: vm
  save_dir: ${OUT_DIR}
  save_name: denoised
EOF

log "PREDICT test -> ${OUT_DIR}"
python run.py --task "$TASK" --seed 123 2>&1 | tee "$PRED_LOG"

log "ZIP -> ${ZIP}"
rm -f "$ZIP"
(cd "$OUT_DIR" && zip -r -1 "../$ZIP" shapenet/)

n="$(find "$OUT_DIR" -name 'denoised.npy' | wc -l)"
log "DONE: zip=$(ls -lh "$ZIP" | awk '{print $5}') denoised.npy=${n}"
echo "$ZIP"
