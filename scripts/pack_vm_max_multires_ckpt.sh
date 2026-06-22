#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cslab/CG"
cd "$ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate jittor
export CUDA_VISIBLE_DEVICES=0

CKPT="${CKPT:?set CKPT=...}"
DATA_COMPONENT="${DATA_COMPONENT:-predict_fast}"
[[ -f "$CKPT" ]] || { echo "missing $CKPT" >&2; exit 1; }

TAG="$(python3 - <<PY
import os, re
b = os.path.basename("$CKPT")
m = re.search(r"epoch(\d+)", b) or re.search(r"checkpoint_(\d+)\.pkl", b)
ep = m.group(1) if m else "unknown"
m2 = re.search(r"cd([\d.]+)_epoch", b)
cd = m2.group(1) if m2 else "unknown"
label = os.environ.get("PACK_LABEL", "")
suffix = f"_{label}" if label else ""
print(f"epoch{ep}_cd{cd}{suffix}")
PY
)"

OUT_DIR="results_vm_max_multires_test_${TAG}"
ZIP="result_vm_max_multires_${TAG}.zip"
PRED_LOG="log/predict_vm_max_multires_test_${TAG}.log"
TASK="configs/task/predict_vm_max_multires_test_${TAG}.yaml"
SUMMARY="${SUMMARY:-log/pack_vm_max_multires_${TAG}_summary.txt}"

mkdir -p log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$SUMMARY"; }
: > "$SUMMARY"
log "PACK VM multires ckpt=$CKPT"

cat > "$TASK" <<EOF
mode: predict
debug: false
load_ckpt: ${CKPT}
components:
  data: ${DATA_COMPONENT}
  transform: predict
  system: vm
  model: vm_max
writer:
  __target__: vm
  save_dir: ${OUT_DIR}
  save_name: denoised
EOF

python run.py --task "$TASK" --seed 123 2>&1 | tee "$PRED_LOG"
rm -f "$ZIP"
(cd "$OUT_DIR" && zip -r -1 "../$ZIP" shapenet/)
log "DONE zip=$ZIP"
echo "$ZIP"
