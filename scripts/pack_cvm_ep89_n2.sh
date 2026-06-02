#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/cslab/CG"
cd "$ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate jittor
export CUDA_VISIBLE_DEVICES=0

CKPT="experiments/cvm_max/checkpoint_89.pkl"
OUT_DIR="results_cvm_max_test_ep89_n2"
ZIP="result_cvm_max_ep89_n2.zip"
LOG="predict_cvm_max_test_ep89_n2.log"

log() { echo "[$(date '+%F %T')] $*"; }

# model: epoch89 + n2 (niters=2, tot_its=3, seed_k=6)
python - <<'PY'
from pathlib import Path
base = Path("configs/model/cvm_max.yaml").read_text().splitlines()
out = []
for line in base:
    if line.startswith("tot_its:"):
        out.append("tot_its: 3")
    elif line.startswith("seed_k:"):
        out.append("seed_k: 6")
    elif line.startswith("niters:"):
        out.append("niters: 2")
    else:
        out.append(line)
if not any(l.startswith("niters:") for l in out):
    out.append("niters: 2")
Path("configs/model/cvm_max_predict_n2.yaml").write_text("\n".join(out) + "\n")
PY

cat > configs/task/predict_cvm_max_test_ep89_n2.yaml <<EOF
mode: predict
debug: false

load_ckpt: ${CKPT}

components:
  data: predict
  transform: predict
  system: vm
  model: cvm_max_predict_n2

writer:
  __target__: vm
  save_dir: ${OUT_DIR}
  save_name: denoised
EOF

log "PREDICT test set: ckpt=${CKPT} niters=2 tot_its=3 seed_k=6"
python run.py --task configs/task/predict_cvm_max_test_ep89_n2.yaml --seed 123 \
  2>&1 | tee "$LOG"

log "PACK zip -> ${ZIP}"
rm -f "$ZIP"
(cd "$OUT_DIR" && zip -r "../$ZIP" shapenet/)

log "DONE: $(ls -lh $ZIP)"
find "$OUT_DIR" -name 'denoised.npy' | wc -l | xargs -I{} log "denoised.npy count: {}"
