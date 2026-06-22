#!/usr/bin/env bash
# Run test inference and pack submission zip (defaults to bundled SPCF checkpoint).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CKPT="${CKPT:-checkpoints/spcf/checkpoint_best.pkl}"
OUT_DIR="${OUT_DIR:-output/predictions}"
ZIP="${ZIP:-submission.zip}"
SEED="${SEED:-123}"

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint not found: $CKPT" >&2
  exit 1
fi

TASK="configs/task/_generated_predict.yaml"
mkdir -p output log "$(dirname "$TASK")"

cat > "$TASK" <<EOF
mode: predict
debug: false

load_ckpt: ${CKPT}

components:
  data: predict
  transform: predict
  system: infer
  model: spcf

writer:
  __target__: vm
  save_dir: ${OUT_DIR}
  save_name: denoised
EOF

echo "[pack] predict ckpt=$CKPT -> $OUT_DIR"
python run.py --task "$TASK" --seed "$SEED" 2>&1 | tee log/predict.log

echo "[pack] zip -> $ZIP"
rm -f "$ROOT/$ZIP"
(cd "$ROOT/$OUT_DIR" && zip -r -1 "$ROOT/$ZIP" shapenet/)

n="$(find "$OUT_DIR" -name 'denoised.npy' | wc -l)"
echo "[pack] done: $ZIP ($n denoised.npy files)"
