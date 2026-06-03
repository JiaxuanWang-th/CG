#!/usr/bin/env bash
# 续训 100→149 结束后：打包两个 test 提交 zip
#   1) val_CD best（续训段内 log 最低 best_CD）
#   2) 接近 150 的最后一档：checkpoint_149（类比首轮 ep97 接近 100）
#
# 用法：
#   bash scripts/pack_cvm_max_resume_two.sh          # 等到训完再打包
#   WAIT=0 bash scripts/pack_cvm_max_resume_two.sh   # 立即打包（ckpt 须已齐）
set -euo pipefail

ROOT="/home/cslab/CG"
cd "$ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate jittor
export CUDA_VISIBLE_DEVICES=0

WAIT="${WAIT:-1}"
LOG="log/train_cvm_max.log"
CKPT_DIR="experiments/cvm_max"
SUMMARY="log/pack_cvm_max_resume_two_summary.txt"

log() { echo "[$(date '+%F %T')] $*"; }

wait_for_training() {
  log "Waiting for resume training to finish (epoch 149 + checkpoint_149.pkl)..."
  while true; do
    if [[ -f "${CKPT_DIR}/checkpoint_149.pkl" ]] && grep -q "\[Epoch 149\]" "$LOG" 2>/dev/null; then
      if ! screen -ls 2>/dev/null | grep -q "train_cvm_max_resume"; then
        log "Training screen gone and ep149 done."
        break
      fi
      # screen 还在但可能刚存盘，再等一轮 log 不再更新 train step
      sleep 30
      if grep -q "\[Epoch 149\]" "$LOG" && [[ -f "${CKPT_DIR}/checkpoint_149.pkl" ]]; then
        log "ep149 checkpoint present; proceed."
        break
      fi
    fi
    last_ep=$(python3 - <<'PY' 2>/dev/null || echo "?"
import re
from pathlib import Path
text = Path("log/train_cvm_max.log").read_text(errors="replace")
rows = [int(m.group(1)) for m in re.finditer(r"\[Epoch (\d+)\] train_patch_loss=", text)]
print(rows[-1] if rows else "?")
PY
)
    log "  ... last completed epoch in log: ${last_ep}; sleeping 120s"
    sleep 120
  done
  sleep 10
}

pick_best_ckpt() {
  python3 - <<'PY'
import re
from pathlib import Path

log = Path("log/train_cvm_max.log").read_text(errors="replace")
best_ep, best_cd = None, float("inf")
for m in re.finditer(
    r"\[Epoch (\d+)\] train_patch_loss=[\d.]+ \| val_patch_loss=[\d.]+ \| val_CD=[\d.]+ \| best_CD=([\d.]+)",
    log,
):
    ep, cd = int(m.group(1)), float(m.group(2))
    if ep >= 100 and cd < best_cd:
        best_ep, best_cd = ep, cd

ckpt_dir = Path("experiments/cvm_max")
if best_ep is None:
    raise SystemExit("No resume epoch (>=100) found in train log")

candidates = sorted(ckpt_dir.glob(f"checkpoint_best_cd*_epoch{best_ep}.pkl"))
if candidates:
    ckpt = candidates[0]
else:
    ckpt = ckpt_dir / f"checkpoint_{best_ep}.pkl"
    if not ckpt.is_file():
        raise SystemExit(f"Missing ckpt for best ep{best_ep}: {ckpt}")

print(f"{best_ep}\t{best_cd:.6f}\t{ckpt}")
PY
}

pack_one() {
  local tag="$1"
  local ckpt="$2"
  local out_dir="results_cvm_max_test_${tag}"
  local zip_name="result_cvm_max_${tag}.zip"
  local task_yaml="configs/task/predict_cvm_max_test_${tag}.yaml"
  local pred_log="log/predict_cvm_max_test_${tag}.log"

  if [[ ! -f "$ckpt" ]]; then
    log "ERROR: missing ckpt: $ckpt"
    return 1
  fi

  cat > "$task_yaml" <<EOF
mode: predict
debug: false

load_ckpt: ${ckpt}

components:
  data: predict
  transform: predict
  system: vm
  model: cvm_max

writer:
  __target__: vm
  save_dir: ${out_dir}
  save_name: denoised
EOF

  log "PREDICT test: tag=${tag} ckpt=${ckpt} (niters=1 tot_its=3 seed_k=6)"
  python run.py --task "$task_yaml" --seed 123 2>&1 | tee "$pred_log"

  log "PACK -> ${zip_name}"
  rm -f "$zip_name"
  (cd "$out_dir" && zip -r "../${zip_name}" shapenet/)

  local n
  n=$(find "$out_dir" -name 'denoised.npy' | wc -l)
  log "DONE ${tag}: ${zip_name} (${n} files)"
  echo -e "${tag}\t${ckpt}\t${zip_name}\t${n}" >> "$SUMMARY"
}

mkdir -p log
: > "$SUMMARY"
echo -e "tag\tckpt\tzip\tnum_files" >> "$SUMMARY"

if [[ "$WAIT" == "1" ]]; then
  wait_for_training
else
  log "WAIT=0: skip wait, pack now"
fi

read -r BEST_EP BEST_CD BEST_CKPT <<< "$(pick_best_ckpt | tr '\t' ' ')"
LATE_EP=149
LATE_CKPT="${CKPT_DIR}/checkpoint_${LATE_EP}.pkl"

log "Best (resume):  ep${BEST_EP} best_CD=${BEST_CD} -> ${BEST_CKPT}"
log "Late (near150): ep${LATE_EP} (final resume epoch) -> ${LATE_CKPT}"

{
  echo "pack_time: $(date '+%F %T')"
  echo "best_ep: ${BEST_EP}"
  echo "best_CD: ${BEST_CD}"
  echo "best_ckpt: ${BEST_CKPT}"
  echo "late_ep: ${LATE_EP}"
  echo "late_ckpt: ${LATE_CKPT}"
  echo "reference_online: CVM97=72.60, ep89_n1=69.34"
  echo "reference_benchmark: ep97=76.53, ep125=76.18"
  echo ""
} >> "$SUMMARY"

pack_one "epoch${BEST_EP}_best" "$BEST_CKPT"
pack_one "epoch${LATE_EP}_near150" "$LATE_CKPT"

log "All done. Summary: ${SUMMARY}"
cat "$SUMMARY"
