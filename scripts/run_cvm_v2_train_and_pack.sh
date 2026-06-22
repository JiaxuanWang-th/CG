#!/usr/bin/env bash
# CVM multires v2：vm_ckpt=299 训练 → 打包(seg+late) → 不自动开 SPCF
set -euo pipefail

ROOT="/home/cslab/CG"
cd "$ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate jittor
export CUDA_VISIBLE_DEVICES=0

VM_CKPT="${VM_CKPT:-experiments/vm_max_multires/checkpoint_299.pkl}"
FINAL_EPOCH=299
PLOG="log/pipeline_cvm_v2.log"
CVM_LOG="log/train_cvm_max_multires_v2.log"
POLL="${POLL:-120}"

log() { echo "[$(date '+%F %T %Z')] $*" | tee -a "$PLOG"; }

wait_last_epoch() {
  local logf="$1" final="$2" ckpt_dir="$3"
  log "wait CVM last epoch ${final} (range ends at epochs-1)"
  while true; do
    if grep -q "\[Epoch ${final}\] train_patch_loss=" "$logf" 2>/dev/null \
       && [[ -f "${ckpt_dir}/checkpoint_${final}.pkl" ]]; then
      log "CVM epoch ${final} done"
      return 0
    fi
    local last
    last=$(python3 - <<PY 2>/dev/null || echo "?"
import re
from pathlib import Path
t = Path("$logf").read_text(errors="replace")
rows = [int(m.group(1)) for m in re.finditer(r"\[Epoch (\d+)\] train_patch_loss=", t)]
print(rows[-1] if rows else "?")
PY
)
    log "  ... last_epoch=${last}; sleep ${POLL}s"
    sleep "$POLL"
  done
}

pack_cvm() {
  local listf="log/pack_list_cvm_v2.txt"
  python3 scripts/pick_multires_stage_ckpts.py \
    --mode epoch --log "$CVM_LOG" --ckpt-dir experiments/cvm_max_multires_v2 \
    --seg-lo 200 --seg-hi "$FINAL_EPOCH" --late-lo 280 --late-hi "$FINAL_EPOCH" \
    > "$listf" 2>&1
  cat "$listf" >> "$PLOG"
  grep '^PACK' "$listf" | while read -r _ label ckpt; do
    log "PACK cvm ${label} ${ckpt}"
    zip=$(CKPT="$ckpt" PACK_LABEL="${label}" SUMMARY="log/pack_cvm_v2_${label}_summary.txt" \
      bash scripts/pack_cvm_max_multires_ckpt.sh 2>&1 | tee -a "$PLOG" | tail -1)
    log "DONE cvm ${label} zip=${zip}"
  done
  log "=== CVM PACK DONE — submit zips for online test before SPCF ==="
}

# -------- start --------
log "=== CVM v2 start vm_ckpt=${VM_CKPT} ==="
log "NO auto SPCF — test CVM zips online first"

python3 <<PY
from pathlib import Path
p = Path("configs/task/train_cvm_max_multires_v2.yaml")
lines = p.read_text().splitlines()
out = []
for line in lines:
    if line.startswith("vm_ckpt:"):
        out.append("vm_ckpt: ${VM_CKPT}")
    else:
        out.append(line)
p.write_text("\n".join(out) + "\n")
PY

screen -S cvm_multires_v2 -X quit 2>/dev/null || true
sleep 2
rm -rf experiments/cvm_max_multires_v2
: > "$CVM_LOG"

screen -dmS cvm_multires_v2 bash -lc "
  source \"\$HOME/miniconda3/etc/profile.d/conda.sh\"
  conda activate jittor
  cd \"$ROOT\"
  export CUDA_VISIBLE_DEVICES=0
  echo \"[\$(date '+%F %T')] CVM v2 0->299 vm=${VM_CKPT} (online VM299=74.05)\" >> \"$CVM_LOG\"
  python run.py --task configs/task/train_cvm_max_multires_v2.yaml --seed 123 2>&1 | tee -a \"$CVM_LOG\"
"
log "LAUNCH screen cvm_multires_v2"

wait_last_epoch "$CVM_LOG" "$FINAL_EPOCH" "experiments/cvm_max_multires_v2"
screen -S cvm_multires_v2 -X quit 2>/dev/null || true
sleep 3
pack_cvm
log "=== ALL DONE ==="
