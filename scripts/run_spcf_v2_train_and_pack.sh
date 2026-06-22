#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/cslab/CG"
cd "$ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate jittor
export CUDA_VISIBLE_DEVICES=0

CVM_CKPT="experiments/cvm_max_multires_v2/checkpoint_best_cd0.0001_epoch225.pkl"
FINAL_SPCF=150000
SPCF_LOG="log/train_straightpcf_max_multires_v2.log"
SPCF_DIR="experiments/straightpcf_max_multires_v2"
PLOG="log/pipeline_spcf_v2.log"
POLL="${POLL:-120}"

log() { echo "[$(date '+%F %T %Z')] $*" | tee -a "$PLOG"; }

wait_iter() {
  log "wait iter ${FINAL_SPCF} in ${SPCF_LOG}"
  while true; do
    if grep -q "\[Val\] Iter ${FINAL_SPCF}" "$SPCF_LOG" 2>/dev/null \
       && [[ -f "${SPCF_DIR}/checkpoint_iter${FINAL_SPCF}.pkl" ]]; then
      log "iter ${FINAL_SPCF} done"
      return 0
    fi
    local last
    last=$(FINAL="$FINAL_SPCF" LOGF="$SPCF_LOG" python3 - <<'PY' 2>/dev/null || echo "?"
import os, re
from pathlib import Path
final = int(os.environ["FINAL"])
t = Path(os.environ["LOGF"]).read_text(errors="replace")
rows = [int(m.group(1)) for m in re.finditer(r"\[Val\] Iter\s+(\d+)", t)]
print(rows[-1] if rows else "?")
PY
)
    log "  ... last_val_iter=${last}; sleep ${POLL}s"
    sleep "$POLL"
  done
}

pack_spcf() {
  local listf="log/pack_list_spcf_v2.txt"
  python3 scripts/pick_multires_stage_ckpts.py \
    --mode iter --log "$SPCF_LOG" --ckpt-dir "$SPCF_DIR" \
    --seg-lo 100000 --seg-hi "$FINAL_SPCF" --late-lo 140000 --late-hi "$FINAL_SPCF" \
    > "$listf" 2>&1
  cat "$listf" >> "$PLOG"
  grep '^PACK' "$listf" | while read -r _ label ckpt; do
    log "PACK spcf ${label} ${ckpt}"
    zip=$(CKPT="$ckpt" CKPT_DIR="$SPCF_DIR" PACK_LABEL="${label}" \
      SUMMARY="log/pack_spcf_v2_${label}_summary.txt" \
      bash scripts/pack_spcf_max_multires_ckpt.sh 2>&1 | tee -a "$PLOG" | tail -1)
    log "DONE spcf ${label} zip=${zip}"
  done
  log "=== SPCF PACK DONE ==="
}

log "=== SPCF v2 0->150k cvm=${CVM_CKPT} ==="

python3 <<PY
from pathlib import Path
p = Path("configs/task/train_straightpcf_max_multires_v2.yaml")
lines = p.read_text().splitlines()
out = []
for line in lines:
    if line.startswith("cvm_ckpt:"):
        out.append("cvm_ckpt: ${CVM_CKPT}")
    else:
        out.append(line)
p.write_text("\n".join(out) + "\n")
PY

: > "$SPCF_LOG"
screen -S spcf_multires_v2 -X quit 2>/dev/null || true
sleep 2

screen -dmS spcf_multires_v2 bash -lc "
  source \"\$HOME/miniconda3/etc/profile.d/conda.sh\"
  conda activate jittor
  cd \"$ROOT\"
  export CUDA_VISIBLE_DEVICES=0
  echo \"[\$(date '+%F %T')] SPCF v2 0->150k cvm=${CVM_CKPT}\" >> \"$SPCF_LOG\"
  python run.py --task configs/task/train_straightpcf_max_multires_v2.yaml --seed 123 \
    2>&1 | tee -a \"$SPCF_LOG\"
"
log "LAUNCH screen spcf_multires_v2"

wait_iter
screen -S spcf_multires_v2 -X quit 2>/dev/null || true
sleep 3
pack_spcf
log "=== ALL DONE ==="
