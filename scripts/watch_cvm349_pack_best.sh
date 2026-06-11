#!/usr/bin/env bash
# CVM multires 续训 v2（ep250->349）结束后：打包全局 best，仅 1 次提交
set -euo pipefail

ROOT="/home/cslab/CG"
cd "$ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate jittor

LOG="log/train_cvm_max_multires.log"
WLOG="log/watch_cvm349_pack_best.log"
CKPT_DIR="experiments/cvm_max_multires"
DONE_FLAG="log/cvm_multires_v2_pack.done"
FINAL_EPOCH="${FINAL_EPOCH:-349}"
SCREEN_NAME="${SCREEN_NAME:-cvm_multires_v2}"
POLL="${POLL:-120}"

log() { echo "[$(date '+%F %T %Z')] $*" | tee -a "$WLOG"; }

pick_best_cvm_ckpt() {
  python3 <<'PY'
import glob, os, re
from pathlib import Path

log_path = Path("log/train_cvm_max_multires.log")
best_ep, best_cd = None, float("inf")
if log_path.exists():
    t = log_path.read_text(errors="replace")
    for m in re.finditer(r"\[Epoch\s+(\d+)\][^\n]*val_CD=([\d.]+)", t):
        ep, cd = int(m.group(1)), float(m.group(2))
        if cd < best_cd:
            best_cd, best_ep = cd, ep

candidates = glob.glob("experiments/cvm_max_multires/checkpoint_best_cd*.pkl")
if not candidates:
    raise SystemExit("no best ckpt")

if best_ep is not None:
    for p in sorted(candidates):
        if re.search(rf"epoch{best_ep}\.pkl$", p):
            print(p)
            print(f"epoch={best_ep} val_CD={best_cd:.6f}")
            raise SystemExit(0)

def score(p):
    m = re.search(r"cd([\d.]+)_epoch(\d+)\.pkl", os.path.basename(p))
    if not m:
        return (999.0, -1, p)
    return (float(m.group(1)), -int(m.group(2)), p)

_, _, p = min(candidates, key=lambda x: score(x))
m = re.search(r"epoch(\d+)", os.path.basename(p))
ep = m.group(1) if m else "?"
print(p)
print(f"epoch={ep} val_CD=fallback")
PY
}

wait_for_finish() {
  log "Watch until ep${FINAL_EPOCH} + checkpoint_${FINAL_EPOCH}.pkl (poll=${POLL}s)"
  log "baseline: ep237 online=73.53 | SPCF80k=74.17"
  while true; do
    if grep -q "\[Epoch ${FINAL_EPOCH}\]" "$LOG" 2>/dev/null \
       && [[ -f "${CKPT_DIR}/checkpoint_${FINAL_EPOCH}.pkl" ]]; then
      log "ep${FINAL_EPOCH} done"
      break
    fi
    last_ep=$(python3 - <<'PY' 2>/dev/null || echo "?"
import re
from pathlib import Path
t = Path("log/train_cvm_max_multires.log").read_text(errors="replace")
rows = [int(m.group(1)) for m in re.finditer(r"\[Epoch (\d+)\] train_patch_loss=", t)]
print(rows[-1] if rows else "?")
PY
)
    log "  ... last epoch=${last_ep}; sleep ${POLL}s"
    sleep "$POLL"
  done

  for _ in $(seq 1 36); do
    if ! screen -ls 2>/dev/null | grep -q "\\.${SCREEN_NAME}"; then
      log "${SCREEN_NAME} exited — GPU free"
      return 0
    fi
    sleep 10
  done
  log "WARN: ${SCREEN_NAME} still up; wait 60s then pack"
  sleep 60
}

main() {
  mkdir -p log
  if [[ -f "$DONE_FLAG" ]]; then
    log "Already packed: $(cat "$DONE_FLAG")"
    exit 0
  fi

  wait_for_finish

  mapfile -t picked < <(pick_best_cvm_ckpt)
  ckpt="${picked[0]}"
  meta="${picked[1]:-}"
  if [[ ! -f "$ckpt" ]]; then
    log "ERROR: best ckpt missing: $ckpt"
    exit 1
  fi
  log "Global best: ${ckpt} (${meta})"

  if screen -ls 2>/dev/null | grep -q "\\.${SCREEN_NAME}"; then
    log "Stop ${SCREEN_NAME} for predict"
    screen -S "$SCREEN_NAME" -X quit || true
    sleep 5
  fi

  log "PACK (single submit slot) predict ~42min"
  zip="$(CKPT="$ckpt" SUMMARY="log/pack_cvm_multires_v2_best_summary.txt" \
    bash scripts/pack_cvm_max_multires_ckpt.sh 2>&1 | tee -a "$WLOG" | tail -1)"

  if [[ ! -f "$zip" ]]; then
    log "ERROR: pack failed: ${zip}"
    exit 1
  fi

  log "CVM_V2_PACK_DONE zip=${zip} size=$(ls -lh "$zip" | awk '{print $5}')"
  echo "$(date '+%F %T') ${zip} ckpt=${ckpt} ${meta}" > "$DONE_FLAG"
  log "Submit this zip (only test slot): ${zip}"
}

main "$@"
