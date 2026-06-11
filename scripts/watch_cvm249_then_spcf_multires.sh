#!/usr/bin/env bash
# ep249 完成 -> 打包 global best CVM -> 再开 SPCF max multires 90k
set -euo pipefail

ROOT="/home/cslab/CG"
cd "$ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate jittor

LOG="log/train_cvm_max_multires.log"
WLOG="log/watch_cvm249_then_spcf_multires.log"
CKPT_DIR="experiments/cvm_max_multires"
TASK="configs/task/train_straightpcf_max_multires.yaml"
POLL="${POLL:-120}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$WLOG"; }

pick_best_cvm_ckpt() {
  python3 <<'PY'
import glob, os, re
from pathlib import Path

# 1) 优先从 log 找全局最低 val_CD 对应 epoch（跨续训段）
log_path = Path("log/train_cvm_max_multires.log")
best_ep, best_cd = None, float("inf")
if log_path.exists():
    t = log_path.read_text(errors="replace")
    for m in re.finditer(
        r"\[Epoch\s+(\d+)\][^\n]*val_CD=([\d.]+)", t
    ):
        ep, cd = int(m.group(1)), float(m.group(2))
        if cd < best_cd:
            best_cd, best_ep = cd, ep

candidates = glob.glob("experiments/cvm_max_multires/checkpoint_best_cd*.pkl")
if not candidates:
    raise SystemExit("")

def score(p):
    m = re.search(r"cd([\d.]+)_epoch(\d+)\.pkl", os.path.basename(p))
    if not m:
        return (999.0, -1, p)
    return (float(m.group(1)), -int(m.group(2)), p)

# 2) 若 log 有 best epoch，优先匹配该 epoch 的 best ckpt
if best_ep is not None:
    for p in candidates:
        if re.search(rf"epoch{best_ep}\.pkl$", p):
            print(p)
            raise SystemExit(0)

# 3) 否则按文件名 cd 标签选最低
_, _, p = min(candidates, key=lambda x: score(x))
print(p)
PY
}

wait_for_cvm249() {
  log "Watching CVM multires until ep249 (poll=${POLL}s)..."
  while true; do
    if grep -q '\[Epoch 249\]' "$LOG" 2>/dev/null && [[ -f "${CKPT_DIR}/checkpoint_249.pkl" ]]; then
      log "ep249 complete + checkpoint_249.pkl present"
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
    log "  ... last epoch in log: ${last_ep}; sleep ${POLL}s"
    sleep "$POLL"
  done

  for _ in $(seq 1 36); do
    if ! screen -ls 2>/dev/null | grep -q '\.cvm_multires'; then
      log "cvm_multires screen exited — GPU free for pack"
      return 0
    fi
    sleep 10
  done
  log "WARN: cvm_multires screen still up; waiting 60s then proceed"
  sleep 60
}

pack_best_cvm() {
  local ckpt="$1"
  log "PACK best CVM for online test (SPCF base ckpt) | ${ckpt}"
  log "  predict ~42min + zip"
  local zip
  zip="$(CKPT="$ckpt" bash scripts/pack_cvm_max_multires_ckpt.sh 2>&1 | tee -a "$WLOG" | tail -1)"
  if [[ ! -f "$zip" ]]; then
    log "ERROR: pack failed, zip missing: $zip"
    exit 1
  fi
  log "CVM_BEST_PACKED zip=${zip} size=$(ls -lh "$zip" | awk '{print $5}')"
}

set_cvm_ckpt_in_yaml() {
  local ckpt="$1"
  python3 <<PY
from pathlib import Path
p = Path("$TASK")
lines = p.read_text().splitlines()
out = []
for line in lines:
    if line.startswith("cvm_ckpt:"):
        out.append(f"cvm_ckpt: {ckpt}")
    else:
        out.append(line)
p.write_text("\n".join(out) + "\n")
PY
}

launch_spcf() {
  local ckpt="$1"
  if screen -ls 2>/dev/null | grep -q '\.spcf_multires'; then
    log "spcf_multires screen already running; skip"
    return 0
  fi
  log "Launch SPCF max multires | cvm_ckpt=${ckpt} | 0->90000 iter"
  screen -dmS spcf_multires bash -lc "
    source \"\$HOME/miniconda3/etc/profile.d/conda.sh\"
    conda activate jittor
    cd \"$ROOT\"
    export CUDA_VISIBLE_DEVICES=0
    echo \"[\$(date '+%F %T')] SPCF multires start | cvm_ckpt=${ckpt}\" >> log/train_straightpcf_max_multires.log
    python run.py --task $TASK --seed 123 2>&1 | tee -a log/train_straightpcf_max_multires.log
  "
  sleep 3
  if screen -ls 2>/dev/null | grep -q '\.spcf_multires'; then
    log "SPCF_MULTIRES_STARTED screen=spcf_multires"
  else
    log "ERROR: failed to start spcf_multires screen"
    exit 1
  fi
}

# 若已有进度则 append，否则清空（支持重启 watcher）
if [[ ! -f "$WLOG" ]]; then
  : > "$WLOG"
fi
log "=== pipeline: ep249 -> pack best CVM -> SPCF multires ==="

wait_for_cvm249
BEST="$(pick_best_cvm_ckpt)"
if [[ -z "$BEST" || ! -f "$BEST" ]]; then
  log "ERROR: no best CVM ckpt found"
  exit 1
fi
log "Selected best CVM ckpt (SPCF base): $BEST"

pack_best_cvm "$BEST"
set_cvm_ckpt_in_yaml "$BEST"
launch_spcf "$BEST"
log "CVM249_DONE pipeline complete: packed + SPCF started"
