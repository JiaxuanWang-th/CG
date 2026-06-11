#!/usr/bin/env bash
# 监督 SPCF max multires：北京时间 PACK_HOUR:PACK_MINUTE 自动打包 1 次（今日提交），打包后续训
set -euo pipefail

ROOT="/home/cslab/CG"
cd "$ROOT"
export TZ=Asia/Shanghai

PACK_HOUR="${PACK_HOUR:-23}"
PACK_MINUTE="${PACK_MINUTE:-0}"
POLL="${POLL:-60}"
TRAIN_TASK="configs/task/train_straightpcf_max_multires.yaml"
TRAIN_LOG="log/train_straightpcf_max_multires.log"
CKPT_DIR="experiments/straightpcf_max_multires"
WLOG="log/watch_spcf_multires_pack_23h.log"
DONE_FLAG="log/spcf_multires_pack_23h.done"

log() { echo "[$(date '+%F %T %Z')] $*" | tee -a "$WLOG"; }

ts_target_today() {
  date -d "today ${PACK_HOUR}:$(printf '%02d' "${PACK_MINUTE}"):00" +%s
}

wait_until_pack_time() {
  local target now last_ep
  target="$(ts_target_today)"
  now="$(date +%s)"
  if (( now >= target )); then
    log "Already past ${PACK_HOUR}:$(printf '%02d' "${PACK_MINUTE}") — pack now"
    return 0
  fi
  log "Wait until ${PACK_HOUR}:$(printf '%02d' "${PACK_MINUTE}") CST | target_ts=${target}"
  while (( $(date +%s) < target )); do
    last_ep="$(current_iter)"
    log "  ... now=$(date '+%H:%M') iter≈${last_ep}; sleep ${POLL}s"
    sleep "$POLL"
  done
  log "Pack time reached: $(date '+%F %T %Z')"
}

current_iter() {
  python3 - <<'PY' 2>/dev/null || echo "?"
import re
from pathlib import Path
p = Path("log/train_straightpcf_max_multires.log")
if not p.exists():
    print("?")
    raise SystemExit
t = p.read_text(errors="replace")
iters = [int(m.group(1)) for m in re.finditer(r"(\d+)/90000", t)]
print(iters[-1] if iters else "?")
PY
}

wait_for_fresh_ckpt() {
  # 23:00 后最多等 20min，直到出现 iter>=40000 的 periodic ckpt 或 val 刚跑完
  local i iter
  log "Wait for saved ckpt (iter>=40000, up to 20min)..."
  for i in $(seq 1 40); do
    iter="$(current_iter)"
    if [[ "$iter" != "?" && "$iter" -ge 40000 ]]; then
      local latest
      latest="$(ls -1 "${CKPT_DIR}"/checkpoint_iter*.pkl 2>/dev/null | sort -V | tail -1 || true)"
      if [[ -n "$latest" ]]; then
        log "Latest periodic ckpt: $(basename "$latest") (train iter≈${iter})"
        return 0
      fi
    fi
    sleep 30
  done
  log "WARN: no iter>=40000 ckpt yet; will pack best available"
}

stop_training() {
  if screen -ls 2>/dev/null | grep -q '\.spcf_multires'; then
    log "Stop spcf_multires screen for predict GPU"
    screen -S spcf_multires -X quit || true
    sleep 5
  fi
}

pick_pack_ckpt() {
  python3 <<'PY'
import glob, os, re
from pathlib import Path

log_path = Path("log/train_straightpcf_max_multires.log")
best_iter, best_cd = None, float("inf")
if log_path.exists():
    for m in re.finditer(r"\[Val\] Iter\s+(\d+)\s+\|\s+Chamfer\s+([\d.]+)", log_path.read_text(errors="replace")):
        it, cd = int(m.group(1)), float(m.group(2))
        if cd < best_cd:
            best_cd, best_iter = cd, it

ckpt_dir = Path("experiments/straightpcf_max_multires")
candidates = []

if best_iter is not None:
    for pat in (
        f"checkpoint_best_cd{best_cd:.4f}_iter{best_iter}.pkl",
        f"checkpoint_best_cd0.0001_iter{best_iter}.pkl",
        f"checkpoint_iter{best_iter}.pkl",
    ):
        p = ckpt_dir / pat
        if p.exists():
            candidates.append((best_cd, best_iter, str(p)))
            break

# fallback: lowest-cd best file on disk
for p in glob.glob(str(ckpt_dir / "checkpoint_best_cd*_iter*.pkl")):
    m = re.search(r"cd([\d.]+)_iter(\d+)\.pkl", os.path.basename(p))
    if m:
        candidates.append((float(m.group(1)), int(m.group(2)), p))

# fallback: latest periodic
periodics = sorted(glob.glob(str(ckpt_dir / "checkpoint_iter*.pkl")),
                   key=lambda x: int(re.search(r"iter(\d+)", x).group(1)))
if periodics:
    p = periodics[-1]
    it = int(re.search(r"iter(\d+)", os.path.basename(p)).group(1))
    candidates.append((999.0, it, p))

if not candidates:
    raise SystemExit("")

candidates.sort(key=lambda x: (x[0], -x[1]))
pack_iter = candidates[0][1]
pack_ckpt = candidates[0][2]

# resume = latest periodic iter
resume_ckpt = periodics[-1] if periodics else pack_ckpt
resume_iter = int(re.search(r"iter(\d+)", os.path.basename(resume_ckpt)).group(1))

# best chamfer for yaml resume
resume_best = best_cd if best_cd < float("inf") else None

print(pack_ckpt)
print(pack_iter)
print(resume_ckpt)
print(resume_iter)
print("" if resume_best is None else f"{resume_best:.6f}")
PY
}

resume_training() {
  local resume_ckpt="$1" resume_iter="$2" resume_best="${3:-}"
  python3 <<PY
from pathlib import Path
p = Path("$TRAIN_TASK")
lines = p.read_text().splitlines()
out, in_trainer = [], False
has_load = any(l.startswith("load_ckpt:") for l in lines)
for line in lines:
    if line.startswith("load_ckpt:"):
        out.append("load_ckpt: ${resume_ckpt}")
        has_load = True
    elif line.strip().startswith("start_iter:"):
        out.append("  start_iter: ${resume_iter}")
        in_trainer = True
    elif line.startswith("  resume_best_chamfer:"):
        if "${resume_best}":
            out.append("  resume_best_chamfer: ${resume_best}")
        else:
            out.append(line)
    else:
        out.append(line)
if not has_load:
    # insert after cvm_ckpt line
    tmp = []
    for line in out:
        tmp.append(line)
        if line.startswith("cvm_ckpt:"):
            tmp.append("load_ckpt: ${resume_ckpt}")
    out = tmp
p.write_text("\n".join(out) + "\n")
PY

  if screen -ls 2>/dev/null | grep -q '\.spcf_multires'; then
    log "spcf_multires already running; skip relaunch"
    return 0
  fi
  log "Resume SPCF from iter ${resume_iter} | ${resume_ckpt}"
  screen -dmS spcf_multires bash -lc "
    source \"\$HOME/miniconda3/etc/profile.d/conda.sh\"
    conda activate jittor
    cd \"$ROOT\"
    export CUDA_VISIBLE_DEVICES=0
    echo \"[\$(date '+%F %T')] SPCF multires resume iter${resume_iter}\" >> \"$TRAIN_LOG\"
    python run.py --task $TRAIN_TASK --seed 123 2>&1 | tee -a \"$TRAIN_LOG\"
  "
  sleep 3
  screen -ls 2>/dev/null | grep -q '\.spcf_multires' && log "SPCF training resumed" || log "WARN: resume screen failed"
}

main() {
  mkdir -p log
  if [[ -f "$DONE_FLAG" ]]; then
    log "Already packed ($(cat "$DONE_FLAG")); exit"
    exit 0
  fi

  log "=== SPCF multires pack watcher | deadline ${PACK_HOUR}:$(printf '%02d' "${PACK_MINUTE}") CST ==="
  log "reference: CVM237 online=73.53 | 32768 anchor=73.59"

  wait_until_pack_time
  wait_for_fresh_ckpt

  mapfile -t picked < <(pick_pack_ckpt)
  pack_ckpt="${picked[0]}"
  pack_iter="${picked[1]}"
  resume_ckpt="${picked[2]}"
  resume_iter="${picked[3]}"
  resume_best="${picked[4]:-}"

  log "Pack ckpt (val best / fallback): ${pack_ckpt} (iter ${pack_iter})"
  log "Resume after pack: ${resume_ckpt} (iter ${resume_iter})"

  stop_training

  zip=""
  zip="$(CKPT="$pack_ckpt" SUMMARY="log/pack_spcf_multires_23h_summary.txt" \
    bash scripts/pack_spcf_max_multires_ckpt.sh 2>&1 | tee -a "$WLOG" | tail -1)"

  if [[ ! -f "$zip" ]]; then
    log "ERROR: pack failed, zip missing: ${zip}"
    resume_training "$resume_ckpt" "$resume_iter" "$resume_best"
    exit 1
  fi

  log "SPCF_PACK_DONE zip=${zip} size=$(ls -lh "$zip" | awk '{print $5}')"
  echo "$(date '+%F %T') ${zip} pack_iter=${pack_iter}" > "$DONE_FLAG"

  resume_training "$resume_ckpt" "$resume_iter" "$resume_best"
  log "Pipeline complete: submit ${zip} tonight"
}

main "$@"
