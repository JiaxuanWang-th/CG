#!/usr/bin/env bash
# SPCF max 续训 90k→250k 结束后：在 200k / 250k 各打一个 test zip
# 每个里程碑：优先用该点附近 best_* ckpt；否则用 periodic ckpt 中 val_CD 最优者
#
# 用法：
#   bash scripts/pack_spcf_max_resume_200_250.sh
#   WAIT=0 bash scripts/pack_spcf_max_resume_200_250.sh
set -euo pipefail

ROOT="/home/cslab/CG"
cd "$ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate jittor
export CUDA_VISIBLE_DEVICES=0

WAIT="${WAIT:-1}"
RESUME_START="${RESUME_START:-90000}"
MAX_ITERS="${MAX_ITERS:-250000}"
LOG="log/train_straightpcf_max_resume.log"
CKPT_DIR="experiments/straightpcf_max"
SUMMARY="log/pack_spcf_max_resume_200_250_summary.txt"
# 默认只打 250k；需要 200k 时：MILESTONES="200000 250000" bash ...
if [[ -n "${MILESTONES:-}" ]]; then
  read -ra MILESTONES <<< "${MILESTONES}"
else
  MILESTONES=(250000)
fi

log() { echo "[$(date '+%F %T')] $*"; }

wait_for_training() {
  log "Waiting for resume training to finish (iter ${MAX_ITERS} + checkpoint_iter${MAX_ITERS}.pkl)..."
  while true; do
    if [[ -f "${CKPT_DIR}/checkpoint_iter${MAX_ITERS}.pkl" ]] \
      && grep -q "\[Val\] Iter ${MAX_ITERS}" "$LOG" 2>/dev/null; then
      if ! screen -ls 2>/dev/null | grep -q "train_spcf_max_resume"; then
        log "Training screen gone and iter${MAX_ITERS} val logged."
        break
      fi
      sleep 30
      if grep -q "\[Val\] Iter ${MAX_ITERS}" "$LOG" \
        && [[ -f "${CKPT_DIR}/checkpoint_iter${MAX_ITERS}.pkl" ]]; then
        log "iter${MAX_ITERS} checkpoint present; proceed."
        break
      fi
    fi
    last_iter=$(python3 - <<'PY' 2>/dev/null || echo "?"
import re
from pathlib import Path
p = Path("log/train_straightpcf_max_resume.log")
if not p.is_file():
    print("?")
    raise SystemExit
text = p.read_text(errors="replace")
rows = [int(m.group(1)) for m in re.finditer(r"\[Val\] Iter (\d+)", text)]
print(rows[-1] if rows else "?")
PY
)
    log "  ... last val iter in log: ${last_iter}; sleeping 120s"
    sleep 120
  done
  sleep 10
}

pick_ckpt_for_milestone() {
  local target="$1"
  python3 - <<PY
import re
from pathlib import Path

target = int("${target}")
resume_start = int("${RESUME_START}")
log_path = Path("${LOG}")
ckpt_dir = Path("${CKPT_DIR}")

text = log_path.read_text(errors="replace") if log_path.is_file() else ""
vals = {}
for m in re.finditer(r"\[Val\] Iter (\d+) \| Chamfer ([\d.]+)", text):
    it, cd = int(m.group(1)), float(m.group(2))
    if resume_start <= it <= target:
        vals[it] = cd

near = [it for it in vals if abs(it - target) <= 5000]
best_near = sorted(
    ckpt_dir.glob(f"checkpoint_best_cd*_iter*.pkl"),
    key=lambda p: p.stat().st_mtime,
)
for p in best_near:
    m = re.search(r"_iter(\d+)\.pkl$", p.name)
    if not m:
        continue
    it = int(m.group(1))
    if resume_start <= it <= target and abs(it - target) <= 5000:
        print(f"{it}\t{vals.get(it, float('nan')):.6f}\t{p}")
        raise SystemExit(0)

candidates = [it for it in vals if it in near] or [it for it in vals if it <= target]
if not candidates:
    periodic = ckpt_dir / f"checkpoint_iter{target}.pkl"
    if periodic.is_file():
        print(f"{target}\tnan\t{periodic}")
        raise SystemExit(0)
    raise SystemExit(f"No val rows <= {target} in {log_path}")

best_it = min(candidates, key=lambda it: (vals[it], -it))
best_cd = vals[best_it]
periodic = ckpt_dir / f"checkpoint_iter{best_it}.pkl"
best_file = sorted(ckpt_dir.glob(f"checkpoint_best_cd*_iter{best_it}.pkl"))
ckpt = best_file[0] if best_file else periodic
if not ckpt.is_file():
    raise SystemExit(f"Missing ckpt for iter {best_it}: {ckpt}")
print(f"{best_it}\t{best_cd:.6f}\t{ckpt}")
PY
}

pack_one() {
  local milestone="$1"
  local ckpt="$2"
  local pick_iter="$3"
  local pick_cd="$4"
  local tag="iter${milestone}_pick${pick_iter}"
  local out_dir="results_spcf_max_test_${tag}"
  local zip_name="result_spcf_max_${tag}.zip"
  local task_yaml="configs/task/predict_spcf_max_test_${tag}.yaml"
  local pred_log="log/predict_spcf_max_test_${tag}.log"

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
  model: straightpcf_max

writer:
  __target__: vm
  save_dir: ${out_dir}
  save_name: denoised
EOF

  log "PREDICT test: milestone=${milestone} pick_iter=${pick_iter} val_CD=${pick_cd} ckpt=${ckpt}"
  python run.py --task "$task_yaml" --seed 123 2>&1 | tee "$pred_log"

  log "PACK -> ${zip_name}"
  rm -f "$zip_name"
  (cd "$out_dir" && zip -r "../${zip_name}" shapenet/)

  local n
  n=$(find "$out_dir" -name 'denoised.npy' | wc -l)
  log "DONE ${tag}: ${zip_name} (${n} files)"
  echo -e "${tag}\t${pick_iter}\t${pick_cd}\t${ckpt}\t${zip_name}\t${n}" >> "$SUMMARY"
}

mkdir -p log
: > "$SUMMARY"
echo -e "tag\tpick_iter\tval_CD\tckpt\tzip\tnum_files" >> "$SUMMARY"

if [[ "$WAIT" == "1" ]]; then
  wait_for_training
else
  log "WAIT=0: skip wait, pack now"
fi

{
  echo "pack_time: $(date '+%F %T')"
  echo "resume_start: ${RESUME_START}"
  echo "max_iters: ${MAX_ITERS}"
  echo "reference_online: SPCF75k=73.53, CVM125=72.72"
  echo ""
} >> "$SUMMARY"

for ms in "${MILESTONES[@]}"; do
  mapfile -t _pick < <(pick_ckpt_for_milestone "$ms")
  PICK_IT="${_pick[0]:-}"
  PICK_CD="${_pick[1]:-}"
  PICK_CKPT="${_pick[2]:-}"
  if [[ -z "$PICK_CKPT" ]]; then
    log "ERROR: failed to pick ckpt for milestone ${ms}"
    exit 1
  fi
  log "Milestone ${ms}: pick iter${PICK_IT} val_CD=${PICK_CD} -> ${PICK_CKPT}"
  pack_one "$ms" "$PICK_CKPT" "$PICK_IT" "$PICK_CD"
done

log "All done. Summary: ${SUMMARY}"
cat "$SUMMARY"
