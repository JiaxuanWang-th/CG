#!/usr/bin/env bash
# 全流程 v2：VM 100->300 -> 打包(200-300 best + near300) -> CVM 0->300 -> 打包 -> SPCF 0->150k -> 打包
set -euo pipefail

ROOT="/home/cslab/CG"
cd "$ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate jittor
export CUDA_VISIBLE_DEVICES=0

PLOG="log/pipeline_multires_300_150k.log"
POLL="${POLL:-120}"
# trainer uses range(start, epochs) -> last trained epoch is epochs-1
FINAL_VM=299
FINAL_CVM=299
FINAL_SPCF=150000
PACK_VM_HI=299
PACK_CVM_HI=299

log() { echo "[$(date '+%F %T %Z')] $*" | tee -a "$PLOG"; }

stop_screen() {
  local name="$1"
  if screen -ls 2>/dev/null | grep -q "\\.${name}"; then
    screen -S "$name" -X quit || true
    sleep 5
  fi
}

wait_epoch() {
  local logf="$1" final="$2" ckpt_dir="$3"
  log "wait last epoch ${final} in ${logf} (range ends at epochs-1)"
  while true; do
    if grep -q "\[Epoch ${final}\] train_patch_loss=" "$logf" 2>/dev/null \
       && [[ -f "${ckpt_dir}/checkpoint_${final}.pkl" || -f "${ckpt_dir}/checkpoint_best_cd"*"_epoch${final}.pkl" ]]; then
      log "epoch ${final} done"
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

wait_iter() {
  local logf="$1" final="$2" ckpt_dir="$3"
  log "wait iter ${final} in ${logf}"
  while true; do
    if grep -q "\[Val\] Iter ${final}" "$logf" 2>/dev/null && [[ -f "${ckpt_dir}/checkpoint_iter${final}.pkl" ]]; then
      log "iter ${final} done"
      return 0
    fi
    local last
    last=$(FINAL="$final" LOGF="$logf" python3 - <<'PY' 2>/dev/null || echo "?"
import os, re
from pathlib import Path
final = int(os.environ["FINAL"])
t = Path(os.environ["LOGF"]).read_text(errors="replace")
rows = [int(m.group(1)) for m in re.finditer(r"(\d+)/" + str(final), t)]
print(rows[-1] if rows else "?")
PY
)
    log "  ... last_iter~=${last}; sleep ${POLL}s"
    sleep "$POLL"
  done
}

pack_epoch_stage() {
  local stage="$1" logf="$2" ckpt_dir="$3" pack_fn="$4" final="$5"
  local listf="log/pack_list_${stage}.txt"
  python3 scripts/pick_multires_stage_ckpts.py \
    --mode epoch --log "$logf" --ckpt-dir "$ckpt_dir" \
    --seg-lo 200 --seg-hi "$final" --late-lo 280 --late-hi "$final" \
    > "$listf" 2>&1 || true
  grep '^PACK' "$listf" | while read -r _ label ckpt; do
    log "PACK ${stage} ${label} ${ckpt}"
    zip=$(CKPT="$ckpt" PACK_LABEL="${stage}_${label}" SUMMARY="log/pack_${stage}_${label}_summary.txt" bash "$pack_fn" 2>&1 | tee -a "$PLOG" | tail -1)
    log "DONE ${stage} ${label} zip=${zip}"
  done
}

pack_stage_spcf() {
  local logf="$1" ckpt_dir="$2"
  local listf="log/pack_list_spcf.txt"
  python3 scripts/pick_multires_stage_ckpts.py \
    --mode iter --log "$logf" --ckpt-dir "$ckpt_dir" \
    --seg-lo 100000 --seg-hi "$FINAL_SPCF" --late-lo 140000 --late-hi "$FINAL_SPCF" \
    > "$listf" 2>&1 || true
  grep '^PACK' "$listf" | while read -r _ label ckpt; do
    log "PACK spcf ${label} ${ckpt}"
    zip=$(CKPT="$ckpt" SUMMARY="log/pack_spcf_v2_${label}_summary.txt" bash scripts/pack_spcf_max_multires_ckpt.sh 2>&1 | tee -a "$PLOG" | tail -1)
    log "DONE spcf ${label} zip=${zip}"
  done
}

global_best_epoch_ckpt() {
  python3 - <<PY
import glob, os, re
from pathlib import Path
log = Path("$1").read_text(errors="replace")
best = None
for m in re.finditer(r"\[Epoch\s+(\d+)\][^\n]*val_CD=([\d.]+)", log):
    ep, cd = int(m.group(1)), float(m.group(2))
    if best is None or cd < best[0]:
        best = (cd, ep)
if not best:
    raise SystemExit(1)
_, ep = best
cdir = Path("$2")
for pat in (f"checkpoint_best_cd*_epoch{ep}.pkl", f"checkpoint_{ep}.pkl"):
    hits = sorted(glob.glob(str(cdir / pat)))
    if hits:
        print(hits[0])
        raise SystemExit(0)
raise SystemExit(1)
PY
}

set_yaml_vm_ckpt() {
  local ckpt="$1"
  python3 <<PY
from pathlib import Path
p = Path("configs/task/train_cvm_max_multires_v2.yaml")
lines = p.read_text().splitlines()
out = []
for line in lines:
    if line.startswith("vm_ckpt:"):
        out.append("vm_ckpt: ${ckpt}")
    else:
        out.append(line)
p.write_text("\n".join(out) + "\n")
PY
}

set_yaml_cvm_ckpt() {
  local ckpt="$1"
  python3 <<PY
from pathlib import Path
p = Path("configs/task/train_straightpcf_max_multires_v2.yaml")
lines = p.read_text().splitlines()
out = []
for line in lines:
    if line.startswith("cvm_ckpt:"):
        out.append("cvm_ckpt: ${ckpt}")
    else:
        out.append(line)
p.write_text("\n".join(out) + "\n")
PY
}

launch_train() {
  local screen_name="$1" task="$2" logf="$3" tag="$4"
  stop_screen "$screen_name"
  log "LAUNCH ${screen_name} task=${task}"
  screen -dmS "$screen_name" bash -lc "
    source \"\$HOME/miniconda3/etc/profile.d/conda.sh\"
    conda activate jittor
    cd \"$ROOT\"
    export CUDA_VISIBLE_DEVICES=0
    echo \"[\$(date '+%F %T')] ${tag}\" >> \"$logf\"
    python run.py --task $task --seed 123 2>&1 | tee -a \"$logf\"
  "
  sleep 3
}

log "=== pipeline resume VM299 CVM299 SPCF150k ==="
log "pack rule: seg_best 200-end + late/near_end (280-end)"

VM_LOG="log/train_vm_max_multires.log"
CVM_LOG="log/train_cvm_max_multires_v2.log"
SPCF_LOG="log/train_straightpcf_max_multires_v2.log"

# -------- Phase 1 VM --------
if ! grep -q "\[Epoch ${FINAL_VM}\] train_patch_loss=" "$VM_LOG" 2>/dev/null; then
  launch_train "vm_multires_v2" "configs/task/train_vm_max_multires.yaml" "$VM_LOG" "VM resume ep100->299"
  wait_epoch "$VM_LOG" "$FINAL_VM" "experiments/vm_max_multires"
fi
stop_screen "vm_multires_v2"
pack_epoch_stage "vm" "$VM_LOG" "experiments/vm_max_multires" "scripts/pack_vm_max_multires_ckpt.sh" "$PACK_VM_HI"

VM_BEST="$(global_best_epoch_ckpt "$VM_LOG" "experiments/vm_max_multires")"
log "VM global best ckpt (for CVM after online test): ${VM_BEST}"
log "SKIP auto CVM — submit VM zips first, then set vm_ckpt in train_cvm_max_multires_v2.yaml"

# -------- Phase 2 CVM (manual gate: set vm_ckpt after VM online test) --------
if [[ "${AUTO_START_CVM:-0}" == "1" ]]; then
  set_yaml_vm_ckpt "$VM_BEST"
  if ! grep -q "\[Epoch ${FINAL_CVM}\] train_patch_loss=" "$CVM_LOG" 2>/dev/null; then
    launch_train "cvm_multires_v2" "configs/task/train_cvm_max_multires_v2.yaml" "$CVM_LOG" "CVM v2 0->299 vm=${VM_BEST}"
    wait_epoch "$CVM_LOG" "$FINAL_CVM" "experiments/cvm_max_multires_v2"
  fi
  stop_screen "cvm_multires_v2"
  pack_epoch_stage "cvm" "$CVM_LOG" "experiments/cvm_max_multires_v2" "scripts/pack_cvm_max_multires_ckpt.sh" "$PACK_CVM_HI"

  CVM_BEST="$(global_best_epoch_ckpt "$CVM_LOG" "experiments/cvm_max_multires_v2")"
  log "CVM global best for SPCF train: ${CVM_BEST}"
  set_yaml_cvm_ckpt "$CVM_BEST"

  # -------- Phase 3 SPCF --------
  if ! grep -q "\[Val\] Iter ${FINAL_SPCF}" "$SPCF_LOG" 2>/dev/null; then
    launch_train "spcf_multires_v2" "configs/task/train_straightpcf_max_multires_v2.yaml" "$SPCF_LOG" "SPCF v2 0->150k cvm=${CVM_BEST}"
    wait_iter "$SPCF_LOG" "$FINAL_SPCF" "experiments/straightpcf_max_multires_v2"
  fi
  stop_screen "spcf_multires_v2"
  pack_stage_spcf "$SPCF_LOG" "experiments/straightpcf_max_multires_v2"
  log "=== PIPELINE_DONE ==="
else
  log "=== STOP after VM pack — set AUTO_START_CVM=1 after online VM test ==="
fi
