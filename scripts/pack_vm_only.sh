#!/usr/bin/env bash
# 只打包 VM，不启动 CVM。测完线上后再手动选 vm_ckpt 开 CVM。
set -euo pipefail

ROOT="/home/cslab/CG"
cd "$ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate jittor
export CUDA_VISIBLE_DEVICES=0

PLOG="log/pipeline_multires_300_150k.log"
VM_LOG="log/train_vm_max_multires.log"

log() { echo "[$(date '+%F %T %Z')] $*" | tee -a "$PLOG"; }

log "=== VM PACK ONLY (no CVM) ==="

python3 scripts/pick_multires_stage_ckpts.py \
  --mode epoch --log "$VM_LOG" --ckpt-dir experiments/vm_max_multires \
  --seg-lo 200 --seg-hi 299 --late-lo 280 --late-hi 299 \
  > log/pack_list_vm.txt 2>&1

pack_one() {
  local label="$1" ckpt="$2"
  log "PACK vm ${label} ${ckpt}"
  zip=$(CKPT="$ckpt" PACK_LABEL="$label" SUMMARY="log/pack_vm_${label}_summary.txt" \
    bash scripts/pack_vm_max_multires_ckpt.sh 2>&1 | tee -a "$PLOG" | tail -1)
  log "DONE vm ${label} zip=${zip}"
}

while read -r _ label ckpt; do
  pack_one "$label" "$ckpt"
done < <(grep '^PACK' log/pack_list_vm.txt)

log "=== VM PACK ONLY DONE — submit zips for online test before CVM ==="
cat log/pack_list_vm.txt
