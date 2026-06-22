#!/usr/bin/env bash
# VM 已训完 ep299 -> 打包 2 个 zip -> 启动 CVM v2
set -euo pipefail

ROOT="/home/cslab/CG"
cd "$ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate jittor
export CUDA_VISIBLE_DEVICES=0

PLOG="log/pipeline_multires_300_150k.log"
VM_LOG="log/train_vm_max_multires.log"
CVM_LOG="log/train_cvm_max_multires_v2.log"

log() { echo "[$(date '+%F %T %Z')] $*" | tee -a "$PLOG"; }

pack_one() {
  local label="$1" ckpt="$2"
  log "PACK vm ${label} ${ckpt}"
  zip=$(CKPT="$ckpt" PACK_LABEL="$label" SUMMARY="log/pack_vm_${label}_summary.txt" bash scripts/pack_vm_max_multires_ckpt.sh 2>&1 | tee -a "$PLOG" | tail -1)
  log "DONE vm ${label} zip=${zip}"
}

log "=== RESUME: VM pack -> CVM train ==="
log "WARN: CVM launch removed — submit & test VM zips first, then set vm_ckpt manually"

pack_one seg_best experiments/vm_max_multires/checkpoint_289.pkl
pack_one near_end experiments/vm_max_multires/checkpoint_299.pkl

log "=== VM pack done. DO NOT auto-start CVM. Test zips online first. ==="
