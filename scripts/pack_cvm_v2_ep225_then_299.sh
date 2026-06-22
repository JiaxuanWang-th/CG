#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/cslab/CG"
cd "$ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate jittor
export CUDA_VISIBLE_DEVICES=0

log() { echo "[$(date '+%F %T %Z')] $*" | tee -a log/pack_cvm_v2_ep225_299.log; }

pack_one() {
  local ckpt="$1" label="$2"
  log "START pack ${label} ${ckpt}"
  zip=$(CKPT="$ckpt" PACK_LABEL="$label" SUMMARY="log/pack_cvm_v2_${label}_summary.txt" \
    bash scripts/pack_cvm_max_multires_ckpt.sh 2>&1 | tee -a log/pack_cvm_v2_ep225_299.log | tail -1)
  log "DONE ${label} zip=${zip}"
}

log "=== pack CVM ep225 then ep299 ==="
pack_one experiments/cvm_max_multires_v2/checkpoint_best_cd0.0001_epoch225.pkl ep225_global_best
pack_one experiments/cvm_max_multires_v2/checkpoint_299.pkl ep299_near_end
log "=== ep225+ep299 DONE (ep179 queued separately) ==="
