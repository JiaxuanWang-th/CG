#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/cslab/CG"
cd "$ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate jittor
export CUDA_VISIBLE_DEVICES=0

LOG="log/pack_cvm_v2_ep179.log"
ZIP299="result_cvm_max_multires_epoch299_cdunknown_ep299_near_end.zip"

log() { echo "[$(date '+%F %T %Z')] $*" | tee -a "$LOG"; }

log "=== waiting for ep299 pack to finish ==="
while ! grep -q "DONE ep299_near_end zip=" log/pack_cvm_v2_ep225_299.log 2>/dev/null; do
  sleep 60
done
log "ep299 done (zip=${ZIP299})"

log "START pack ep179_try experiments/cvm_max_multires_v2/checkpoint_179.pkl"
zip=$(CKPT=experiments/cvm_max_multires_v2/checkpoint_179.pkl \
  PACK_LABEL=ep179_try \
  SUMMARY=log/pack_cvm_v2_ep179_try_summary.txt \
  bash scripts/pack_cvm_max_multires_ckpt.sh 2>&1 | tee -a "$LOG" | tail -1)
log "DONE ep179_try zip=${zip}"
log "=== ep179 pack complete ==="
