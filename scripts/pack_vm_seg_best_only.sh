#!/usr/bin/env bash
# 只重打 seg_best（ep289）。near_end ep299 用已有 zip，勿重复 predict。
set -euo pipefail
ROOT="/home/cslab/CG"
cd "$ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate jittor
export CUDA_VISIBLE_DEVICES=0
PLOG="log/pipeline_multires_300_150k.log"
log() { echo "[$(date '+%F %T %Z')] $*" | tee -a "$PLOG"; }
CKPT="experiments/vm_max_multires/checkpoint_289.pkl"
log "PACK vm seg_best ONLY ${CKPT}"
zip=$(CKPT="$CKPT" PACK_LABEL=seg_best SUMMARY=log/pack_vm_seg_best_summary.txt \
  bash scripts/pack_vm_max_multires_ckpt.sh 2>&1 | tee -a "$PLOG" | tail -1)
log "DONE seg_best zip=${zip}"
log "ep299 zip already: result_vm_max_multires_epoch299_cdunknown_near_end.zip (from first pack, no re-predict)"
