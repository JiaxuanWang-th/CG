#!/usr/bin/env bash
# CVM max multires：串行打包 ep121(best) -> ep149(final)，zip 用 -1 加速
set -euo pipefail

ROOT="/home/cslab/CG"
cd "$ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate jittor
export CUDA_VISIBLE_DEVICES=0

MILESTONES=(121 149)
DATA_COMPONENT="${DATA_COMPONENT:-predict_fast}"
SUMMARY="log/pack_cvm_max_multires_121_149_summary.txt"

mkdir -p log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$SUMMARY"; }

pick_ckpt() {
  local epoch="$1"
  local best="experiments/cvm_max_multires/checkpoint_best_cd0.0002_epoch${epoch}.pkl"
  local periodic="experiments/cvm_max_multires/checkpoint_${epoch}.pkl"
  if [[ -f "$best" ]]; then
    echo "$best"
  elif [[ -f "$periodic" ]]; then
    echo "$periodic"
  else
    log "ERROR: no ckpt for epoch ${epoch}"
    return 1
  fi
}

pack_one() {
  local epoch="$1"
  local ckpt out_dir zip pred_log task_yaml n

  ckpt="$(pick_ckpt "$epoch")"
  out_dir="results_cvm_max_multires_test_epoch${epoch}"
  zip="result_cvm_max_multires_epoch${epoch}.zip"
  pred_log="log/predict_cvm_max_multires_test_epoch${epoch}.log"
  task_yaml="configs/task/predict_cvm_max_multires_test_epoch${epoch}.yaml"

  log "=== PACK epoch=${epoch} ckpt=${ckpt} ==="
  log "reference: ep80_online=70.52 | CVM125=72.72 | best_val=0.000152@121"

  cat > "$task_yaml" <<EOF
mode: predict
debug: false

load_ckpt: ${ckpt}

components:
  data: ${DATA_COMPONENT}
  transform: predict
  system: vm
  model: cvm_max

writer:
  __target__: vm
  save_dir: ${out_dir}
  save_name: denoised
EOF

  log "PREDICT test -> ${out_dir} (data=${DATA_COMPONENT})"
  python run.py --task "$task_yaml" --seed 123 2>&1 | tee "$pred_log"

  log "ZIP -> ${zip} (compression -1)"
  rm -f "$zip"
  (cd "$out_dir" && zip -r -1 "../$zip" shapenet/)

  n="$(find "$out_dir" -name 'denoised.npy' | wc -l)"
  log "DONE epoch=${epoch}: zip=$(ls -lh "$zip" | awk '{print $5}') denoised.npy=${n}"
}

: > "$SUMMARY"
log "CVM max multires batch pack | milestones=${MILESTONES[*]}"

for epoch in "${MILESTONES[@]}"; do
  pack_one "$epoch"
done

log "ALL DONE: $(ls -lh result_cvm_max_multires_epoch121.zip result_cvm_max_multires_epoch149.zip 2>/dev/null | awk '{print $9, $5}')"
