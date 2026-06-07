#!/usr/bin/env bash
# SPCF max cvm125 50k train：打包 test zip
set -euo pipefail

ROOT="/home/cslab/CG"
cd "$ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate jittor
export CUDA_VISIBLE_DEVICES=0

CKPT_DIR="experiments/straightpcf_max_cvm125_50k"
SUMMARY="log/pack_spcf_max_cvm125_50k_summary.txt"
if [[ -n "${MILESTONES:-}" ]]; then
  read -ra MILESTONES <<< "${MILESTONES}"
else
  MILESTONES=(75000 90000)
fi

log() { echo "[$(date '+%F %T')] $*" | tee -a "$SUMMARY"; }

pick_ckpt() {
  local iter="$1"
  local best="${CKPT_DIR}/checkpoint_best_cd0.0001_iter${iter}.pkl"
  local periodic="${CKPT_DIR}/checkpoint_iter${iter}.pkl"
  if [[ -f "$best" ]]; then
    echo "$best"
  elif [[ -f "$periodic" ]]; then
    echo "$periodic"
  else
    log "ERROR: no ckpt for iter ${iter}"
    return 1
  fi
}

write_predict_yaml() {
  local task_yaml="$1"
  local ckpt="$2"
  local out_dir="$3"
  cat > "$task_yaml" <<YAML
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
YAML
}

pack_one() {
  local iter="$1"
  local ckpt tag out_dir zip pred_log task_yaml n

  ckpt="$(pick_ckpt "$iter")"
  tag="iter${iter}_pick${iter}"
  out_dir="results_spcf_max_cvm125_50k_test_${tag}"
  zip="result_spcf_max_cvm125_50k_${tag}.zip"
  pred_log="log/predict_spcf_max_cvm125_50k_test_${tag}.log"
  task_yaml="configs/task/predict_spcf_max_cvm125_50k_test_${tag}.yaml"

  log "=== PACK ${iter} ckpt=${ckpt} ==="
  write_predict_yaml "$task_yaml" "$ckpt" "$out_dir"

  log "PREDICT test (niters=1 tot_its=3 seed_k=6) -> ${out_dir}"
  python run.py --task "$task_yaml" --seed 123 2>&1 | tee "$pred_log"

  log "ZIP -> ${zip}"
  rm -f "$zip"
  (cd "$out_dir" && zip -r "../$zip" shapenet/)

  n="$(find "$out_dir" -name 'denoised.npy' | wc -l)"
  log "DONE ${tag}: zip=$(ls -lh "$zip" | awk '{print $5}') denoised.npy=${n}"
  echo -e "${tag}\t${iter}\t${ckpt}\t${zip}\t${n}" >> "$SUMMARY"
}

: > "$SUMMARY"
log "SPCF max cvm125 50k pack start | milestones: ${MILESTONES[*]}"
log "reference_online: 32768 SPCF ep125 75k/90k=73.58-73.59"

for iter in "${MILESTONES[@]}"; do
  pack_one "$iter"
done

log "ALL DONE. Summary: ${SUMMARY}"
