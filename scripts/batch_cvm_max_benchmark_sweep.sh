#!/usr/bin/env bash
# Phase 1: benchmark all remaining cvm_max checkpoints
# Phase 2: predict-param sweep on best checkpoint
set -euo pipefail

ROOT="/home/cslab/CG"
cd "$ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate jittor
export CUDA_VISIBLE_DEVICES=0

SUMMARY="$ROOT/cvm_max_benchmark_sweep_summary.log"
TASK_SWEEP="$ROOT/configs/task/predict_val_cvm_max_sweep.yaml"
MODEL_SWEEP="$ROOT/configs/model/cvm_max_predict_sweep.yaml"
BASE_MODEL="$ROOT/configs/model/cvm_max.yaml"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$SUMMARY"; }

resolve_ckpt() {
  local ep="$1"
  local best periodic
  for best in "$ROOT"/experiments/cvm_max/checkpoint_best_cd*_epoch${ep}.pkl; do
    if [[ -f "$best" ]]; then echo "$best"; return 0; fi
  done
  periodic="$ROOT/experiments/cvm_max/checkpoint_${ep}.pkl"
  if [[ -f "$periodic" ]]; then echo "$periodic"; return 0; fi
  return 1
}

run_eval() {
  local tag="$1" ckpt="$2" pred_dir="$3"
  local pred_log="$ROOT/eval_cvm_max_${tag}_predict.log"
  local score_log="$ROOT/eval_cvm_max_${tag}_score.log"

  python - "$ckpt" "$pred_dir" <<'PY'
import re, sys
from pathlib import Path
ckpt, pred_dir = sys.argv[1], sys.argv[2]
p = Path("configs/task/predict_val_cvm_max_sweep.yaml")
text = p.read_text() if p.exists() else Path("configs/task/predict_val_cvm_max_epoch97.yaml").read_text()
text = re.sub(r"load_ckpt: .*", f"load_ckpt: {ckpt}", text)
text = re.sub(r"save_dir: .*", f"save_dir: {pred_dir}", text)
text = re.sub(r"model: .*", "model: cvm_max", text)
Path("configs/task/predict_val_cvm_max_sweep.yaml").write_text(text)
PY

  log "PREDICT start tag=$tag ckpt=$ckpt"
  python run.py --task configs/task/predict_val_cvm_max_sweep.yaml --seed 123 \
    2>&1 | tee "$pred_log"

  python evaluate.py \
    --pred_dir "$pred_dir" \
    --gt_dir benchmark_val \
    --noisy_dir benchmark_val \
    --mesh_dir /home/dataset_train \
    --gt_filename clean.npy \
    --noisy_filename noisy.npy \
    --pred_filename denoised.npy \
    --workers 8 \
    --verbose 2>&1 | tee "$score_log"

  local score cd
  score=$(grep '最终得分' "$score_log" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "NA")
  cd=$(grep '平均 CD_pred' "$score_log" | tail -1 | grep -oE '0\.[0-9]+' | head -1 || echo "NA")
  log "SCORE tag=$tag total=$score CD_pred=$cd ckpt=$ckpt"
  printf '%s\t%s\t%s\t%s\n' "$tag" "$score" "$cd" "$ckpt" >> "$ROOT/cvm_max_benchmark_sweep_scores.tsv"
}

write_model_sweep() {
  local niters="$1" tot_its="$2" seed_k="$3"
  python - "$niters" "$tot_its" "$seed_k" "$BASE_MODEL" "$MODEL_SWEEP" <<'PY'
import sys
from pathlib import Path
niters, tot_its, seed_k = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
base = Path(sys.argv[4]).read_text().splitlines()
out = []
for line in base:
    if line.startswith("tot_its:"):
        out.append(f"tot_its: {tot_its}")
    elif line.startswith("seed_k:"):
        out.append(f"seed_k: {seed_k}")
    elif line.startswith("niters:"):
        out.append(f"niters: {niters}")
    else:
        out.append(line)
if not any(l.startswith("niters:") for l in out):
    out.append(f"niters: {niters}")
Path(sys.argv[5]).write_text("\n".join(out) + "\n")
PY
}

run_eval_sweep() {
  local tag="$1" ckpt="$2" niters="$3" tot_its="$4" seed_k="$5"
  local pred_dir="results_val_cvm_max_sweep_${tag}"
  local pred_log="$ROOT/eval_cvm_max_sweep_${tag}_predict.log"
  local score_log="$ROOT/eval_cvm_max_sweep_${tag}_score.log"

  write_model_sweep "$niters" "$tot_its" "$seed_k"

  python - "$ckpt" "$pred_dir" <<'PY'
import re, sys
from pathlib import Path
ckpt, pred_dir = sys.argv[1], sys.argv[2]
text = Path("configs/task/predict_val_cvm_max_epoch97.yaml").read_text()
text = re.sub(r"load_ckpt: .*", f"load_ckpt: {ckpt}", text)
text = re.sub(r"save_dir: .*", f"save_dir: {pred_dir}", text)
text = re.sub(r"model: .*", "model: cvm_max_predict_sweep", text)
Path("configs/task/predict_val_cvm_max_sweep.yaml").write_text(text)
PY

  log "SWEEP PREDICT tag=$tag niters=$niters tot_its=$tot_its seed_k=$seed_k"
  python run.py --task configs/task/predict_val_cvm_max_sweep.yaml --seed 123 \
    2>&1 | tee "$pred_log"

  python evaluate.py \
    --pred_dir "$pred_dir" \
    --gt_dir benchmark_val \
    --noisy_dir benchmark_val \
    --mesh_dir /home/dataset_train \
    --gt_filename clean.npy \
    --noisy_filename noisy.npy \
    --pred_filename denoised.npy \
    --workers 8 \
    --verbose 2>&1 | tee "$score_log"

  local score cd
  score=$(grep '最终得分' "$score_log" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "NA")
  cd=$(grep '平均 CD_pred' "$score_log" | tail -1 | grep -oE '0\.[0-9]+' | head -1 || echo "NA")
  log "SWEEP SCORE tag=$tag total=$score CD_pred=$cd params=n${niters}_t${tot_its}_k${seed_k}"
  printf 'sweep_%s\t%s\t%s\tn%s_t%s_k%s\t%s\n' "$tag" "$score" "$cd" "$niters" "$tot_its" "$seed_k" "$ckpt" >> "$ROOT/cvm_max_benchmark_sweep_scores.tsv"
}

# ---------- init ----------
: > "$SUMMARY"
printf 'tag\ttotal\tcd_pred\tckpt\n' > "$ROOT/cvm_max_benchmark_sweep_scores.tsv"

log "========== Phase 1: remaining CVM max checkpoints =========="
# Already done: 39,49,56,66,97,99
REMAINING_EPOCHS=(0 6 8 9 19 29 59 69 79 88 89)

for ep in "${REMAINING_EPOCHS[@]}"; do
  if ckpt=$(resolve_ckpt "$ep"); then
    run_eval "epoch${ep}" "$ckpt" "results_val_cvm_max_epoch${ep}"
  else
    log "SKIP epoch${ep}: no checkpoint found"
  fi
done

log "========== Phase 1 done; picking best checkpoint =========="

# Merge with known scores from prior runs
KNOWN=(
  "epoch39|75.28|experiments/cvm_max/checkpoint_best_cd0.0002_epoch39.pkl"
  "epoch49|76.42|experiments/cvm_max/checkpoint_best_cd0.0002_epoch49.pkl"
  "epoch56|77.37|experiments/cvm_max/checkpoint_best_cd0.0002_epoch56.pkl"
  "epoch66|74.27|experiments/cvm_max/checkpoint_best_cd0.0002_epoch66.pkl"
  "epoch97|76.53|experiments/cvm_max/checkpoint_best_cd0.0002_epoch97.pkl"
  "epoch99|76.28|experiments/cvm_max/checkpoint_best_cd0.0002_epoch99.pkl"
)
for row in "${KNOWN[@]}"; do
  IFS='|' read -r tag score ckpt <<< "$row"
  printf '%s\t%s\tNA\t%s\n' "$tag" "$score" "$ckpt" >> "$ROOT/cvm_max_benchmark_sweep_scores.tsv"
done

read -r BEST_SCORE BEST_TAG BEST_CKPT <<< "$(python - <<'PY'
import csv
best = None
with open("cvm_max_benchmark_sweep_scores.tsv") as f:
    for tag, total, cd_pred, ckpt in csv.reader(f, delimiter="\t"):
        if tag == "tag" or tag.startswith("sweep_"):
            continue
        try:
            score = float(total)
        except ValueError:
            continue
        if best is None or score > best[0]:
            best = (score, tag, ckpt)
if best:
    print(best[0], best[1], best[2])
else:
    print("77.37 epoch56 experiments/cvm_max/checkpoint_best_cd0.0002_epoch56.pkl")
PY
)"

log "BEST checkpoint: tag=$BEST_TAG score=$BEST_SCORE ckpt=$BEST_CKPT"

log "========== Phase 2: predict param sweep on best ckpt =========="
SWEEPS=(
  "base 1 3 6"
  "n2 2 3 6"
  "n3 3 3 6"
  "t4 1 4 6"
  "n2t4 2 4 6"
  "n3t4 3 4 6"
  "t5 1 5 6"
  "n2t5 2 5 6"
  "k4 1 3 4"
  "k8 1 3 8"
  "n2k8 2 3 8"
  "n3t4k8 3 4 8"
)

for item in "${SWEEPS[@]}"; do
  read -r tag niters tot_its seed_k <<< "$item"
  run_eval_sweep "$tag" "$BEST_CKPT" "$niters" "$tot_its" "$seed_k"
done

log "========== ALL DONE =========="
log "See $SUMMARY and cvm_max_benchmark_sweep_scores.tsv"
{ echo ""; echo "=== Top 20 by total score ==="; sort -t$'\t' -k2 -rn cvm_max_benchmark_sweep_scores.tsv | head -20; } | tee -a "$SUMMARY"
