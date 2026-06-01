#!/usr/bin/env bash
# 本地验证集打分（A 榜同款 CD+P2S 百分制）
#
# 首次使用：需先有 benchmark_val（clean/noisy），只需生成一次：
#   python scripts/export_val_benchmark.py
#
# 之后每次改模型，只需下面两步（本脚本默认跳过已存在的 export）：
#   1) predict  2) evaluate.py
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MESH_ROOT="${MESH_ROOT:-/home/dataset_train}"
BENCH_DIR="${BENCH_DIR:-$ROOT/benchmark_val}"
PRED_DIR="${PRED_DIR:-$ROOT/results_val_benchmark}"
TASK="${TASK:-configs/task/predict_val_straightpcf.yaml}"
WORKERS="${WORKERS:-8}"
DO_EXPORT="${DO_EXPORT:-0}"   # 设为 1 才重新导出 benchmark

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate jittor
fi

if [[ "$DO_EXPORT" == "1" ]] || [[ ! -f "$BENCH_DIR/shapenet/03642806/2134ad3fc25a6284193a4c984002ed32/clean.npy" ]]; then
  echo "== [once] Export benchmark_val =="
  python scripts/export_val_benchmark.py \
    --mesh-root "$MESH_ROOT" --out "$BENCH_DIR" --workers "$WORKERS" --skip-existing
else
  echo "== Skip export (benchmark_val exists). Set DO_EXPORT=1 to regenerate. =="
fi

echo "== 1/2 Predict =="
python run.py --task "$TASK" --seed 123

echo "== 2/2 Evaluate (official formula) =="
python evaluate.py \
  --pred_dir "$PRED_DIR" \
  --gt_dir "$BENCH_DIR" \
  --noisy_dir "$BENCH_DIR" \
  --mesh_dir "$MESH_ROOT" \
  --gt_filename clean.npy \
  --noisy_filename noisy.npy \
  --pred_filename denoised.npy \
  --workers "$WORKERS" \
  --verbose

echo "Done. Score above ≈ local A榜 proxy (validate 100 samples, not identical to online test)."
