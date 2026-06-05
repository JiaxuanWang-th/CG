#!/usr/bin/env bash
# SPCF max 续训：仅打包 250k 里程碑（不打包 200k）
#
# 用法：
#   bash scripts/pack_spcf_max_iter250000.sh
#   WAIT=0 bash scripts/pack_spcf_max_iter250000.sh
set -euo pipefail
ROOT="/home/cslab/CG"
cd "$ROOT"
export MILESTONES="250000"
export WAIT="${WAIT:-1}"
exec bash scripts/pack_spcf_max_resume_200_250.sh
