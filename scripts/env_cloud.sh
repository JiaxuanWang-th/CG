#!/bin/bash
# Source before running CG training on this cloud machine.
export CG=/home/ubuntu/CG
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$HOME/miniconda3/envs/jittor/bin:$HOME/miniconda3/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/jittor/lib:${LD_LIBRARY_PATH:-}"