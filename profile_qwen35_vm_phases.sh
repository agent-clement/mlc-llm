#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-/home/cwong/Projects/miniconda/envs/mlc}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"

export CUDA_HOME
export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
export PATH="$CONDA_ENV/bin:$CUDA_HOME/bin:$PATH"
export TVM_LIBRARY_PATH="${TVM_LIBRARY_PATH:-$SCRIPT_DIR/build-tvm-full}"
export MLC_LIBRARY_PATH="${MLC_LIBRARY_PATH:-$SCRIPT_DIR/build-mlc-env}"
export PYTHONPATH="$SCRIPT_DIR/3rdparty/tvm/python:$SCRIPT_DIR/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$SCRIPT_DIR/build-mlc-env:$SCRIPT_DIR/build-mlc-env/tvm:$SCRIPT_DIR/build-tvm-full:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

exec "$CONDA_ENV/bin/python" "$SCRIPT_DIR/profile_qwen35_vm_phases.py" "$@"
