#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

CONDA_ENV="${CONDA_ENV:-/home/cwong/Projects/miniconda/envs/vllm}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"

export CUDA_HOME
export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
export PATH="$CONDA_ENV/bin:$CUDA_HOME/bin:$PATH"
CU13_LIB="$CONDA_ENV/lib/python3.11/site-packages/nvidia/cu13/lib"
if [[ -d "$CU13_LIB" ]]; then
  export LD_LIBRARY_PATH="$CU13_LIB:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

exec "$CONDA_ENV/bin/python" "$SCRIPT_DIR/run_vllm_im.py" "$@"
