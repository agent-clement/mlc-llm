#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

CONDA_ENV="${CONDA_ENV:-/home/cwong/Projects/miniconda/envs/mlc}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"

export CUDA_HOME
export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
export PATH="$CONDA_ENV/bin:$CUDA_HOME/bin:$PATH"
if [[ "${MLC_USE_LOCAL_TVM:-1}" == "1" && -d "$SCRIPT_DIR/build-tvm-full" ]]; then
  export TVM_LIBRARY_PATH="${TVM_LIBRARY_PATH:-$SCRIPT_DIR/build-tvm-full}"
  export LD_LIBRARY_PATH="$SCRIPT_DIR/build-tvm-full:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="$SCRIPT_DIR/3rdparty/tvm/python:$SCRIPT_DIR/python:${PYTHONPATH:-}"
else
  export PYTHONPATH="$SCRIPT_DIR/python:${PYTHONPATH:-}"
fi
export LD_LIBRARY_PATH="$SCRIPT_DIR/build-mlc-env:$SCRIPT_DIR/build-mlc-env/tvm:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export MLC_LIBRARY_PATH="${MLC_LIBRARY_PATH:-$SCRIPT_DIR/build-mlc-env}"

exec "$CONDA_ENV/bin/python" "$SCRIPT_DIR/check_qwen35_im_parity.py" "$@"
