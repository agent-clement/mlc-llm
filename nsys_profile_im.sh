#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

CONDA_ENV="${CONDA_ENV:-/home/cwong/Projects/miniconda/envs/mlc}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
NSYS="${NSYS:-nsys}"

MODEL_NAME="${MODEL_NAME:-logos-multitask-qwen3.5-2026-05-03-best}"
QUANT="${QUANT:-q0f16}"
CONTEXT_WINDOW_SIZE="${CONTEXT_WINDOW_SIZE:-2048}"
PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE:-320}"
MODEL_SUFFIX="${MODEL_SUFFIX:--fusedinproj}"
LIB_SUFFIX="${LIB_SUFFIX:--combinedkvcross-chunkedgdn-packedgdnopt-cg-skipprefill}"

export CUDA_HOME
export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
export PATH="$CONDA_ENV/bin:$CUDA_HOME/bin:$PATH"
export TVM_LIBRARY_PATH="${TVM_LIBRARY_PATH:-$SCRIPT_DIR/build-tvm-full}"
export LD_LIBRARY_PATH="$SCRIPT_DIR/build-tvm-full:$SCRIPT_DIR/build-mlc-env:$SCRIPT_DIR/build-mlc-env/tvm:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$SCRIPT_DIR/3rdparty/tvm/python:$SCRIPT_DIR/python:${PYTHONPATH:-}"
export MLC_LIBRARY_PATH="${MLC_LIBRARY_PATH:-$SCRIPT_DIR/build-mlc-env}"

export MLC_MODEL="${MLC_MODEL:-$SCRIPT_DIR/dist/${MODEL_NAME}${MODEL_SUFFIX}-${QUANT}-ctx${CONTEXT_WINDOW_SIZE}-pc${PREFILL_CHUNK_SIZE}-MLC}"
export MLC_MODEL_LIB="${MLC_MODEL_LIB:-$SCRIPT_DIR/dist/libs/${MODEL_NAME}-${QUANT}-ctx${CONTEXT_WINDOW_SIZE}-pc${PREFILL_CHUNK_SIZE}${LIB_SUFFIX}-cuda.so}"

REPORT="${NSYS_REPORT:-$SCRIPT_DIR/nsys-qwen35-$(date +%Y%m%d-%H%M%S)}"
CAPTURE_RANGE="${NSYS_CAPTURE_RANGE:-none}"

if [[ "$#" -eq 0 ]]; then
  set -- \
    --image "$SCRIPT_DIR/kentucky.png" \
    --fit-image-size 512 \
    --prompt "What is in the image?" \
    --max-tokens 128 \
    --warmup-runs 1 \
    --benchmark-runs 3
fi

NSYS_ARGS=(
  profile
  --force-overwrite=true
  --trace=cuda,nvtx,osrt
  --sample=none
  --cpuctxsw=none
)

if [[ "$CAPTURE_RANGE" != "none" ]]; then
  NSYS_ARGS+=(
    --capture-range="$CAPTURE_RANGE"
    --capture-range-end=stop
  )
fi

NSYS_ARGS+=(
  --output "$REPORT"
  "$CONDA_ENV/bin/python" "$SCRIPT_DIR/run_im.py"
)

exec "$NSYS" "${NSYS_ARGS[@]}" "$@"
