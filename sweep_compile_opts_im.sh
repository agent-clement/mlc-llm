#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${1:-familiar-ai/logos-multitask-qwen3.5-2026-05-03-best}"

CONTEXT_WINDOW_SIZE="${CONTEXT_WINDOW_SIZE:-2048}"
PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE:-320}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-1}"
QUANT="${QUANT:-q0f16}"

run_variant() {
  local name="$1"
  local opt="$2"
  echo "=== compile ${name}: ${opt} ==="
  QUANT="$QUANT" \
  CONTEXT_WINDOW_SIZE="$CONTEXT_WINDOW_SIZE" \
  PREFILL_CHUNK_SIZE="$PREFILL_CHUNK_SIZE" \
  MAX_BATCH_SIZE="$MAX_BATCH_SIZE" \
  OPT="$opt" \
  LIB_SUFFIX="-${name}" \
  SKIP_GEN_CONFIG="${SKIP_GEN_CONFIG:-1}" \
  SKIP_CONVERT="${SKIP_CONVERT:-1}" \
    ./compile_im.sh "$MODEL_ID"
}

status=0

try_variant() {
  local name="$1"
  local opt="$2"
  if run_variant "$name" "$opt"; then
    echo "=== ${name}: ok ==="
  else
    echo "=== ${name}: failed ===" >&2
    status=1
  fi
}

# Baseline is O0 from compile_im.sh. These isolate compile-time knobs that can
# plausibly affect warmed batch-1 decode for q0f16 Qwen3.5.
try_variant "cublas" "flashinfer=0;cublas_gemm=1;faster_transformer=0;cudagraph=0;cutlass=0"
try_variant "cudagraph" "flashinfer=0;cublas_gemm=0;faster_transformer=0;cudagraph=1;cutlass=0"
try_variant "flashinfer" "flashinfer=1;cublas_gemm=0;faster_transformer=0;cudagraph=0;cutlass=0"
try_variant "cutlass" "flashinfer=0;cublas_gemm=0;faster_transformer=0;cudagraph=0;cutlass=1"
try_variant "ft" "flashinfer=0;cublas_gemm=0;faster_transformer=1;cudagraph=0;cutlass=0"
try_variant "flashinfer-cudagraph" "flashinfer=1;cublas_gemm=0;faster_transformer=0;cudagraph=1;cutlass=0"

try_variant "o2" "O2"
try_variant "o3" "O3"

exit "$status"
