#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-/home/cwong/Projects/miniconda/envs/mlc}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
PYTHON="${PYTHON:-$CONDA_ENV/bin/python}"

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

export MLC_CUBLAS_SINGLE_DECODE="${MLC_CUBLAS_SINGLE_DECODE:-1}"
export TVM_CUDA_GRAPH_CAPTURE_RNN_STATE_LOOKUPS="${TVM_CUDA_GRAPH_CAPTURE_RNN_STATE_LOOKUPS:-1}"
export MLC_QWEN35_DIRECT_RNN_STATE="${MLC_QWEN35_DIRECT_RNN_STATE:-1}"
export MLC_QWEN35_DIRECT_CONV_STATE="${MLC_QWEN35_DIRECT_CONV_STATE:-1}"
export MLC_QWEN35_COMBINED_KV_CROSS_ATTENTION="${MLC_QWEN35_COMBINED_KV_CROSS_ATTENTION:-0}"
export MLC_QWEN35_EXPLICIT_PAGED_CROSS_ATTENTION="${MLC_QWEN35_EXPLICIT_PAGED_CROSS_ATTENTION:-1}"
export TVM_CUDA_GRAPH_CAPTURE_DECODE_INPUTS="${TVM_CUDA_GRAPH_CAPTURE_DECODE_INPUTS:-1}"
export TVM_CUDA_GRAPH_CAPTURE_KV_CACHE_CROSS_ATTENTION="${TVM_CUDA_GRAPH_CAPTURE_KV_CACHE_CROSS_ATTENTION:-1}"
export TVM_CUDA_GRAPH_CAPTURE_KV_CACHE_APPEND_METADATA="${TVM_CUDA_GRAPH_CAPTURE_KV_CACHE_APPEND_METADATA:-1}"
export TVM_CUDA_GRAPH_RELAXED_KV_CACHE_METADATA_TUPLES="${TVM_CUDA_GRAPH_RELAXED_KV_CACHE_METADATA_TUPLES:-1}"
export TVM_CUDA_GRAPH_CAPTURE_FUNC_OUTPUTS="${TVM_CUDA_GRAPH_CAPTURE_FUNC_OUTPUTS:-1}"
export MLC_QWEN35_FUSED_GDN_DECODE_PREPARE="${MLC_QWEN35_FUSED_GDN_DECODE_PREPARE:-1}"
export MLC_QWEN35_FUSED_GDN_PREFILL_PREPARE="${MLC_QWEN35_FUSED_GDN_PREFILL_PREPARE:-0}"
export MLC_QWEN35_CHUNKED_GDN_PREFILL="${MLC_QWEN35_CHUNKED_GDN_PREFILL:-1}"
export MLC_QWEN35_PACKED_GDN_DECODE="${MLC_QWEN35_PACKED_GDN_DECODE:-1}"
export MLC_QWEN35_DIRECT_RNN_STORAGE_BATCH="${MLC_QWEN35_DIRECT_RNN_STORAGE_BATCH:-2}"
export MLC_QWEN35_DIRECT_RNN_MAX_HISTORY="${MLC_QWEN35_DIRECT_RNN_MAX_HISTORY:-1}"
if [[ "${MLC_QWEN35_FUSED_LM_HEAD_ARGMAX:-0}" != "0" ]]; then
  export MLC_QWEN35_ENABLE_HIDDEN_FUNCS="${MLC_QWEN35_ENABLE_HIDDEN_FUNCS:-1}"
fi
if [[ "${MLC_QWEN35_CHUNKED_GDN_PREFILL:-0}" != "0" ]]; then
  export TVM_CUDA_GRAPH_SKIP_FUNCTIONS="${TVM_CUDA_GRAPH_SKIP_FUNCTIONS:-batch_prefill,batch_prefill_mrope,batch_prefill_to_last_hidden_states}"
fi

MODEL_ID="${1:-familiar-ai/logos-multitask-qwen3.5-2026-05-03-best}"
MODEL_NAME="${MODEL_ID##*/}"

QUANT="${QUANT:-q0f16}"
CONTEXT_WINDOW_SIZE="${CONTEXT_WINDOW_SIZE:-2048}"
PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE:-320}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-1}"
KV_CACHE_PAGE_SIZE="${KV_CACHE_PAGE_SIZE:-16}"
OPT="${OPT:-flashinfer=1;cublas_gemm=1;faster_transformer=0;cudagraph=1;cutlass=1;ipc_allreduce_strategy=NONE}"
MLC_QWEN35_OMIT_VERIFY="${MLC_QWEN35_OMIT_VERIFY:-1}"
export MLC_QWEN35_OMIT_VERIFY
DEFAULT_LIB_SUFFIX="-explicitcrosscg-appendmetacg-relaxedmeta-outputcapture"
if [[ "$MLC_QWEN35_OMIT_VERIFY" == "1" ]]; then
  DEFAULT_LIB_SUFFIX="${DEFAULT_LIB_SUFFIX}-noverify-pc${PREFILL_CHUNK_SIZE}"
fi
LIB_SUFFIX="${LIB_SUFFIX:-$DEFAULT_LIB_SUFFIX}"
MODEL_SUFFIX="${MODEL_SUFFIX:--fusedinproj}"
SKIP_GEN_CONFIG="${SKIP_GEN_CONFIG:-0}"
SKIP_CONVERT="${SKIP_CONVERT:-0}"
DEBUG_DUMP="${DEBUG_DUMP:-}"
DRY_RUN="${DRY_RUN:-0}"

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'dry_run'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

if [[ "$KV_CACHE_PAGE_SIZE" != "16" ]]; then
  export MLC_QWEN35_KV_PAGE_SIZE="$KV_CACHE_PAGE_SIZE"
fi

OUT="dist/${MODEL_NAME}${MODEL_SUFFIX}-${QUANT}-ctx${CONTEXT_WINDOW_SIZE}-pc${PREFILL_CHUNK_SIZE}-MLC"
LIB="dist/libs/${MODEL_NAME}-${QUANT}-ctx${CONTEXT_WINDOW_SIZE}-pc${PREFILL_CHUNK_SIZE}${LIB_SUFFIX}-cuda.so"

mkdir -p dist/models dist/libs

if [[ "$DRY_RUN" == "1" ]]; then
  MODEL_PATH="$MODEL_ID"
else
  MODEL_PATH="$(
    "$PYTHON" - "$MODEL_ID" <<'PY'
import sys
from pathlib import Path

model = sys.argv[1]
path = Path(model)
if path.exists():
    print(path)
else:
    from huggingface_hub import snapshot_download

    print(snapshot_download(repo_id=model))
PY
  )"
fi

if [[ "$SKIP_GEN_CONFIG" != "1" ]]; then
  run_cmd "$PYTHON" -m mlc_llm gen_config "$MODEL_PATH" \
    --model-type qwen3_5 \
    --quantization "$QUANT" \
    --conv-template qwen3_5 \
    --context-window-size "$CONTEXT_WINDOW_SIZE" \
    --prefill-chunk-size "$PREFILL_CHUNK_SIZE" \
    --max-batch-size "$MAX_BATCH_SIZE" \
    --output "$OUT"
fi

if [[ "$SKIP_CONVERT" != "1" ]]; then
  run_cmd "$PYTHON" -m mlc_llm convert_weight "$MODEL_PATH" \
    --model-type qwen3_5 \
    --device cuda \
    --source "$MODEL_PATH" \
    --source-format huggingface-safetensor \
    --quantization "$QUANT" \
    --output "$OUT"
fi

DEBUG_ARGS=()
if [[ -n "$DEBUG_DUMP" ]]; then
  DEBUG_ARGS=(--debug-dump "$DEBUG_DUMP")
fi

run_cmd "$PYTHON" -m mlc_llm compile "$OUT" \
  --model-type qwen3_5 \
  --device cuda \
  --quantization "$QUANT" \
  --opt "$OPT" \
  --overrides "context_window_size=${CONTEXT_WINDOW_SIZE};prefill_chunk_size=${PREFILL_CHUNK_SIZE};max_batch_size=${MAX_BATCH_SIZE}" \
  "${DEBUG_ARGS[@]}" \
  --output "$LIB"
