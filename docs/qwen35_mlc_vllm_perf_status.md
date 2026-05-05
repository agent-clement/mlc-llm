# Qwen3.5 MLC vs vLLM Performance Status

This note tracks the current apples-to-apples batch-1 image benchmark state for
`familiar-ai/logos-multitask-qwen3.5-2026-05-03-best`.

## Current Baseline

Run the current MLC path with:

```bash
MLC_MODEL=dist/logos-multitask-qwen3.5-2026-05-03-best-fusedinproj-q0f16-ctx2048-pc320-MLC \
MLC_MODEL_LIB=dist/libs/logos-multitask-qwen3.5-2026-05-03-best-q0f16-ctx2048-pc320-explicitcrosscg-appendmetacg-relaxedmeta-outputcapture-cuda.so \
./run_im.sh \
  --image kentucky.png \
  --fit-image-size 512 \
  --prompt "What is in the image?" \
  --max-tokens 128 \
  --warmup-runs 1 \
  --benchmark-runs 5 \
  --ignore-eos
```

Or run both current wrappers with:

```bash
MLC_MODEL=dist/logos-multitask-qwen3.5-2026-05-03-best-fusedinproj-q0f16-ctx2048-pc320-MLC \
MLC_MODEL_LIB=dist/libs/logos-multitask-qwen3.5-2026-05-03-best-q0f16-ctx2048-pc320-explicitcrosscg-appendmetacg-relaxedmeta-outputcapture-cuda.so \
./benchmark_qwen35_vllm_mlc.py \
  --image kentucky.png \
  --max-tokens 128 \
  --warmup-runs 1 \
  --benchmark-runs 5
```

Short end-to-end wrapper smoke, used to validate command wiring and guard
parsing without treating the result as a performance comparison:

```bash
./benchmark_qwen35_vllm_mlc.py \
  --image kentucky.png \
  --max-tokens 16 \
  --warmup-runs 0 \
  --benchmark-runs 2
```

On 2026-05-04 this smoke passed the real MLC and vLLM paths with
`image_grid_match=1x32x32`, `generated_token_count_match=16`, and the expected
`comparison_warning=short_or_cold_run`.

The combined wrapper parses both engines' image grids and generated token
counts. It fails by default if either differs or is missing on a comparable
two-engine run. Use
`--allow-image-grid-mismatch` or `--allow-token-count-mismatch` only for
intentional diagnostics. It only reports MLC-vLLM throughput deltas for
comparable settings (`max_tokens >= 128`, `warmup_runs >= 1`, and
`benchmark_runs >= 5`). Short smoke runs still validate commands, paths,
image-grid matching, and generated-token matching, but print a comparison
warning instead of a misleading throughput gap.

The vLLM and Transformers runners accept `--processor-use-fast auto|true|false`
or `VLM_PROCESSOR_USE_FAST` to make the `AutoProcessor(use_fast=...)` choice
explicit. The default `auto` preserves the installed Transformers default, and
invalid `VLM_PROCESSOR_USE_FAST` values fail during argument parsing instead of
being silently coerced.

The Transformers runner also accepts `--ignore-eos`, matching the fixed-token
MLC/vLLM benchmark mode. This keeps standalone Transformers baseline runs from
ending early on EOS when the goal is a decode-throughput comparison.

The wrapper's non-engine logic is covered by a focused pytest:

```bash
/home/cwong/Projects/miniconda/envs/mlc/bin/python \
  -m pytest -q tests/python/support/test_qwen35_benchmark_wrapper.py
```

That test covers normalized MLC/vLLM image-grid parsing, the default failure on
grid mismatch, the intentional mismatch override, and the no-delta warning for
short or cold runs. It also checks that remote HF model IDs stay unchanged while
existing local vLLM model paths are resolved to absolute paths before launch,
and that `run_vllm_im.py` uses `HF_MODEL` first, then the prepared local model
copy when present. The same test also covers vLLM tokenizer compatibility
rewriting from `TokenizersBackend` to `Qwen2TokenizerFast` for local model
copies, plus MLC wrapper command construction for the current runtime defaults
and experimental KV-cache page-size overrides. It also tests representative
MLC/vLLM subprocess output parsing without launching either engine, including
throughput, image-grid, MLC runtime, effective decode, and vLLM cache fields.
It also verifies fail-fast behavior for subprocess failures and missing
benchmark metric lines, plus missing or mismatched image-grid and
generated-token-count fields between engines. It also covers the explicit
processor-speed argument plumbing, environment defaults, CLI override
precedence, and keeps the runner imports lazy enough for cheap `--help` and
parser tests.
It includes a fixed-canvas local image resize parity check across the MLC,
vLLM, and Transformers runner code paths.

For phase-attribution runs, use synchronized decode timing explicitly:

```bash
./run_im.sh \
  --image kentucky.png \
  --fit-image-size 512 \
  --prompt "What is in the image?" \
  --max-tokens 128 \
  --warmup-runs 1 \
  --benchmark-runs 5 \
  --ignore-eos \
  --sync-decode-timing
```

or through the two-engine wrapper:

```bash
./benchmark_qwen35_vllm_mlc.py \
  --image kentucky.png \
  --max-tokens 128 \
  --warmup-runs 1 \
  --benchmark-runs 5 \
  --mlc-sync-decode-timing 1
```

This sets `MLC_SYNC_DECODE_TIMING=1`, making the MLC
`model_seconds`/`probs_seconds`/`sample_seconds` buckets wait for queued CUDA
work before recording. It is the right mode for diagnosing whether a branch
really moved model compute or just shifted the async synchronization point; it
is not the default fast throughput mode.

The shell entrypoints were also checked with `./run_vllm_im.sh --help` and
`./run_transformers_im.sh --help`; both expose `--processor-use-fast` without
loading the full model stack, and the Transformers shell entrypoint exposes
`--ignore-eos`.

The Qwen3.5 parity helper also accepts `--processor-use-fast` and lazy-loads
torch/Transformers model code. `./check_qwen35_im_parity.sh --help` exposes the
flag without loading the model stack.

The vLLM checkpoint-preparation helper has pure tests for the Familiar AI
Qwen3.5 key remapping rules and for copying tokenizer/config files while
skipping heavyweight safetensor weight files. It lazy-loads torch,
Hugging Face Hub, and safetensors only when actually preparing weights;
`prepare_vllm_qwen35_im_model.py --help` works without loading them.

The shell entrypoints pass syntax validation with:

```bash
bash -n compile_im.sh run_im.sh run_vllm_im.sh run_transformers_im.sh \
  check_qwen35_im_parity.sh
```

This specifically covers the quoted semicolon-heavy `--opt` and `--overrides`
arguments in `compile_im.sh` without starting a full compile.

`compile_im.sh` also supports `DRY_RUN=1`, which skips model download/compile
and prints the exact commands with shell escaping:

```bash
DRY_RUN=1 \
OPT='flashinfer=1;cublas_gemm=1;cudagraph=1;cutlass=1' \
PREFILL_CHUNK_SIZE=320 \
CONTEXT_WINDOW_SIZE=2048 \
./compile_im.sh familiar-ai/logos-multitask-qwen3.5-2026-05-03-best
```

The dry run shows semicolons escaped in both `--opt` and `--overrides`, directly
guarding the original shell-splitting failure mode.

The Nsight decode comparison helper has a separate pure-SQLite test:

```bash
/home/cwong/Projects/miniconda/envs/mlc/bin/python \
  -m pytest -q tests/python/support/test_compare_nsys_decode.py
```

That test covers kernel/runtime categorization, per-token normalization, and
the auto token-count inference from dominant decode GEMV/cuBLAS kernels.

The current saved Nsight exports were compared with:

```bash
/home/cwong/Projects/miniconda/envs/mlc/bin/python scripts/compare_nsys_decode.py \
  --left nsys-qwen35-currentbest-latest.sqlite \
  --left-name MLC \
  --right nsys-vllm-qwen35-current.sqlite \
  --right-name vLLM
```

The helper inferred `MLC=512` decode tokens and `vLLM=384` decode tokens from
dominant GEMV/cuBLAS kernels before normalizing per token. The largest
kernel-side MLC overages were elementwise work (~93 us/tok), flash/full
attention work (~52 us/tok), GEMV/GEMM (~28 us/tok), and other small kernels
(~25 us/tok). The runtime-side overages were dominated by memcpy
(~1144 us/tok), sync (~180 us/tok), alloc/free (~58 us/tok), graph launch
(~26 us/tok), and kernel launch (~16 us/tok). Treat these as profile-alignment
signals from saved traces, not as a replacement for the guarded wrapper
throughput run.

A refreshed current-best MLC capture on 2026-05-04, using
`MLC_DEFER_CPU_TOKEN_BURST=1`, `MLC_DECODE_BURST_STEPS=16`, and
`MLC_STABLE_DECODE_EMBEDDING=1`, shows that the earlier memcpy and alloc/free
signals are no longer the main gap. Compared with the saved vLLM trace, the
remaining normalized MLC overages are:

- elementwise kernels: about 86.6 us/tok
- flash/full-attention kernels: about 52.5 us/tok
- GEMV/GEMM: about 19.2 us/tok
- graph launches: about 30.6 us/tok

The same refresh shows MLC is now better than the saved vLLM trace on memcpy
and alloc/free in this isolated run, so those should no longer be treated as
the next optimization target.

The Qwen3.5 KV-cache / FA2 metadata regression was also re-run after the
automatic CUDA graph capture-stream runtime change:

```bash
PYTHONPATH=$PWD/3rdparty/tvm/python:$PWD/python \
TVM_LIBRARY_PATH=$PWD/build-tvm-full \
LD_LIBRARY_PATH=$PWD/build-tvm-full:$PWD/build-mlc-env:$PWD/build-mlc-env/tvm:${LD_LIBRARY_PATH:-} \
/home/cwong/Projects/miniconda/envs/mlc/bin/python tests/python/model/test_kv_cache.py
```

It completed successfully.

The current decode graph shape can be checked mechanically with:

```bash
/home/cwong/Projects/miniconda/envs/mlc/bin/python analyze_cuda_graph_dump.py \
  debug-explicitcrosscg-appendmetacg-relaxedmeta-outputcapture/debug-phase6.py \
  --function decode_mrope \
  --expect-cuda-graph-calls 1 \
  --expect-packed-call vm.builtin.attention_kv_cache_get_query_positions=1 \
  --expect-no-packed-call vm.builtin.attention_kv_cache_append_mha_kv \
  --expect-no-packed-call vm.builtin.attention_kv_cache_get_paged_decode_metadata
```

This is the lightweight guard for the current best graph structure: append,
paged-metadata, cross-attention, and LM-head calls are captured in one decode
graph, while the single query-position lookup remains outside graph capture.

To summarize only the lifted regions for the fixed batch-1 decode path, use:

```bash
/home/cwong/Projects/miniconda/envs/mlc/bin/python analyze_cuda_graph_dump.py \
  debug-explicitcrosscg-appendmetacg-relaxedmeta-outputcapture/debug-phase6.py \
  --regions \
  --parent decode
```

This avoids mixing `batch_decode_cuda_graph_capture*` with
`decode_cuda_graph_capture*` when counting captured regions. The region summary
prints the ordered capture index, line count, packed-call counters, and first
and last significant calls for each lifted graph. That is the quickest way to
check whether append/metadata creation and the following paged cross-attention
are still split across adjacent captures.

The focused Qwen3.5 model test was also re-run against the local TVM/MLC build:

```bash
PYTHONPATH=$PWD/3rdparty/tvm/python:$PWD/python \
TVM_LIBRARY_PATH=$PWD/build-tvm-full \
LD_LIBRARY_PATH=$PWD/build-tvm-full:$PWD/build-mlc-env:$PWD/build-mlc-env/tvm:${LD_LIBRARY_PATH:-} \
/home/cwong/Projects/miniconda/envs/mlc/bin/python tests/python/model/test_qwen35.py
```

It completed successfully.

The focused MRoPE op regression was also re-run:

```bash
PYTHONPATH=$PWD/3rdparty/tvm/python:$PWD/python \
TVM_LIBRARY_PATH=$PWD/build-tvm-full \
LD_LIBRARY_PATH=$PWD/build-tvm-full:$PWD/build-mlc-env:$PWD/build-mlc-env/tvm:${LD_LIBRARY_PATH:-} \
/home/cwong/Projects/miniconda/envs/mlc/bin/python tests/python/op/test_mrope.py
```

It completed successfully.

The Triton chunk-local cumulative-sum op regression, used by the chunked GDN
prefill path, was also re-run:

```bash
PYTHONPATH=$PWD/3rdparty/tvm/python:$PWD/python \
TVM_LIBRARY_PATH=$PWD/build-tvm-full \
LD_LIBRARY_PATH=$PWD/build-tvm-full:$PWD/build-mlc-env:$PWD/build-mlc-env/tvm:${LD_LIBRARY_PATH:-} \
/home/cwong/Projects/miniconda/envs/mlc/bin/python tests/python/op/test_triton_chunk_local_cumsum.py
```

It completed successfully.

Current measured MLC generated throughput is about 414 tokens/s with the
cross-attention graph capture plus append/metadata capture plus relaxed
metadata-tuple propagation plus function-output capture artifact:

- model lib: `dist/libs/logos-multitask-qwen3.5-2026-05-03-best-q0f16-ctx2048-pc320-explicitcrosscg-appendmetacg-relaxedmeta-outputcapture-cuda.so`
- model dir: `dist/logos-multitask-qwen3.5-2026-05-03-best-fusedinproj-q0f16-ctx2048-pc320-MLC`
- existing serving defaults:
  `MLC_DEFER_CPU_TOKEN_BURST=1`, `MLC_DECODE_BURST_STEPS=16`,
  `MLC_STABLE_DECODE_EMBEDDING=1`, and `MLC_KV_CACHE_PAGE_SIZE=16`

The TVM runtime now marks the active thread while it is capturing a CUDA graph,
so FlashInfer MHA uses the capture stream automatically during capture. The
manual `TVM_FLASHINFER_MHA_USE_CURRENT_STREAM=1` knob remains available as an
override, but is no longer required for this path.

On 2026-05-04, after rebuilding TVM with the automatic capture-stream scope,
append/metadata capture, and relaxed paged-metadata tuple propagation, this
path produced coherent Kentucky Derby output without
`TVM_FLASHINFER_MHA_USE_CURRENT_STREAM`. A standalone 10-run serving benchmark
measured 414.41 generated tok/s / 446.05 effective decode tok/s. A matching
10-run two-engine wrapper comparison measured MLC at 414.23 generated tok/s /
446.09 effective decode tok/s and vLLM at 413.30 generated tok/s, putting MLC
ahead by 0.93 tok/s, or 0.23%, on that run. The previous append/metadata
capture artifact measured 410.27 generated tok/s / 441.48 effective decode
tok/s with vLLM at 414.48 generated tok/s. The relaxed metadata-tuple path is
the best confirmed MLC path so far.

The latest 10-run wrapper comparison in the working `vllm` conda env measured the
automatic capture-stream MLC cross-attention graph plus append/metadata,
relaxed metadata-tuple propagation, and output-capture candidate at 414.23
tokens/s and vLLM at 413.30 tokens/s using the prepared local vLLM model copy,
the same
normalized image grid (`1x32x32`), and the same generated token count (`128`):

```bash
./run_vllm_im.sh \
  --image kentucky.png \
  --fit-image-width 512 \
  --fit-image-height 512 \
  --prompt "What is in the image?" \
  --max-tokens 128 \
  --warmup-runs 1 \
  --benchmark-runs 10
```

The raw HF checkpoint path still fails in the installed vLLM env because that
loader expects the vLLM-renamed parameter keys. Use the prepared local copy for
valid apples-to-apples measurements. `run_vllm_im.py` now defaults to that
local copy when it exists, using a path relative to the script location; `HF_MODEL`
or `--hf-model` still override it. The current measured gap is about
4.20 tokens/s, or roughly 1.01%, not the older 412 vs 341 tokens/s gap.

Most recent aligned direct `run_im.sh` measurement:

- MLC current best clean one-token serving run:
  399.91 generated tokens/s, 433.19 decode tokens/s.
- MLC current best deferred-burst run:
  404.67-404.75 generated tokens/s and 434.9-435.3 effective decode tokens/s.
- Fresh 5-run wrapper check after updating defaults and stricter
  apples-to-apples guards:
  MLC 403.10 generated tok/s / 433.26 effective decode tok/s, vLLM 411.95
  generated tok/s with 23.68 s model load/startup, delta 8.85 tok/s.
- Fresh direct 5-run check after rebuilding both TVM runtime trees:
  MLC 404.70 generated tok/s / 435.04 effective decode tok/s.
- Fresh 5-run wrapper check with the diagnostic current-stream fixed
  cross-attention graph candidate:
  MLC 406.24 generated tok/s / 436.80 effective decode tok/s, vLLM 413.58
  generated tok/s with 23.53 s model load/startup, delta 7.34 tok/s.
- Fresh 5-run wrapper check with diagnostic current-stream fixed
  cross-attention graph plus function-output capture:
  MLC 406.28 generated tok/s / 436.61 effective decode tok/s, vLLM 413.03
  generated tok/s with 23.69 s model load/startup, delta 6.75 tok/s.
- Fresh 5-run wrapper check after making capture-stream selection automatic:
  MLC 406.68 generated tok/s / 437.17 effective decode tok/s, vLLM 414.10
  generated tok/s with 23.82 s model load/startup, delta 7.42 tok/s.
- Fresh 5-run wrapper check after capturing append and paged-metadata builtins:
  MLC 411.02 generated tok/s / 441.83 effective decode tok/s, vLLM 411.59
  generated tok/s with 23.65 s model load/startup, delta 0.57 tok/s.
- Fresh 10-run wrapper check after capturing append and paged-metadata builtins:
  MLC 410.27 generated tok/s / 441.48 effective decode tok/s, vLLM 414.48
  generated tok/s with 23.50 s model load/startup, delta 4.20 tok/s.
- Fresh 10-run wrapper check after relaxed paged-metadata tuple propagation
  merged decode into one CUDA graph region:
  MLC 414.23 generated tok/s / 446.09 effective decode tok/s, vLLM 413.30
  generated tok/s with 23.71 s model load/startup, delta -0.93 tok/s.

I tried to refresh the vLLM number again on 2026-05-04 against the raw
`familiar-ai/...` HF repo. The installed conda package failed to load that
checkpoint because its expected parameter names no longer matched the checkpoint
names. The prepared local vLLM-compatible copy in `dist/` is the valid current
comparator.

Important timing caveat: when `MLC_DEFER_CPU_TOKEN_BURST=1` is enabled, the
inner `decode_tokens_per_second` counter can exclude GPU work that is later
paid at the final deferred token commit. Use
`mlc_effective_decode_metrics` from `run_im.py` when comparing decode speed.
The current honest decode speed is roughly 424-435 effective decode tok/s,
depending on synchronization mode and artifact, while generated throughput is
roughly 396-407 tok/s.

A synchronized 5-run check with the current-best artifact and
`--sync-decode-timing` measured 392.58 generated tok/s and 421.33 effective
decode tok/s. With synchronization inserted at the decode subphase boundaries,
the phase attribution was:

- `model_seconds=1.452710`
- `logits_update_seconds=0.000793`
- `probs_seconds=0.021494`
- `sample_seconds=0.024219`
- `device_token_copy_seconds=0.006553`

This confirms that the large unsynchronized `sample_seconds`/deferred-commit
buckets are mostly wait points for previously queued CUDA work. The next
optimization target should remain model-side kernel count, full-attention
backend shape, and state traffic, not a standalone sampler rewrite.

Two follow-up FA2 checks on 2026-05-04 did not beat this current best:

- MLC-layout vLLM-style FA2 backend:
  386.77 generated tok/s, 414.22 effective decode tok/s synchronized.
- Physical vLLM-layout FA2 backend with the fast layout kernel:
  390.62 generated tok/s, 418.38 effective decode tok/s synchronized.
- Physical vLLM-layout FA2 backend with fast layout plus cross-attention graph
  capture refresh:
  392.09 generated tok/s, 420.09 effective decode tok/s synchronized.

A short traced comparison showed why this is not the next winning path yet:
for a 64-token run, the visible `decode` spans were essentially unchanged
between current best and FA2, while the later `update logits` synchronization
span was worse for FA2. The fast attention microkernel is real, but whole-model
decode is still dominated by model/LM-head work and graph/runtime boundaries.

## Successful Changes

- Fused Q/Gate/K/V input projection for Qwen3.5 attention.
- Combined append-KV and cross-attention path for full-attention decode layers.
- Explicit paged-metadata cross-attention path for full-attention decode layers,
  now paired with stable decode-input CUDA graph capture as the default local
  artifact.
- FlashInfer MHA capture-stream fix:
  CUDA graph capture now sets a VM-local thread scope, and FlashInfer MHA uses
  the active capture stream while that scope is set. This fixes the previously
  corrupt direct cross-attention CUDA graph artifact without requiring
  `TVM_FLASHINFER_MHA_USE_CURRENT_STREAM=1`, confirming the stale stream used
  by FlashInfer's MHA wrapper was part of the graph-replay correctness bug.
- Experimental CUDA graph capture of KV-cache append and paged-metadata view
  builtins:
  `TVM_CUDA_GRAPH_CAPTURE_KV_CACHE_APPEND_METADATA=1` lets the graph planner
  keep `attention_kv_cache_append_mha_kv` and
  `attention_kv_cache_get_paged_decode_metadata` inside the full-attention
  graph regions. This reduced `decode_mrope` from 13 to 7 graph regions and
  moved the wrapper result to within 0.14% of vLLM on the 5-run batch-1 image
  benchmark. The longer 10-run comparison is the more conservative current
  baseline: MLC is 410.27 tok/s versus vLLM at 414.48 tok/s, a 1.01% gap.
  In the current graph dump, the only ordinary packed call left outside graph
  capture is `attention_kv_cache_get_query_positions`; appending K/V and
  reading paged metadata are no longer the exposed boundary.
- Experimental relaxed static propagation for paged metadata tuples:
  `TVM_CUDA_GRAPH_RELAXED_KV_CACHE_METADATA_TUPLES=1` treats only
  `attention_kv_cache_get_paged_decode_metadata` result tuples as static enough
  to keep propagating the graph region. This reduces `decode_mrope` from 7 graph
  regions to 1 graph region and moves the 10-run wrapper result to MLC 414.23
  tok/s versus vLLM 413.30 tok/s on the same image grid and generated token
  count.
- Packed GDN decode path with repeated Q/K norm, gate, beta, and V work hoisted
  out of the innermost update loop.
- Chunked GDN prefill enabled while skipping CUDA graph capture for prefill
  functions.
- Direct RNN and convolution state paths enabled.
- cublas single-token decode enabled.
- CPU token transfer/commit deferred across decode bursts.

## Experiments That Did Not Win

- Forcing FlashInfer paged decode for Qwen3.5 full-attention layers:
  about 403-404 tokens/s, slightly below the default. A fresh 5-run check with
  `TVM_FORCE_FLASHINFER_DECODE_KERNEL=1` measured 403.43 tokens/s versus the
  current-best artifact's roughly 404-407 tokens/s range, so the existing
  GQA>=4 paged-prefill heuristic should stay in place for now.
- Delaying Q RMSNorm/MRoPE until after explicit paged metadata lookup:
  this was intended to merge the small Q-normalization graph with the following
  cross-attention graph. The first source-order-only compile still produced 13
  `decode_mrope` CUDA graph regions because TVM moved the pure metadata lookup
  after Q-normalization. A second compile used a metadata-dependent barrier and
  did preserve the metadata ordering, but the final debug dump still showed 13
  CUDA graph regions with six append calls and six metadata calls outside graph
  capture. This is not a winning route unless the CUDA graph partitioner itself
  can capture or otherwise internalize those KV-cache metadata builtins.
- Capturing `attention_kv_cache_get_query_positions` under the same append /
  metadata graph-capture gate:
  not viable in the current rewrite. The compile reached VM codegen and failed
  with `Var storage4 is not defined`, so this was reverted. The working
  append/metadata capture still leaves the single query-position packed call
  outside graph capture. The CUDA graph runtime also explains why a plain
  allowlist is the wrong shape: once a graph is cached, `run_or_capture`
  launches the cached graph and returns the captured state tuple without
  re-invoking the lifted closure. Query positions change every generated token,
  so they must remain a replay input or be produced by a replay-safe device-side
  update. Moving the CPU packed lookup inside capture would risk a stale first
  token position even if the VM storage issue were fixed.
  Moving the lookup out to the C++ serving path and passing query positions as
  a new decode input would be safer, but it is not expected to be a large speed
  win by itself: the serving path would still need to call `GetQueryPositions()`,
  which synchronizes the compute stream with the auxiliary-data copy stream and
  returns the same `q_rope_position_map_view_` prepared during `BeginForward`.
  That signature experiment is useful only if it is paired with a broader plan
  to make query-position data an explicit stable replay input and then merge
  graph regions.
  A diagnostic implementation confirmed this. The compile-time export is gated
  by `MLC_QWEN35_EXPORT_DECODE_MROPE_QUERY_POSITIONS=1`, and the runtime path is
  gated by `MLC_QWEN35_DECODE_MROPE_QUERY_POSITIONS=1`. The opt-in artifact
  compiled and generated coherent Kentucky output, but measured slower on the
  10-run wrapper benchmark:
  `generated_tokens_per_second=403.847`,
  `decode_tokens_per_second=433.596`. The same artifact with the runtime opt-in
  disabled measured `generated_tokens_per_second=407.547`,
  `decode_tokens_per_second=437.541`. Keep this as a diagnostic hook, not as a
  default compile path.
- KV page size 32:
  about 407 tokens/s, no gain over page size 16.
- KV page size 544, inspired by vLLM's Qwen3.5 block sizing:
  about 38 tokens/s. This does not map cleanly to MLC page size.
- MLC top-level `OPT=O3` on the current packed-GDN build:
  about 407 tokens/s, no meaningful gain.
- Decode burst steps 8, 32, and 64:
  no improvement over the default 16 in this wrapper.
- Enabling `MLC_ENABLE_GREEDY_ARGMAX=1`, with and without
  `MLC_FUSED_LM_HEAD_ARGMAX=1`:
  about 391-392 tokens/s end to end. This removes the softmax/sampling kernels,
  but the current engine path still pays enough token commit/synchronization
  overhead that it does not beat the probabilistic sampler path.
- Recompiling the current-best artifact with
  `MLC_EXPERIMENTAL_FAST_ARGMAX_TIR=1`, then running with
  `MLC_ENABLE_GREEDY_ARGMAX=1`:
  249.21 tokens/s end to end on a 5-run 512x512 check. The timing attributed
  2.36 s of 2.45 s decode time to `sample_seconds`, so this standalone
  fast-argmax TIR path is currently a regression and should not be used for the
  benchmark.
- Recompiling with hidden-state exports and fused `get_token_ids` support:
  397.71 tokens/s end to end with `MLC_ENABLE_GREEDY_ARGMAX=1` and
  `MLC_FUSED_LM_HEAD_ARGMAX=1`. The fused token-id kernel is fast in isolation,
  but the end-to-end path regresses because token commit/synchronization still
  dominates the loop.
- Recompiling with `MLC_QWEN35_FUSED_LM_HEAD_ARGMAX=1` and using the serving
  greedy fused-LM-head path directly:
  the default `MLC_QWEN35_LM_HEAD_ARGMAX_BLOCK_M=32` variant measured
  397.83 tokens/s on a 5-run 512x512 check. Increasing the vocab block to
  `BLOCK_M=64` reduced the exported `get_token_ids` temporary memory estimate
  from 18.95 MB to 9.47 MB, but regressed badly to 214.78 tokens/s. This path
  removes probability and sampler kernels, but the current Triton partial-argmax
  formulation is not competitive with the cuBLAS GEMV plus existing greedy
  sampler path. Do not use the fused LM-head artifact as the current baseline.
- Reusing the deferred batch-1 token staging buffers across BatchDecode actions:
  407.03 tokens/s, effectively neutral versus the previous 406.98 tokens/s. It
  is a reasonable allocation cleanup, but not a material throughput win.
- Compiling with `MLC_QWEN35_BATCH_RNN_STORAGE=1` to batch raw RNN storage
  lookup across linear layers:
  381.16 tokens/s, a regression. The raw-access microprofile is much faster
  than copy-style RNN state access, but this graph shape does not improve the
  full decode loop.
- Capturing KV-cache cross-attention calls into the decode CUDA graph:
  the old implicit-metadata variants were not correct. The `decodefullcg`
  variant reached about 415.44 tokens/s but produced corrupted text. The older
  `kvattncg-decodeonly` variant also produced corrupted text and only about
  313.11 tokens/s. Keep `TVM_CUDA_GRAPH_CAPTURE_KV_CACHE_ATTENTION=0`; the
  current valid path instead uses explicit paged metadata plus
  `TVM_CUDA_GRAPH_CAPTURE_KV_CACHE_CROSS_ATTENTION=1` and
  `TVM_CUDA_GRAPH_CAPTURE_KV_CACHE_APPEND_METADATA=1`.
- Capturing function-output storage without KV-cache attention capture:
  correct but not a win. The `outputcapture` artifact measured about
  405.06 tokens/s.
- Capturing stable decode input tensors with
  `TVM_CUDA_GRAPH_CAPTURE_DECODE_INPUTS=1` plus the explicit-paged path:
  correct and now the default local artifact. On the current explicit-paged
  branch it reduces `decode_mrope` graph regions from 15 to 13. The
  synchronized 10-run check measured 399.91 generated tok/s / 433.19 decode
  tok/s, while deferred-burst serving measured 404.75 generated tok/s /
  435.31 effective decode tok/s. This is a small but real current-best update,
  not the full vLLM-class gap closer.
- Splitting full-attention decode into `append_mha_kv` plus captured
  `cross_attention`:
  fast but incorrect. The `separatekv-crosscg-decodeinputcg` artifact measured
  about 420.40 tokens/s, but generated corrupted text. This strongly suggests
  the cross-attention runtime call reads dynamic KV-cache auxiliary data that is
  not represented as a safe CUDA graph replay input. Do not capture
  `vm.builtin.attention_kv_cache_cross_attention` until that metadata dependency
  is modeled explicitly.
- Refreshing direct decode cross-attention capture on the current
  explicit-metadata path with
  `TVM_CUDA_GRAPH_CAPTURE_KV_CACHE_CROSS_ATTENTION=1`:
  originally fast but incorrect. The artifact measured 406.42 generated tok/s
  and 429.80 effective decode tok/s synchronized, but output was corrupted.
  Adding `TVM_FLASHINFER_FIXED_SPLIT_SIZE=64` did not fix correctness; it
  measured 406.02 generated tok/s and 429.14 effective decode tok/s with the
  same corruption pattern. Forcing the FlashInfer decode backend with
  `TVM_FORCE_FLASHINFER_DECODE_KERNEL=1` also did not fix correctness; it
  measured 406.02 generated tok/s and 429.20 effective decode tok/s with the
  same corruption. Rebuilding the runtime with a capture-stream scope for CUDA
  graph capture fixed the refreshed `explicitcrosscg` artifact: it produced
  coherent Kentucky Derby output without a manual stream environment variable.
  Before making the behavior automatic, the diagnostic
  `TVM_FLASHINFER_MHA_USE_CURRENT_STREAM=1` override measured 405.93 generated
  tok/s / 436.59 effective decode tok/s in the normal 5-run serving benchmark,
  and 395.67 generated tok/s / 424.17 effective decode tok/s with synchronized
  decode timing. The automatic path preserves the same correctness fix and
  keeps the diagnosis focused on FlashInfer's stream/graph replay contract
  rather than only on page metadata tensor explicitness or paged-prefill versus
  decode backend choice.
- Combining the now-correct current-stream cross-attention graph path with
  `TVM_CUDA_GRAPH_CAPTURE_FUNC_OUTPUTS=1`:
  coherent output and a tiny gain. The new artifact
  `dist/libs/logos-multitask-qwen3.5-2026-05-03-best-q0f16-ctx2048-pc320-explicitcrosscg-currentstream-outputcapture-cuda.so`
  measured 406.68 generated tok/s / 437.17 effective decode tok/s in a
  standalone normal 5-run serving benchmark after the capture-stream behavior
  was made automatic. The two-engine wrapper measured 406.68 generated tok/s
  for MLC versus 414.10 tok/s for vLLM, leaving a 7.42 tok/s gap. Treat this as
  the current best candidate, but the improvement over the non-output-capture
  cross-attention graph artifact is within normal run-to-run noise.
- Forcing the plain paged-KV aux-data manager with
  `TVM_PAGED_KV_CACHE_FORCE_PLAIN_AUX=1`, then repeating the split
  append/captured-cross-attention experiment:
  still incorrect. The `separatekv-crosscg-plainaux-decodeinputcg` artifact
  measured about 420.54 tokens/s but generated corrupted text. This rules out
  unstable aux view addresses as the only bug. Capturing
  `attention_kv_cache_cross_attention` also skips CPU-side attention planning
  and synchronization logic that must run each decode token.
- Repeating the same split append/captured-cross-attention experiment with
  `flashinfer=0`:
  correct but slower. The `separatekv-crosscg-tir-decodeinputcg` artifact
  measured 367.70 tokens/s and produced a sane Kentucky Derby answer. This
  isolates the correctness problem to the FlashInfer-backed paged-attention
  graph replay contract. The graph transformation and split append/cross shape
  can be correct when the attention backend does not need dynamic launch
  planning hidden behind the packed call.
- Enabling FlashInfer's existing CUDA graph planning knob,
  `TVM_FLASHINFER_ENABLE_CUDA_GRAPH_PLAN=1`, on the split/captured-cross
  FlashInfer artifact:
  still incorrect. It measured about 420.46 tokens/s and produced the same
  corrupted text pattern. The current TVM FlashInfer plan flag is not enough to
  make MLC's packed `CrossAttention` method graph-safe.
- Enabling `TVM_FLASHINFER_ENABLE_CUDA_GRAPH_PLAN=1` on the current correct
  combined-cross artifact:
  correct but no improvement. A short 5-run check measured 406.77 tokens/s,
  effectively the same as the then-current 407-408 tok/s baseline.
- Rechecking the combined append/cross output-capture artifact with synchronized
  decode timing on 2026-05-04:
  correct but still below the explicit-metadata path under the same timing mode.
  It measured 393.60 generated tok/s and 421.71 effective decode tok/s, versus
  396.00 generated tok/s and 424.65 effective decode tok/s for the current
  explicit-metadata artifact. Reducing graph count through the combined builtin
  does not offset its slower model-side path.
- Forcing the plain aux manager and returning fixed-capacity page-indices
  tensors with `TVM_PAGED_KV_CACHE_STATIC_AUX_SHAPES=1`, together with
  FlashInfer graph planning:
  still incorrect on the captured-cross FlashInfer artifact. It measured about
  419.77 tokens/s and produced the same corrupted text. A broader static-shape
  attempt that also made RoPE position maps fixed-size failed during prefill
  because RoPE kernels require the position-map shape to match the current query
  length. This rules out page-indices shape stability as the sole missing piece.
- Extending `TVM_PAGED_KV_CACHE_STATIC_AUX_SHAPES=1` to also return
  fixed-capacity `page_indptr`, `length_info`, and `k_rope_pos_offset` tensors
  in the plain aux manager, while keeping query RoPE exact-size:
  still incorrect on the captured-cross FlashInfer artifact. With FlashInfer
  graph planning, fixed split size 16, split-KV disabled, and stable decode
  embeddings, the short run still produced corrupted text at about 404 tok/s.
  This rules out simple aux tensor shape stability as the missing contract.
- Exposing FlashInfer paged/ragged prefill plan knobs
  `TVM_FLASHINFER_FIXED_SPLIT_SIZE` and `TVM_FLASHINFER_DISABLE_SPLIT_KV`,
  then running the captured-cross artifact with
  `TVM_FLASHINFER_ENABLE_CUDA_GRAPH_PLAN=1`,
  `TVM_FLASHINFER_FIXED_SPLIT_SIZE=16`,
  `TVM_FLASHINFER_DISABLE_SPLIT_KV=1`, and stable decode embeddings:
  still incorrect. A short 64-token run remained corrupted, despite measuring
  around 405 tok/s. The same knobs on the known-good combined-cross artifact
  preserved correctness but slowed the short run from about 381 tok/s to about
  362 tok/s. This rules out FlashInfer's exposed fixed-split / split-KV toggles
  as the missing graph-replay contract.
- Adding `TVM_FLASHINFER_LOG_PLAN_INFO=1` and logging FlashInfer plan vectors:
  on the current combined-cross artifact, the decode-time paged-prefill plan
  vector stayed constant across a 40-token run:
  `[3,1,0,16,0,16,32,64,48,56,0,393216,80,0,1]`. The image prefill ragged
  plan was `[54,276,0,64,0,224,448,688,672,680,0,28311552,1808,0,1]`. This
  weakens the simple stale-`plan_info_vec` theory; the captured-cross corruption
  is more likely an internal FlashInfer CUDA-graph replay limitation or another
  hidden pointer/state dependency.
- Forcing `TVM_FORCE_FLASHINFER_DECODE_KERNEL=1` on the current known-good
  combined-cross artifact:
  correct but not faster in a short 64-token check, about 379 tok/s versus about
  381 tok/s for the default paged-prefill route. This confirms the existing
  FlashInfer decode-kernel route is not an easy local win for this GQA=4 model.
  A sequential synchronized timing check reached 424.5 decode tok/s for the
  default paged-prefill route versus 420.8 decode tok/s for the forced decode
  kernel route, so the conclusion holds when async timing distortion is removed.
- Letting FlashInfer consume MLC K/V pages in NHD logical layout
  `[block, page, kv_head, dim]` with `TVM_FLASHINFER_KV_LAYOUT_NHD=1`:
  correct but neutral. A 20-run batch-1 MLC-only check measured 407.15 tok/s,
  effectively the same as the then-current 407-408 tok/s baseline. Keep it as a
  useful layout-control knob for FA2 alignment experiments, not as a current win.
- Adding a narrow direct paged-decode CUDA backend registered as
  `tvm.contrib.flash_attn.fa2_paged_decode`:
  correct for the current batch-1 fp16 test case, but not a performance win yet.
  With `MLC_QWEN35_FA2_PAGED_DECODE_ATTENTION=1`, the prototype generated sane
  Kentucky Derby output. After caching Q and token page offsets in shared memory
  inside the prototype kernel, a short 5-run image benchmark measured about
  364.2 generated tok/s / 388.2 decode tok/s, up from the earlier 338.7
  generated tok/s / 359.0 decode tok/s. A sequential direct-VM decode-body check
  measured 394.3 tok/s for the FA2-dispatch artifact versus 434.8 tok/s for the
  current best artifact. This backend is a correctness bridge, not a real
  FA2-class kernel. It uses a simple one-block-per-query-head softmax over paged
  context and is expected to lose to the current FlashInfer-backed best.
- Capturing the tensor-only FA2 paged decode call inside `decode_mrope` with
  `TVM_CUDA_GRAPH_CAPTURE_KV_CACHE_CROSS_ATTENTION=1`:
  correct only when the runtime backend knob
  `TVM_FA2_PAGED_DECODE_VLLM_STYLE=1` is also set, but still slower than the
  current best. The compiled artifact
  `dist/libs/logos-multitask-qwen3.5-2026-05-03-best-q0f16-ctx2048-pc320-fa2vllmstyle-crosscg-cuda.so`
  produced sane Kentucky Derby output and measured 392.18 generated tok/s /
  424.17 decode tok/s in a synchronized 10-run check. Without the runtime
  `TVM_FA2_PAGED_DECODE_VLLM_STYLE=1` knob, the same artifact hit a CUDA illegal
  memory access during sampling. This is a useful graph-shape experiment but not
  a candidate baseline.
- Grouping all Q heads that share a KV head into one prototype GQA decode block:
  correct but slower. With `TVM_FA2_PAGED_DECODE_GROUPED_GQA=1`, the direct
  backend microbenchmark at context 320 regressed to about 230.8 us/call versus
  about 56.0 us/call for the default per-Q-head prototype. The grouped kernel is
  left behind a feature flag as a diagnostic, but it should not be used for the
  current benchmark.
- Adding a half2-specialized variant of the prototype FA2 paged decode kernel:
  correct but neutral-to-slower. At context 320, the direct backend
  microbenchmark measured about 56.53 us/call with `TVM_FA2_PAGED_DECODE_HALF2=1`
  versus about 56.04 us/call for the default scalar path. The half2 path is left
  behind an opt-in feature flag for diagnostics, but the default remains the
  scalar kernel.
- Adding a `TVM_FA2_PAGED_DECODE_THREADS` diagnostic knob for the prototype FA2
  paged decode backend:
  correct, but it confirms the current 256-thread launch is the right local
  shape. At context 320, a 1000-run microbenchmark measured about 115.6 us/call
  with 64 threads, 74.0 us/call with 128 threads, 53.4 us/call with 256 threads,
  and 54.2 us/call with 512 threads. The default remains 256 threads.
- Disabling FlashInfer split-KV on the current best artifact with
  `TVM_FLASHINFER_DISABLE_SPLIT_KV=1`:
  correct but slightly slower. A serial 20-run batch-1 MLC-only check measured
  405.40 tok/s versus 407.42 tok/s for the adjacent baseline.
- Forcing a FlashInfer fixed split size of 64 on the current best artifact with
  `TVM_FLASHINFER_FIXED_SPLIT_SIZE=64`:
  correct but slightly slower. A serial 20-run batch-1 MLC-only check measured
  404.89 tok/s. The existing exposed split-KV knobs are therefore not an easy
  path to removing the per-layer FlashInfer merge overhead.
- Enabling TVM's bundled contrib vLLM kernels with `USE_VLLM=ON`:
  now builds through `tvm_runtime_objs` after fixing the stale TVM FFI wrappers.
  It is still not the main target because those kernels expect vLLM's older
  packed K/V cache layout and derive cache strides from shape, so MLC would need
  per-token K/V repacking to use them.
- Reusing CUDA events inside TVM's `SyncStreamFromTo` with
  `TVM_CUDA_REUSE_SYNC_EVENTS=1`:
  correct but neutral. After rebuilding `build-mlc-env`, a synchronized
  10-run current-best check measured 398.23 generated tok/s / 431.71 decode
  tok/s with event reuse versus 397.91 generated tok/s / 431.67 decode tok/s
  without it. This confirms the Nsight event churn is visible but not the
  throughput-limiting gap for this benchmark.
- Capturing decode function output storage with
  `TVM_CUDA_GRAPH_CAPTURE_FUNC_OUTPUTS=1` on top of the current explicit-paged
  stable-input branch:
  correct and it removes the final LM-head cuBLAS packed call from the
  uncaptured `decode_mrope` call list, but it is not a material win. The
  synchronized 10-run check measured 399.19 generated tok/s / 432.10 decode
  tok/s, and deferred-burst serving measured 404.88 generated tok/s /
  435.09 effective decode tok/s. This is within run-to-run noise of the simpler
  current default, so the default artifact remains
  `explicit-paged-stabledecodeinput`.
- Capturing RNN-state lookup packed calls with
  `TVM_CUDA_GRAPH_CAPTURE_RNN_STATE_LOOKUPS=1` on the current explicit-paged
  stable-input branch:
  correct but neutral. The synchronized 5-run check measured 393.96 generated
  tok/s / 422.15 effective decode tok/s, compared with the adjacent
  synchronized control at 392.58 generated tok/s / 421.33 effective decode
  tok/s. The phase buckets were nearly unchanged
  (`model_seconds=1.450378`, `probs_seconds=0.020906`,
  `sample_seconds=0.024652`). This is useful evidence that RNN lookup capture
  is not the remaining gap; keep it as an optional diagnostic rather than the
  default.
- Rechecking batch explicit-metadata cross-attention capture with
  `TVM_CUDA_GRAPH_CAPTURE_BATCH_EXPLICIT_KV_CACHE_CROSS_ATTENTION=1`:
  coherent output, but still neutral. The refreshed artifact measured
  393.11 generated tok/s and 421.50 effective decode tok/s synchronized, versus
  the adjacent current-best control at 392.58 generated tok/s and 421.33
  effective decode tok/s. A direct decode-body Nsight capture showed the same
  graph structure as current best: 1664 `cudaGraphLaunch` calls for 128
  decode-body calls, or exactly 13 graph launches/token. So this flag does not
  currently merge the remaining graph partitions.

## Phase Profile

Use the direct VM phase profiler with the runtime wrapper:

```bash
./profile_qwen35_vm_phases.sh \
  --image kentucky.png \
  --fit-image-size 512 \
  --phase decode-body \
  --runs 128 \
  --warmup-runs 4 \
  --reuse-input-tensors
```

Current synchronized direct-VM measurements:

- full per-token decode: about 429 tokens/s
- decode body inside one begin/end-forward window: about 435 tokens/s
- decode body for the relaxed paged-metadata tuple one-graph artifact:
  432.59 tokens/s
- decode body for the FA2-dispatch prototype artifact: about 394 tokens/s
- decode body for the physical vLLM-layout FA2 fast/cross-graph artifact:
  425.17 tokens/s

A fresh Nsight Systems capture of `decode-body` on the relaxed paged-metadata
tuple one-graph artifact produced:

```text
128 x cudaGraphLaunch                       3.80 ms total, 29.65 us/call
128 x cuLaunchKernel                        0.92 ms total,  7.18 us/call
128 x mrope/query-position kernel           0.13 ms total,  0.98 us/call
256 x cudaMemcpyAsync                       1.28 ms total,  4.98 us/call
```

This confirms the intended runtime shape: one CUDA graph launch per
decode-body call, with the LM head, six FlashInfer attention calls,
append/metadata work, and most elementwise kernels hidden inside the captured
graph. The compare helper now falls back to `cudaGraphLaunch` count for token
inference when the dominant GEMV kernel is no longer visible because it is
inside a graph.

A previous Nsight Systems capture of `decode-body` on the current-best
append/metadata artifact before relaxed metadata propagation
measured 424.21 tokens/s and showed the clean per-token kernel shape:

```text
128 x cuBLAS GEMV LM-head kernel                  66.51 ms total, 519.59 us/call
768 x FlashInfer paged attention kernel           5.36 ms total,   6.97 us/call
768 x FlashInfer merge-states kernel              1.27 ms total,   1.65 us/call
768 x tir_kv_cache_transpose_append_kernel        0.97 ms total,   1.26 us/call
128 x small mrope/add/cast kernel                 0.13 ms total,   0.98 us/call
```

The LM head is the largest single kernel, but it is not the biggest
MLC-vs-vLLM gap because vLLM's trace also has a similar GEMV-class LM-head
kernel. The more actionable compile/runtime gap is graph structure. Earlier
current-best artifacts issued about 13 CUDA graph launches per decode-body call,
while the saved vLLM trace was close to one graph launch per generated token.
After append/metadata capture, the current `decode_mrope` Relax dump has 7 CUDA
graph regions and only one remaining non-graph packed call:
`attention_kv_cache_get_query_positions`. Reducing those remaining graph
regions, or making query-position metadata graph-safe without breaking VM
storage planning, is now a higher-priority compiler target than further tuning
the standalone FA2 kernel.

A fresh Nsight Systems capture of `decode-body` on the current-stream fixed
cross-attention graph candidate measured 427.84 tokens/s. The CUDA graph launch
count was still 1664 launches for 128 decode-body calls, or 13 launches/token,
but the six FlashInfer attention and merge kernel pairs per token disappeared
from the visible standalone kernel list:

```text
128 x cuBLAS GEMV LM-head kernel                  66.50 ms total, 519.50 us/call
768 x tir_kv_cache_transpose_append_kernel        0.97 ms total,   1.26 us/call
128 x small mrope/add/cast kernel                 0.13 ms total,   0.98 us/call
```

Compared with the old current-best decode-body profile, this removes about
6.6 ms of visible FlashInfer attention/merge work across 128 tokens and
reduces uncaptured `cudaLaunchKernel` calls from 1664 to 128. The later
append/metadata-capture artifact goes further by reducing `decode_mrope` to 7
CUDA graph regions. The next compiler target is no longer a simple allowlist of
append or paged-metadata builtins; it is merging the remaining graph regions and
handling the query-position / LM-head output boundaries safely.

A follow-up direct-VM Nsight capture of the combined cross-attention graph plus
function-output capture artifact measured 427.33 tokens/s. It removed the
visible standalone LM-head `cudaLaunchKernel` from the top kernel list, but
still issued 1664 `cudaGraphLaunch` calls for 128 decode-body calls, or 13
launches/token:

```text
768 x tir_kv_cache_transpose_append_kernel        0.97 ms total,   1.26 us/call
128 x small mrope/add/cast kernel                 0.13 ms total,   0.98 us/call
```

This confirms function-output capture can hide the LM-head launch inside graph
replay, but by itself it does not merge graph partitions or improve the direct
decode-body microprofile. The remaining compiler target is still reducing the
number of decode graph regions.

A matching direct-VM Nsight capture of the graph-safe physical vLLM-layout FA2
fast/cross-graph artifact measured 425.17 tokens/s. The visible uncaptured
kernel list collapsed to the LM-head GEMV, six KV-append kernels/token, and one
small MRoPE/add/cast kernel/token:

```text
128 x cuBLAS GEMV LM-head kernel                  66.51 ms total, 519.57 us/call
768 x tir_kv_cache_transpose_append_vllm_layout   1.14 ms total,   1.49 us/call
128 x small mrope/add/cast kernel                 0.13 ms total,   0.98 us/call
```

That profile also reduced visible graph launches from 13/token to 7/token.
However, the serving benchmark for the same artifact remains slightly below the
current best. The direct `decode-body` phase intentionally replays calls inside
one begin/end-forward window with reused input tensors, so it is a graph-shape
microprofile rather than a valid generation benchmark. The useful conclusion is
that FA2 can make the attention boundary graph-safe, but the remaining serving
gap is now outside the standalone FA2 microkernel: LM-head/output boundary,
graph partition count, and token commit/synchronization still dominate the
end-to-end result.

Use the Nsight comparison helper for current MLC-vLLM traces:

```bash
./scripts/compare_nsys_decode.py \
  --left nsys-qwen35-currentbest-latest.sqlite \
  --left-name MLC \
  --right nsys-vllm-qwen35-current.sqlite \
  --right-name vLLM
```

The helper now infers decode-token counts from the dominant GEMV-class decode
kernel by default. The existing traces are not perfectly isolated identical
decode windows, so treat this as directional rather than a final benchmark. The
useful signal is still clear: vLLM's trace is dominated by one GEMV-class kernel
per token, while MLC shows extra full-attention FlashInfer calls, KV-cache
append/RoPE kernels, CUDA graph launches, memcpy, and synchronization. That
supports making the next compiler/runtime target explicit-metadata graph-safe
attention or reducing the remaining per-token launch/copy/sync overhead, not
quantization.

Use the vLLM FlashAttention microbenchmark to measure the standalone paged
attention ceiling for Qwen3.5-like shapes:

```bash
cd /tmp
/home/cwong/Projects/miniconda/envs/vllm/bin/python \
  /home/cwong/Projects/familiar/mlc-llm/scripts/benchmark_vllm_flash_attn_paged.py \
  --context-len 320 \
  --benchmark-runs 1000 \
  --warmup-runs 50
```

On the current machine this measured about 34.54 us per FA2 paged-attention call
for `q_heads=8`, `kv_heads=2`, `head_dim=256`, `block_size=16`, and
`context_len=320`. This is faster than the simple custom FA2 prototype in MLC,
which measured about 56.00 us per call on the same synthetic shape with
`benchmark_fa2_paged_decode_backend.py`. It does not by itself prove that
swapping kernels beats the current FlashInfer-backed MLC path. The value is that
the vLLM backend demonstrates the right explicit metadata contract with
`block_table`, `seqused_k`, and `cu_seqlens_q`.

## vLLM Source Alignment

The local vLLM source at `/home/cwong/Projects/familiar/vllm` confirms that the
main architectural difference is not startup or quantization. The decode path is
more explicit about dynamic attention metadata:

- `vllm/v1/attention/backends/flash_attn.py` passes `cu_seqlens_q`,
  `seqused_k`, `block_table`, and optional `scheduler_metadata` directly into
  `flash_attn_varlen_func`. For CUDA graphs, vLLM preallocates scheduler
  metadata storage and constrains `num_splits` so graph replay has stable
  buffers.
- The same file documents that FA2 CUDA graph support is not universal; it is
  treated as uniform-batch compatible, while FA3 has broader full-graph support.
  This matches our observation that simply turning on FlashInfer's graph-plan
  flag in MLC did not make captured FlashInfer cross-attention correct.
- `vllm/model_executor/layers/mamba/gdn_linear_attn.py` calls
  `fused_recurrent_gated_delta_rule_packed_decode` for the Qwen3.5 linear
  attention layers. MLC has already copied the key idea here with packed GDN
  decode, which is why the remaining gap is now mostly full-attention/engine
  overhead rather than the original linear-attention bottleneck.

The MLC side still differs in the FlashInfer bridge. In
`3rdparty/tvm/src/runtime/vm/attn_backend.h`, `FlashInferPagedPrefillFunc` and
`FlashInferPagedDecodeFunc` compute a host-side `plan_info_vec` in
`BeginForward`, store it in `cached_buffers_`, and then consume it implicitly in
`MHA`. The Python Relax frontend installs these as opaque extern tuples:
`("flashinfer", "batch_prefill_paged_run", "batch_prefill_plan")` and
`("flashinfer", "batch_decode_run", "batch_decode_plan")`. That means a graph
capturing `attention_kv_cache_cross_attention` does not see all of the replay
inputs that vLLM exposes as tensors.

Recommendation: do not spend more time on the current
`TVM_FLASHINFER_ENABLE_CUDA_GRAPH_PLAN` knob by itself. The next compiler/runtime
patch should make the FlashInfer decode/cross-attention metadata explicit in the
VM call contract, or keep the custom FA2 bridge and replace its prototype kernel
with a real vLLM/FA2-class paged decode kernel. A small Python wrapper around the
current FlashInfer object is unlikely to fix correctness, because the hidden
state is below Python in the C++ backend object.

Installing upstream `flash-attn` into the `mlc` conda environment is not the next
step by itself. The working vLLM environment exposes FlashAttention through
`vllm.vllm_flash_attn`, which is bundled with the vLLM wheel, while the MLC env
already has FlashInfer and TVM's bundled `libflash_attn.so`. Pulling vLLM into
the MLC env risks changing Torch/FlashInfer versions without giving MLC a C++/VM
call path. Keep the envs separate unless we intentionally build a throwaway
Python prototype.

If we do want to run direct Python FlashInfer JIT microbenchmarks in the `mlc`
env, install `ninja` first. A direct `flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper`
microbenchmark failed because the env could not find the `ninja` executable.
- decode to last hidden state on the fused-LM-head artifact: about
  555 tokens/s
- fused `get_token_ids` on a cached hidden state: about 1897 calls/s

The current compiled decode body is therefore already close to the latest vLLM
end-to-end benchmark. The remaining end-to-end gap is mostly serving-loop,
sampling, and token-commit overhead rather than the synchronized decode VM body
alone. The default current-best artifact does not expose
`decode_mrope_to_last_hidden_states`; the separate fused-LM-head artifact does,
and confirms that hidden decode plus token selection is not enough by itself to
win unless the serving-loop commit path is also improved.

## Current Interpretation

The GDN decode path is no longer the main gap. Nsight and benchmark evidence
point to the full-attention backend shape and runtime scheduling/commit path as
the remaining areas.

Generate-only Nsight traces show MLC still launches far more work per token than
vLLM. The current MLC trace has about 41.6 kernel calls/token and about 8.9
CUDA graph launches/token, while the vLLM trace has about 29.2 kernel
calls/token and about 1.2 CUDA graph launches/token. MLC's top kernels still
include one large LM-head GEMV, GDN kernels, FlashInfer paged-attention pieces,
RNN state get/set kernels, and sampling kernels. That points to reducing
per-token launch count and state traffic, not another image preprocessing or
quantization change.

The serving-loop timing now splits deferred token commit into GPU copy/sync wait
and CPU append work. On the current 20-run benchmark:

- `deferred_commit_seconds`: 5.397 s
- `deferred_commit_copy_sync_seconds`: 5.393 s
- `deferred_commit_cpu_seconds`: 0.0045 s

So the large deferred-commit bucket should be read as "the final synchronization
point that waits for accumulated async decode work", not as expensive CPU token
bookkeeping. The useful target is still fewer/faster decode kernels before that
sync, especially full-attention and state traffic.

vLLM uses a FlashAttention varlen path that accepts a `block_table` and reads
paged KV directly. TVM's bundled `flash_attn::flash_attention_var_len_forward`
exists, but it accepts contiguous K/V plus cumulative sequence lengths; it does
not accept a paged-KV block table. MLC's current attention backend kinds are TIR
and FlashInfer, so a vLLM-like path is a new backend/storage integration rather
than a small compile flag. For Qwen3.5 batch-1 with GQA ratio 4, MLC's paged KV
cache routes full-attention cross attention through FlashInfer's paged prefill
kernel by default, not the FlashInfer decode kernel, unless
`TVM_FORCE_FLASHINFER_DECODE_KERNEL` is set.

The local FA2 wrapper ABI in vLLM is concrete enough to target:

- `q`: `[total_q, num_q_heads, head_dim]`
- `k` / `v`: paged cache tensors accepted by the vLLM FA2 extension
- `out`: `[total_q, num_q_heads, head_dim]`
- `cu_seqlens_q`: `[batch + 1]`, int32
- `seqused_k`: `[batch]`, int32
- `block_table`: `[batch, max_blocks_per_seq]`, int32
- scalar launch arguments: `max_seqlen_q`, `max_seqlen_k`, `softmax_scale`,
  `causal`, `window_left`, `window_right`, `softcap`, `num_splits`

For the local FA2 path, `scheduler_metadata`, descale tensors, sinks, and
`num_splits > 1` are explicitly unsupported. This is useful: the batch-1 MLC
target can be simpler than the Hopper FA3 path. It needs an explicit paged
block table and sequence lengths; it does not need the FA3 AOT scheduler path
to match the current vLLM result on RTX 4090.

Local vLLM source confirms the two most relevant decode ideas. On this RTX
4090 / `sm_89` machine, the working vLLM env reports `get_flash_attn_version()`
as `2`, so the current apples-to-apples competitor is FA2 paged varlen
attention rather than FA3 AOT scheduler metadata. FA3 scheduler metadata is
still relevant for Hopper, but it should not be treated as the immediate source
of the local vLLM result.

- Full-attention layers call `flash_attn_varlen_func(..., block_table=...)` in
  `/home/cwong/Projects/familiar/vllm/vllm/v1/attention/backends/flash_attn.py`
  around lines 817-839. For FA2 on this machine, the important inputs are the
  explicit `block_table`, `seqused_k`, and fixed graph-compatible decode shape.
  The same metadata builder has FA3-only scheduler metadata support, but that is
  not active on `sm_89`.
- vLLM's FA2 host wrapper has a batch-1/GQA decode specialization before launch:
  when `max_seqlen_q == 1`, `num_q_heads > num_kv_heads`, no local window, no
  dropout, and no alibi, it reshapes query from `[batch, 1, num_q_heads,
  head_dim]` into `[batch * groups, num_kv_heads, head_dim]`. For Qwen3.5 0.8B
  this turns 8 query heads and 2 KV heads into 4 query rows with 2 heads. This
  likely improves the FA2 split-KV decode shape and is a compiler/runtime
  optimization MLC can copy without changing model accuracy.
- vLLM's batch-1 token selection is also leaner. The V1 GPU sampler includes a
  Triton Gumbel/argmax path in
  `/home/cwong/Projects/familiar/vllm/vllm/v1/worker/gpu/sample/gumbel.py`
  that block-reduces logits in 1024-wide chunks, then reduces the per-block
  maxima. MLC's current serving path still routes many default requests through
  probability/top-p sampler functions and token-result synchronization. For
  greedy batch-1 benchmarking, a compiler-visible LM-head-plus-token-selection
  path remains worth revisiting, but only if it also avoids the surrounding
  commit/sync overhead that made the previous fused-token-id experiment regress.
- The concrete vLLM C++ entry is
  `/home/cwong/Projects/familiar/vllm/.deps/vllm-flash-attn-src/csrc/flash_attn/flash_api.cpp::mha_fwd_kvcache`.
  Its paged path checks `block_table`, derives `num_blocks`,
  `page_block_size`, `max_num_blocks_per_seq`, sets
  `params.block_table`, `params.block_table_batch_stride`, and forces the split
  kernel for paged KV.
- The FA2 paged path accepts K/V logically as `[num_blocks, page_block_size,
  num_kv_heads, head_dim]`. It requires only last-dim contiguity, then passes
  `k_batch_stride`, `k_row_stride`, and `k_head_stride` to the kernel. MLC stores
  MHA KV pages as `[num_pages, 2, num_kv_heads, page_size, head_dim]`, so a good
  MLC integration should pass the base pointer plus explicit K/V strides instead
  of materializing/repacking K/V every token. MLC page size 16 satisfies FA2's
  CUDA page-size divisibility check.
- TVM's bundled `3rdparty/tvm/3rdparty/libflash_attn` does not expose this
  paged-KV contract. Its `Flash_fwd_params` has cumulative sequence-length
  fields such as `cu_seqlens_q` / `cu_seqlens_k`, but no `block_table`,
  `page_block_size`, or paged row-offset logic. So the vLLM idea cannot be
  enabled by switching MLC to the existing bundled FA2 varlen helper; MLC needs
  a new paged FA2 wrapper or an adapted vendored subset.
- The MLC runtime now has a batch-1 bridge API for this metadata:
  `vm.builtin.attention_kv_cache_get_fa2_paged_decode_metadata`, exposed in
  Python as `PagedKVCache.get_fa2_paged_decode_metadata`. The compiler-visible
  frontend form is `PagedKVCache.get_fa2_paged_decode_metadata_tensors`, which
  returns six typed Relax tensor expressions instead of an opaque object. It now
  returns logical K/V views shaped as `[num_pages, page_size, num_kv_heads,
  head_dim]` plus the block-table metadata. The focused runtime regression now
  passes against the rebuilt `libtvm.so` and verifies the expected FA2-style
  layout and metadata after a 33-token prefill plus 1-token decode: K/V strides
  `[2 * H * N * D, D, N * D, 1]`, `(1, 3)` block table, `seqused_k=[34]`,
  `cu_seqlens_q=[0, 1]`, and `q_rope_pos=[33]`. The frontend regression in
  `tests/python/model/test_kv_cache.py` verifies that the typed call carries
  two float16 rank-4 K/V tensors plus int32 rank-2/rank-1 metadata tensors.
- `PagedKVCache.fa2_paged_decode_attention` now emits the explicit future
  extern call `vm.builtin.attention_kv_cache_fa2_paged_decode` with Q, K pages,
  V pages, block table, sequence length tensors, query RoPE positions, and
  scalar launch parameters as ordinary arguments. Qwen3.5 can be made to lower
  through this path with `MLC_QWEN35_FA2_PAGED_DECODE_ATTENTION=1`; it remains
  off by default because the current CUDA backend is only a narrow correctness
  prototype. The VM symbol now dispatches to
  `tvm.contrib.flash_attn.fa2_paged_decode` when that backend is registered, and
  otherwise throws a precise error. The focused frontend test verifies the
  lowered call. The focused runtime tests now verify both the dispatcher ABI with
  a fake backend and the real CUDA backend against a NumPy reference for batch-1
  fp16 paged decode.
- A real Qwen3.5 compile with `MLC_QWEN35_FA2_PAGED_DECODE_ATTENTION=1` now
  succeeds through library generation:
  `dist/libs/logos-multitask-qwen3.5-2026-05-03-best-q0f16-ctx2048-pc320-fa2dispatch-compilecheck-cuda.so`.
  Running that library against the locally rebuilt TVM runtime now reaches the
  prototype backend and produces coherent image output. The benchmark result is
  slower than the current best, so the next step is replacing the prototype
  kernel with a real FA2/vLLM-style paged decode implementation rather than
  enabling this path by default.
- The direct backend microbenchmark is:

  ```bash
  TVM_LIBRARY_PATH=$PWD/build-tvm-full \
  LD_LIBRARY_PATH=$PWD/build-tvm-full:$PWD/build-mlc-env:$PWD/build-mlc-env/tvm:$LD_LIBRARY_PATH \
  PYTHONPATH=$PWD/3rdparty/tvm/python:$PWD/python \
  /home/cwong/Projects/miniconda/envs/mlc/bin/python \
    benchmark_fa2_paged_decode_backend.py \
    --context-len 320 \
    --benchmark-runs 2000 \
    --warmup-runs 50
  ```

  Current prototype backend timing on Qwen3.5-like shape
  `(8 Q heads, 2 KV heads, head_dim 256, page_size 16)` is about 23.7 us at
  context 128, 56.0 us at context 320, 83.1 us at context 512, and 157.5 us at
  context 1024 after caching Q and token page offsets in shared memory. The
  context-320 timing was about 92.1 us before those changes. At the image prompt
  length, six
  full-attention layers therefore contribute roughly 0.34 ms/token in this
  prototype path. That is meaningful,
  but it is not the full end-to-end slowdown; launch count, sampling, and
  serving-loop overhead remain material.
- GDN non-spec decode uses `causal_conv1d_update` followed by
  `fused_recurrent_gated_delta_rule_packed_decode` in
  `/home/cwong/Projects/familiar/vllm/vllm/model_executor/layers/mamba/gdn_linear_attn.py`
  around lines 1041-1062.
- TVM's legacy `src/runtime/contrib/vllm` path was stale against the current TVM
  FFI. The wrapper compile blockers are fixed: returned cache arrays now use
  `ffi::Array<Tensor>`, and the registered attention entry points accept `Tensor`
  handles before passing `DLTensor*` to the existing CUDA launchers. Verified by
  temporarily configuring `USE_VLLM=ON` and building `tvm_runtime_objs`, then
  restoring the main build tree to `USE_VLLM=OFF`. This is a baseline/debug path,
  not the final FA2 paged-varlen backend.

MLC's batch-1 serving path uses `decode_mrope`, not `batch_decode_mrope`:

- `cpp/serve/model.cc` dispatches to `single_batch_decode_mrope_func_` when
  `seq_ids.size() == 1`.
- `cpp/serve/function_table.cc` maps that function to the VM function
  `decode_mrope`.
- `python/mlc_llm/interface/compile.py` currently adds symbolic CUDA graph
  hints for `batch_decode_mrope`, but not for `decode_mrope`. Fixed-shape
  `decode_mrope` still receives CUDA graph rewriting, but the shape of the
  resulting graph matters for batch-1 performance.

Existing graph dumps show three useful cases:

- `build-qwen35-kvattncg-decodeonly-debug/debug-phase6.py`: `decode_mrope`
  has three CUDA graph regions plus several external packed calls.
- `build-qwen35-batchstorages-debug/debug-phase6.py`: `decode_mrope` has 61
  CUDA graph regions and many RNN/KV packed calls outside those regions. This
  matches the Nsight symptom of many graph launches per token.
- `build-qwen35-fullcg-experiment-debug/debug-phase6.py`: `decode_mrope` is
  wrapped in one large CUDA graph region, but the closest existing experiment
  library crashes during prefill in TVM CUDA graph capture/register handling.
  Treat this as evidence that larger capture is the right direction, not as a
  usable artifact yet.
- `build-qwen35-decodeinputcg-debug/debug-phase6.py`: `decode_mrope` has seven
  CUDA graph regions and six
  `vm.builtin.attention_kv_cache_append_mha_kv_cross_attention` calls outside
  the graphs. This confirms that stable input capture helps graph shape but the
  full-attention KV append/cross-attention boundary is still the remaining
  launch-fragmentation barrier.
- `build-qwen35-separate-crosscg-debug/debug-phase6.py`: `decode_mrope` has
  seven CUDA graph regions and six `attention_kv_cache_append_mha_kv` calls
  outside the graphs; `attention_kv_cache_cross_attention` is moved into lifted
  graph functions. Runtime output is corrupt, so this is evidence that
  cross-attention's KV-cache metadata dependencies are not graph-safe yet.
- `build-qwen35-separate-crosscg-plainaux-debug/debug-phase6.py`: same graph
  shape as the split-cross experiment, but with the runtime forced to use the
  plain aux manager. Runtime output is still corrupt, so the issue is broader
  than cached aux-buffer view offsets.
- `build-qwen35-separate-crosscg-tir-debug/debug-phase6.py`: same seven-region
  split-cross graph shape, compiled with `flashinfer=0`. Runtime output is
  correct but slower, which narrows the unsafe path to FlashInfer's dynamic
  planning/launch assumptions under CUDA graph replay.

TVM has an older contrib vLLM paged-attention kernel in
`3rdparty/tvm/src/runtime/contrib/vllm/attention_kernels.cu`, registered as
`tvm.contrib.vllm.single_query_cached_kv_attention` when `USE_VLLM` is enabled.
The current local MLC Python env does not register these contrib functions, and
that code path is not the current vLLM FlashAttention block-table path. A direct
`USE_VLLM=ON` runtime-object build now succeeds after the local FFI wrapper
fixes, but this remains only a historical/control path unless we add a
repacking-free layout adapter.

There is also a layout mismatch with that older contrib kernel. MLC stores MHA
pages as `[block, 2, kv_head, page, head_dim]`. The contrib vLLM kernel expects
K as `[block, kv_head, head_dim / x, page, x]` and V as
`[block, kv_head, head_dim, page]`. V can be represented as a strided view of
MLC pages, but K cannot be represented as that vectorized layout without
repacking or changing the cache write layout. A per-token repack would likely
erase the win.

## Next High-Value Work

Prototype a paged-FlashAttention backend for MLC full-attention decode layers,
or change the Qwen3.5 full-attention KV storage path so a contiguous
FlashAttention-varlen call is possible without a per-token gather/copy that
erases the win.

In parallel, continue fixing larger `decode_mrope` CUDA graph capture for the
batch-1 path, but use the latest graph dump as the starting point. The current
append/metadata-capture artifact has 7 graph regions, and append plus paged
metadata are already inside captured regions. The only ordinary packed call
still outside graph capture is
`attention_kv_cache_get_query_positions`. A simple attempt to capture that
packed call failed during VM codegen with `Var storage4 is not defined`, so the
next compiler step is deeper graph-region merging / output-storage planning, or
a replay-safe device-side query-position update, not another symbol allowlist.
Use the analyzer's `--regions --parent decode` mode to inspect the exact split
before changing the CUDA graph partitioner; the current useful question is why
the graph that produces append/paged metadata cannot be merged with the adjacent
graph that consumes that metadata in FlashInfer cross-attention.
That question now has a concrete answer: the paged metadata tuple is typed with
`R.Tensor(dtype=..., ndim=...)` fields but no static shapes, so tuple extraction
fails the planner's static-struct-info check and terminates the region after
`get_paged_decode_metadata`.

An opt-in compiler experiment,
`TVM_CUDA_GRAPH_RELAXED_KV_CACHE_METADATA_TUPLES=1`, treats only
`attention_kv_cache_get_paged_decode_metadata` result tuples as static enough
for graph-region propagation. With the same current-best compile knobs and
`SKIP_GEN_CONFIG=1 SKIP_CONVERT=1`, this generated:

```text
dist/libs/logos-multitask-qwen3.5-2026-05-03-best-q0f16-ctx2048-pc320-explicitcrosscg-appendmetacg-relaxedmeta-outputcapture-cuda.so
```

The analyzer then reported `decode_mrope` with one
`vm.builtin.cuda_graph.run_or_capture` call instead of seven, while leaving the
single query-position lookup outside graph capture. The single lifted region
contains all six append calls, all six paged-metadata calls, all six
cross-attention calls, and the LM head. A 64-token Kentucky smoke run produced
coherent output. A full 128-token, 10-run wrapper comparison then measured MLC
at 414.23 tok/s and vLLM at 413.30 tok/s. The follow-up Nsight decode-body
check verified the expected runtime shape: one `cudaGraphLaunch` per
decode-body call.

An alternate `decode_mrope` entry that accepts query positions directly would
was tested as a diagnostic for removing the packed call from the Relax model
body. It compiled and ran, but did not eliminate the underlying synchronization
or metadata dependency and was slower than the current best. It is now kept
behind explicit compile/runtime gates only.
The failed split-cross and forced-plain-aux experiments, plus the correct TIR
control, narrow the required fix further: the FlashInfer full-attention decode
path needs graph-compatible planning. Either make the KV-cache auxiliary
tensors, query-position tensors, FlashInfer planning buffers, launch shape, and
stream synchronization explicit replay inputs for cross-attention, or build a
decode-specific paged-attention backend whose metadata reads are normal tensor
arguments rather than hidden object state.

The vLLM pattern to copy is not just "use FlashAttention"; it is "use fixed or
graph-compatible launch shapes plus explicit metadata tensors." For the local
`sm_89` result, that means an FA2-compatible paged decode call where
`block_table` and `seq_lens`/`seqused_k` are explicit tensor inputs. On Hopper,
the same source also adds FA3 scheduler metadata and split bounds. MLC's next
implementation target should mirror the local FA2 shape for batch-1
full-attention layers instead of replaying a packed
`AttentionKVCacheObj::CrossAttention` method.

Concrete target:

- In `PagedAttentionKVCacheObj::BeginForward`, materialize the batch-1 decode
  page metadata into stable device tensors with shapes fixed to the compiled
  capacity, not views sized to the current number of pages.
- The first runtime bridge now exists:
  `vm.builtin.attention_kv_cache_get_paged_decode_metadata` returns
  `pages`, `page_indptr`, `page_indices`, `length_info`, `k_rope_pos_offset`,
  and `q_rope_position_map` for the current `BeginForward` window. The Python
  wrapper is `PagedKVCache.get_paged_decode_metadata(...)`. This is a
  correctness-oriented bridge only; the returned tensors are still the current
  views, so the next step is fixed-capacity graph replay metadata.
- A second diagnostic bridge exists via `TVM_PAGED_KV_CACHE_STATIC_AUX_SHAPES=1`.
  It now forces fixed-capacity `page_indptr`, `page_indices`, `length_info`, and
  `k_rope_pos_offset` tensors in the plain aux manager, while leaving the query
  RoPE position map exact-size. That still did not fix FlashInfer
  captured-cross replay, so the next implementation should expose FlashInfer's
  planning result and launch assumptions explicitly, not only stabilize page
  metadata tensor shapes.
- In the FlashInfer paged prefill/cross-attention path used by Qwen3.5 GQA
  decode, make the graph-compatible plan produce stable planning buffers and
  fixed launch choices. The current `TVM_FLASHINFER_ENABLE_CUDA_GRAPH_PLAN=1`
  flag and the exposed fixed-split / split-KV-disable toggles do not accomplish
  this for this model.
- Add a compiler-visible attention call for Qwen3.5 decode that takes explicit
  metadata tensors and planning buffers. Do not capture
  `vm.builtin.attention_kv_cache_cross_attention` as a black-box packed method.
- Wire Qwen3.5 full-attention decode layers to that call from
  `python/mlc_llm/model/qwen35/qwen35_model.py`, replacing the current
  `append_mha_kv` + object-state `cross_attention` experiment.
- Only after that lands, re-enable larger CUDA graph capture around the six
  full-attention layers and compare against the current correct 414.23 tok/s
  10-run baseline.

Second-tier work is to test TVM's contrib vLLM single-query kernel behind a
feature flag after building TVM with `USE_VLLM=ON`. This is a useful control
experiment, but it is unlikely to be the main win because it is an older
non-FlashAttention kernel and expects vLLM-style K/V cache tensors plus
`block_tables`/`context_lens`.

That second-tier work is blocked on the layout contract, not on compilation:
the old contrib kernel assumes vLLM's packed K/V cache layout and derives cache
strides from tensor shape. The higher-value route is still a fresh
FA2-compatible paged backend, not reviving the older FasterTransformer-style
contrib kernel first.

## FA2 Paged Backend Integration Sketch

A minimal MLC implementation should avoid a black-box
`AttentionKVCacheObj::CrossAttention` capture. The desired shape is:

1. Runtime: add a new attention backend kind, tentatively `flash_attn_paged`,
   beside `tirx` and `flashinfer` in `attn_backend.{h,cc}`.
2. Runtime: implement a `PagedPrefillFunc`/decode-compatible wrapper whose call
   signature takes the current Q tensor, MLC page storage, explicit page
   metadata, sequence lengths, and output tensor as ordinary `Tensor`
   arguments.
3. Metadata: specialize the batch-1 decode metadata into FA2-style
   `block_table` and `seqused_k`. For MLC's current MHA page layout
   `[block, 2, kv_head, page, head_dim]`, the first batch-1 version can treat
   the block table as a direct view/copy of the physical page ids. Avoid
   per-token K/V repacking. The runtime and frontend metadata bridge for this
   exists; the missing piece is the backend call that consumes the typed tensor
   tuple.
4. Compiler: generate a Qwen3.5 full-attention decode call that invokes the new
   backend with explicit metadata tensors instead of
   `paged_kv_cache.attention_with_fused_qkv(...)` lowering to a packed
   object-state cross-attention call. The guarded lowering hook now exists under
   `MLC_QWEN35_FA2_PAGED_DECODE_ATTENTION=1`; the VM dispatcher for its extern
   now exists, and a narrow prototype CUDA backend is registered as
   `tvm.contrib.flash_attn.fa2_paged_decode`. The next required work is to
   replace that prototype with a production decode kernel or a vendored FA2
   paged-KV subset.
5. CUDA graph: once the FA2 call has fixed tensor inputs and stable metadata
   buffers, re-enable larger `decode_mrope` capture around the six
   full-attention layers and compare against the current correct 414.23 tok/s
   10-run baseline.

The first correctness test is now in place: batch size 1, no sliding window,
`max_seqlen_q = 1`, no softcap, no sinks, and fp16 KV. Only after a real
FA2-class backend beats the current FlashInfer path on this narrow case should
we generalize to batch sizes, quantized KV, split-KV tuning, or Hopper FA3
scheduler metadata.

## 2026-05-04 Explicit Metadata Probe

Implemented a compiler-visible MLC paged cross-attention path under
`MLC_QWEN35_EXPLICIT_PAGED_CROSS_ATTENTION=1`:

- Runtime VM builtin:
  `vm.builtin.attention_kv_cache_cross_attention_with_paged_metadata`.
- Python frontend:
  `PagedKVCache.get_paged_decode_metadata_tensors(...)` and
  `PagedKVCache.cross_attention_with_paged_metadata(...)`.
- Qwen3.5 full-attention decode can now append K/V and call cross-attention
  with `pages`, `page_indptr`, `page_indices`, `length_info`,
  `k_rope_pos_offset`, and `q_rope_position` as normal Relax tensor arguments.
- The CUDA graph rewriter now has two bug fixes needed by this shape:
  recursive argument collection inside tuple arguments, and output marking for
  special captured RNN/KV-cache calls.

Results:

- Correct no-cross-capture artifact:
  `dist/libs/logos-multitask-qwen3.5-2026-05-03-best-q0f16-ctx2048-pc320-explicitpagedcross-nocrosscg-cuda.so`
  ran correctly at about `404 generated tok/s`, `434 decode tok/s`.
- Captured artifact:
  `dist/libs/logos-multitask-qwen3.5-2026-05-03-best-q0f16-ctx2048-pc320-explicitpagedcross-cgfixed2-skipprefill-cuda.so`
  compiled and measured about `416 generated tok/s`, `441 decode tok/s`, but
  produced incorrect text. Do not use it as a valid performance result.

Conclusion: passing MLC page metadata as explicit tensor inputs is useful
plumbing, but capturing this packed cross-attention call is still not
correct. The builtin still depends on runtime-side cache/FlashInfer state and
planning behavior that CUDA graph replay does not model. The next speed work
should move the full-attention decode kernel to a truly explicit backend call
with no hidden planning state, preferably the FA2/vLLM-style paged decode path,
then reattempt graph capture.

Environment note: `flashinfer==0.6.9` in the `mlc` conda env can JIT now that
`ninja` is installed and the env `bin` is on `PATH`. A direct synthetic
`flashinfer.single_decode_with_kv_cache` check at Qwen3.5 shape
`context_len=320, q_heads=8, kv_heads=2, head_dim=256` measured about
`21.5 us/call` after JIT warmup.

## 2026-05-04 FlashInfer/Ninja Follow-up

After installing `ninja` in the `mlc` conda env, the FlashInfer compile path no
longer falls back at the old missing-object failure:

- The compile-only check generated
  `dist/libs/logos-multitask-qwen3.5-2026-05-03-best-q0f16-ctx2048-pc320-flashinfer-ninja-check-cuda.so`.
- The log contained `create_flashinfer_paged_kv_cache` and did not contain the
  earlier `Cannot open ... cached_ops ... The model will fallback to TIR-based
  KV cache` message.
- The 0.6.9 object-path compatibility shim is still needed because FlashInfer
  writes object files with the URI prefix, e.g.
  `<uri>_batch_prefill_paged_kernel_mask_0.cuda.o`, while the TVM wrapper may
  ask for `batch_prefill_paged_kernel_mask_0.cuda.o`.

The end-to-end benchmark remains in the same range as the previous best:

```text
generated_tokens_per_second=404.857
decode_tokens_per_second=435.034
prefill_tokens_per_second=881.294
```

Direct backend microbenchmarks on the same Qwen3.5 decode shape:

```text
flashinfer_single_decode                   21.540 us/call
vllm_flash_attn_paged FA2                  34.516 us/call
mlc fa2 prototype default                  56.014 us/call
mlc fa2 MLC-layout vLLM-style scheduler    17.185 us/call
mlc fa2 physical vLLM-layout scalar        28.983 us/call
mlc fa2 physical vLLM-layout fast          12.101 us/call
```

The newer vLLM-style kernels are fast in isolation. The current issue is not
backend microkernel speed; it is that using these paths inside the full MLC
decode graph has not reduced total synchronized model time. Treat FA2 as a
graph-integration problem now, not as a scalar-kernel tuning problem.

Greedy-output experiments:

- `MLC_ENABLE_GREEDY_ARGMAX=1` on full logits was slower, about
  `391 generated tok/s`.
- Compiling `get_token_ids` with `MLC_QWEN35_FUSED_LM_HEAD_ARGMAX=1` and running
  with `MLC_FUSED_LM_HEAD_ARGMAX=1` was also slower, about
  `395 generated tok/s`.
- Device-token burst with `MLC_DECODE_BURST_STEPS=128`,
  `MLC_DEFER_CPU_TOKEN_BURST=1`, and `--ignore-eos` improved the internal
  `decode_tokens_per_second` metric to about `550`, but did not improve
  end-to-end generated tokens/sec because deferred commit/copy sync moved the
  time outside the inner decode metric.

Recommendation: do not spend more time on the current scalar FA2 prototype or
the current fused argmax path. The next useful work is either:

1. Integrate a FlashInfer-style single-decode backend with explicit tensor
   metadata and no hidden KV-cache planning state, then CUDA-graph the resulting
   decode region.
2. Replace the prototype FA2 paged decode kernel with a vLLM/FlashInfer-class
   split-KV/scheduler design rather than tuning its scalar softmax loop.

## 2026-05-04 vLLM-Style Paged Decode Experiment

After `ninja` fixed FlashInfer JIT, I tested a more direct vLLM-inspired
single-query paged decode kernel inside
`src/runtime/contrib/flash_attn/fa2_paged_decode.cu`. It is gated behind:

```bash
TVM_FA2_PAGED_DECODE_VLLM_STYLE=1
```

The kernel specializes the Qwen3.5 batch-1 decode shape
`q_heads=8, kv_heads=2, head_dim=256, page_size=16`, uses vLLM-style
thread-group QK scheduling, and reads MLC's current page layout directly:
`[page, token, kv_head, head_dim]`.

Microbenchmark result:

```text
context_len=320
seconds_per_call=0.000017248
~17.25 us/call
```

A NumPy reference check passed:

```text
max_abs_out=0.00017476082
mean_abs_out=2.9291657e-05
max_abs_lse=4.7683716e-07
```

Full-model compile:

```bash
SKIP_GEN_CONFIG=1 SKIP_CONVERT=1 \
MLC_QWEN35_FA2_PAGED_DECODE_ATTENTION=1 \
LIB_SUFFIX=-fa2vllmstyle-cg-check \
OPT='flashinfer=1;cublas_gemm=1;faster_transformer=0;cudagraph=1;cutlass=1;ipc_allreduce_strategy=NONE' \
./compile_im.sh
```

Generated:

```text
dist/libs/logos-multitask-qwen3.5-2026-05-03-best-q0f16-ctx2048-pc320-fa2vllmstyle-cg-check-cuda.so
```

Full-model result with the Kentucky image:

```text
generated_tokens_per_second=395.529
decode_tokens_per_second=424.219
```

Synchronized timing:

```text
generated_tokens_per_second=385.309
decode_tokens_per_second=412.124
model_seconds=1.485690
probs_seconds=0.021458
sample_seconds=0.030400
```

This is correct enough for an experiment, but it is not a win. The reason is
now clearer: vLLM's kernel assumes value-cache layout with tokens contiguous for
each value row, while MLC's current page layout stores each token's full
`kv_head x head_dim` vector contiguously. The adapted kernel can vectorize QK
loads, but its V pass must gather across tokens with a large stride. In the
current valid MLC path, FlashInfer's paged-prefill full-attention layer is still
faster than this adapted kernel.

Next compiler/runtime target: do not tune this kernel further in-place. The
larger win is a decode-time KV layout or sidecar full-attention cache layout
that matches the vLLM/FlashAttention value access pattern, or a backend kernel
designed around MLC's layout rather than partially adapting vLLM's layout.

Follow-up sidecar-layout microbench:

```text
context_len=320
layout=mlc,  TVM_FA2_PAGED_DECODE_VLLM_STYLE=1       0.000017185 s/call
layout=vllm, default scalar path                      0.000028983 s/call
layout=vllm, TVM_FA2_PAGED_DECODE_FAST_VLLM_LAYOUT=1  0.000012101 s/call
```

The physical vLLM layout is now the fastest standalone FA2 backend, but the
existing full-model physical-layout artifact still measured only 390.62
generated tok/s and 418.38 effective decode tok/s synchronized. This argues
against making a maintained second KV-cache layout the default until the graph
integration overhead is understood. The next target should stay on the
whole-decode graph: launch count, elementwise/fusion overhead, LM-head cost, and
why the valid best still spends nearly all synchronized decode time in model
kernels.

Refreshing the full physical-layout FA2 combination with
`MLC_QWEN35_FA2_PAGED_DECODE_ATTENTION=1`,
`MLC_QWEN35_FA2_VLLM_CACHE_LAYOUT=1`,
`TVM_FA2_PAGED_DECODE_FAST_VLLM_LAYOUT=1`, and
`TVM_CUDA_GRAPH_CAPTURE_KV_CACHE_CROSS_ATTENTION=1` produced coherent output
but still did not win:

```text
generated_tokens_per_second=392.088
effective_decode_tokens_per_second=420.092
model_seconds=1.457906
```

So the custom FA2 path is graph-safe, unlike captured FlashInfer cross-attn,
but it is still not faster than the current FlashInfer explicit-paged
stable-input baseline.

## 2026-05-04: Batch Decode Graph Boundary Check

After installing `ninja`, FlashInfer codegen is healthy in the `mlc` conda env.
The current valid best remains the FlashInfer/CUDA-graph build:

```text
dist/libs/logos-multitask-qwen3.5-2026-05-03-best-q0f16-ctx2048-pc320-flashinfer-ninja-check-cuda.so
```

Baseline timing on the Kentucky image was roughly:

```text
unsynchronized: generated_tokens_per_second=405, decode_tokens_per_second=435
synchronized:   generated_tokens_per_second=396, decode_tokens_per_second=424
```

A model-side direct RNN-state lookup fix was added for symbolic `batch_decode`
when `max_batch_size=1`. Before this, the direct-state path only triggered when
the batch size was a Python integer literal, so `batch_decode` kept copy-style
RNN state get/set boundaries even in a batch-1 compile.

Structural result:

```text
batch_decode CUDA graph regions: 43 -> 9
batch_decode temp buffer:        3.06 MB -> 1.01 MB
```

Timing result with:

```text
dist/libs/logos-multitask-qwen3.5-2026-05-03-best-q0f16-ctx2048-pc320-symbatch-directrnn-cuda.so
```

was effectively unchanged:

```text
unsynchronized: generated_tokens_per_second=404.322, decode_tokens_per_second=434.061
synchronized:   generated_tokens_per_second=396.086, decode_tokens_per_second=424.287
```

The remaining 9 graph regions are split around the six full-attention layers'
`attention_kv_cache_append_mha_kv_cross_attention` calls, plus the initial and
final decode regions.

An experimental attempt to capture those `batch_decode` KV-cache attention
boundaries was invalid. It produced corrupted text, even though it reported
higher throughput. That confirms the existing compiler warning: the current
KV-cache attention builtin reads changing metadata through object state, so
CUDA graph replay can reuse stale metadata unless page/length/position metadata
is made explicit in the graph inputs.

Next valid attack:

```text
append KV -> expose updated page/length/position tensors -> captured paged attention
```

Do not capture the existing object-state KV-cache attention call directly. The
correct path is a compiler-visible metadata path for decode, or a fused
append+attention runtime op whose replay dependencies are modeled explicitly.

## 2026-05-04: Deferred Commit Timing Correction

`run_im.py` now reports `mlc_effective_decode_metrics` when the serving loop
defers device-token copy/commit. This adds the final
`deferred_commit_copy_sync_seconds` back to the decode denominator so async GPU
work is not hidden outside the inner decode counter.

The rebuilt explicit paged-metadata capture artifact is valid on the Kentucky
image prompt:

```text
dist/libs/logos-multitask-qwen3.5-2026-05-03-best-q0f16-ctx2048-pc320-explicit-paged-packedcapture-rebuilt-cuda.so
```

It structurally captures the explicit paged cross-attention packed calls inside
`batch_decode`, but it did not improve real speed. A clean one-token serving
loop check with `MLC_DEFER_CPU_TOKEN_BURST=0` and
`MLC_DECODE_BURST_STEPS=1` measured:

```text
generated_tokens_per_second=399.237
decode_tokens_per_second=432.757
sample_seconds=2.682864
```

That is the baseline for normal per-token serving. A deferred-burst 10-run
check measured:

```text
generated_tokens_per_second=403.581
mlc_effective_decode_tokens_per_second=433.844
```

The same artifact with `MLC_DECODE_BURST_STEPS=128`,
`MLC_DEFER_CPU_TOKEN_BURST=1`, and `--ignore-eos` can report inner decode
above 500 tok/s, but the final deferred commit sync absorbs the queued GPU
work. Effective decode remains about 431-434 tok/s. Treat the burst result as
a diagnostic for synchronization placement, not as a throughput win.

Conclusion: explicit metadata plumbing and larger packed-call capture are now
verified directionally, but the next speedup must reduce actual decode GPU work
or launch/scheduling overhead. Moving time from per-token sampling into a final
sync does not close the remaining vLLM gap.

A forced-dispatch experiment confirms why the single-token path cannot simply
reuse the better-looking symbolic batch graph. I added the default-off runtime
flag:

```bash
MLC_FORCE_BATCH_DECODE_FOR_SINGLE_SEQ=1
```

With this flag, batch-1 serving calls `batch_decode_mrope` instead of
`decode_mrope`. The graph shape is attractive: `batch_decode_mrope` has 9 CUDA
graph calls and captures the explicit paged cross-attention calls, while
`decode_mrope` has 15 CUDA graph calls and leaves six append/metadata/cross
attention boundaries outside the graphs. However, the forced-batch run produced
corrupted text, despite lowering model time:

```text
generated_tokens_per_second=392.239
decode_tokens_per_second=443.325
model_seconds=0.153855
sample_seconds=2.688739
```

So this is not a valid speedup. It is useful evidence: the symbolic batch graph
still captures or replays state that is unsafe for Qwen3.5 full-attention
decode. The right fix is to move the safe graph structure into the fixed
`decode_mrope` path only after the remaining KV-cache/FlashInfer replay
dependencies are modeled explicitly.

Control check: forcing batch dispatch on the older
`symbatch-directrnn-cuda.so` artifact, which does not use the same captured
explicit-cross graph, produced coherent text but no speed win:

```text
MLC_FORCE_BATCH_DECODE_FOR_SINGLE_SEQ=1
generated_tokens_per_second=272.711  # short 64-token, 2-run smoke
decode_tokens_per_second=426.041
model_seconds=0.025267
sample_seconds=0.268328
```

This separates the two issues. Batch dispatch itself can be correct; the bad
output comes from the captured explicit cross-attention graph shape. But batch
dispatch alone is not a throughput optimization.

I also tested a tensor-only FA2/vLLM cache-layout variant to see whether the
local FA2 prototype was slow mainly because the MLC-layout V cache forced
strided scalar gathers. The opt-in flags are:

```bash
MLC_QWEN35_FA2_PAGED_DECODE_ATTENTION=1
MLC_QWEN35_FA2_VLLM_CACHE_LAYOUT=1
TVM_FA2_PAGED_DECODE_VLLM_STYLE=1
```

The generated library was:

```text
dist/libs/logos-multitask-qwen3.5-2026-05-03-best-q0f16-ctx2048-pc320-fa2-vllmcachelayout-vllmstyle-cuda.so
```

After rebuilding both runtime trees used by `run_im.sh` (`build-mlc-env` and
`build-tvm-full`), this path is not functionally valid. It generates corrupted
text on the Kentucky image prompt:

```text
Thisivelyрусизованіяныныны...
```

The earlier coherent result came from stale runtime loading: `run_im.sh` can
prefer `build-tvm-full` through `TVM_LIBRARY_PATH`, while I had rebuilt only
`build-mlc-env`. Once the loaded TVM runtime matched the new metadata code, the
vLLM-style K/V view was active (`k_ndim=5`, `v_ndim=4`) and the output was bad.

Root cause: vLLM's flash-attention cache layout is a physical storage choice,
not something we can recover with a metadata-only strided view. The local
`FA2PagedDecodeBatch1VllmLayoutKernel` assumes the inner page layout is already
physically transposed for vLLM-style K/V access:

```text
K: [page, kv_head, head_dim / x, page_size, x]
V: [page, kv_head, head_dim, page_size]
```

MLC's paged cache is physically laid out as one combined page with K and V:

```text
[page, 2, kv_head, page_size, head_dim]
```

The kernel's fast path only accepts the outer page/head strides and then
hardcodes the inner vLLM addressing. A strided `Tensor` view can report
`k_ndim=5`, but it cannot make the V values for a fixed head dimension
contiguous across page offsets when the underlying memory is MLC's
page-offset-major layout. This explains why both FA2/vLLM-cache-layout artifacts
generate corrupted text once the correct runtime is loaded.

Conclusion: this local vLLM-cache-layout view has incorrect semantics for the
current FA2 kernel path. The current best remains the explicit paged-metadata
stable decode-input path. A real win here would need a physical KV-cache layout
specialization: append/store K and V in the layout the decode kernel consumes,
then compile the decode graph against that layout.

The relevant MLC storage contract is spread across the page-management kernels
and the runtime page allocation. The current MHA layout is hard-coded in:

- `3rdparty/tvm/python/tvm/relax/frontend/nn/llm/_page_kernels.py`
  `_kv_cache_transpose_append`, `_kv_cache_debug_get_kv`, `_copy_single_page`,
  and `_compact_kv_copy`, all of which index pages as
  `[num_pages, 2, num_heads, page_size, head_dim]`.
- `3rdparty/tvm/python/tvm/relax/frontend/nn/llm/kv_cache.py`, where both
  `FlashInferPagedKVCache` and `TIRPagedKVCache` register those page kernels
  into `vm.builtin.paged_attention_kv_cache_create`.
- `3rdparty/tvm/src/runtime/vm/paged_kv_cache.cc`, where `pages_` is allocated
  per layer and all append, page-copy, compaction, debug, and attention calls
  pass the same physical page tensor.

The next architectural optimization is now started as an opt-in physical layout
specialization, not another metadata view. The minimal viable design is:

1. Add a compile-time page-layout enum for MHA pages, initially
   `mlc_combined` and `vllm_split`.
2. Allocate/store vLLM-split pages for the full-attention Qwen3.5 layers as
   K `[page, kv_head, head_dim / x, page_size, x]` and
   V `[page, kv_head, head_dim, page_size]`, or store separate K/V tensors with
   equivalent strides.
3. Generate matching append/debug/page-copy/compact kernels for that layout.
4. Route only the FA2 paged decode backend through this storage at first.
   Existing FlashInfer/TIR attention should keep the current combined layout
   until separate layout support is added there.
5. Validate against `get_kv`/NumPy-reference paged decode tests before running
   the image benchmark. Any metadata-only vLLM-cache-layout benchmark should be
   treated as invalid.

Current implementation status: `MLC_QWEN35_FA2_VLLM_CACHE_LAYOUT=1` now changes
the generated append kernel to physically write K/V in the vLLM-style layout
while keeping the backing allocation size unchanged. The FA2 metadata path then
returns rank-5 K and rank-4 V views matching that physical write order.
Layout-aware debug/page-copy/compact helper factories are also selected behind
the same flag. Unit coverage now prevents mixing vLLM-layout append with
default-layout page helpers and runs an LLVM functional roundtrip that appends
logical K/V into the vLLM physical layout, reads it back, copies a page, and
compacts selected positions while preserving the logical K/V values. This is
also covered through a real `PagedKVCache` runtime object: a CPU cache wired
with the vLLM-layout append/debug/page-copy/compact functions can append MHA
K/V through `vm.builtin.attention_kv_cache_append_mha_kv` and read back the same
logical values through `vm.builtin.attention_kv_cache_debug_get_kv`. The same
test forks the sequence, appends one token to the fork, and verifies both the
parent and forked logical K/V values, which exercises the runtime page-copy
path behind the vLLM-layout helper. This is still a prototype gate, not a new
default: the first valid performance target remains the simple batch-1 Qwen3.5
image benchmark path that appends then immediately calls the FA2 paged decode
backend.

The first correctness-fixed physical-layout benchmark is coherent but slower
than the current best:

```text
MLC_QWEN35_FA2_VLLM_CACHE_LAYOUT=1
MLC_MODEL_LIB=dist/libs/logos-multitask-qwen3.5-2026-05-03-best-q0f16-ctx2048-pc320-fa2-physicalvllmkv-cuda.so
generated_tokens_per_second=387.661
mlc_effective_decode_tokens_per_second=415.245
```

This establishes the physical-layout plumbing and removes the previous
metadata-only corruption, but it is not a speed win yet. The reason is that the
current correct backend path uses a scalar vLLM-layout decode kernel. The older
vectorized `FA2PagedDecodeBatch1VllmLayoutKernel` failed a direct NumPy
reference check (`max_out_abs_err` around 1.70), so it is now gated behind
`TVM_FA2_PAGED_DECODE_FAST_VLLM_LAYOUT` until its inner addressing/reduction is
fixed. The corrected rank-5 K / rank-4 V scalar backend matches NumPy reference
with `max_out_abs_err` around `4.5e-4`.

After fixing the vLLM-layout metadata strides and enabling the fast vectorized
branch, the direct backend microbenchmark improved substantially, but the
full-model benchmark still did not beat the current best:

```text
MLC_QWEN35_FA2_VLLM_CACHE_LAYOUT=1
TVM_FA2_PAGED_DECODE_FAST_VLLM_LAYOUT=1
MLC_MODEL_LIB=dist/libs/logos-multitask-qwen3.5-2026-05-03-best-q0f16-ctx2048-pc320-fa2-physicalvllmkv-cuda.so
generated_tokens_per_second=403.314
mlc_effective_decode_tokens_per_second=433.348
```

With `MLC_SYNC_DECODE_TIMING=1`, the deferred commit copy/sync bucket collapses
to about `0.00035 s`; the model execution bucket dominates:

```text
physical vLLM-layout FA2: generated_tokens_per_second=390.903, model_seconds=0.877466
current best stable path: generated_tokens_per_second=393.714, model_seconds=0.869761
```

So the large unsynchronized deferred-commit number was a timing attribution
artifact: it was the first final sync that waited for queued decode kernels, not
an expensive CPU token commit. The current physical vLLM-layout FA2 path is
coherent and useful as infrastructure, but it is not a baseline candidate.

I also tried compiling the same FA2/vLLM-cache-layout path with the existing
CUDA graph capture gate enabled:

```bash
TVM_CUDA_GRAPH_CAPTURE_KV_CACHE_CROSS_ATTENTION=1
```

The generated library was:

```text
dist/libs/logos-multitask-qwen3.5-2026-05-03-best-q0f16-ctx2048-pc320-fa2-vllmcachelayout-crosscg-cuda.so
```

After rebuilding the loaded TVM runtime, this no longer fails the rank check.
The metadata diagnostic confirms the captured call receives the intended view
rank:

```text
FA2 metadata layout layer=0 depth=0 use_vllm_cache_layout=1
k_ndim=5 v_ndim=4 page_size=16 num_kv_heads=2 head_dim=256
```

However, it produces the same corrupted text as the non-captured
vLLM-cache-layout path. Leave both artifacts rejected unless the K/V view
mapping is corrected against the FA2 kernel's expected physical layout.

Control experiment: keep MLC's physical cache layout and use only the local
FA2/vLLM-style scheduler path:

```text
dist/libs/logos-multitask-qwen3.5-2026-05-03-best-q0f16-ctx2048-pc320-fa2vllmstyle-crosscg-cuda.so
TVM_FA2_PAGED_DECODE_VLLM_STYLE=1
```

This path is coherent, but slower than the current best on the same 512x512
Kentucky prompt benchmark:

```text
generated_tokens_per_second=398.297
mlc_effective_decode_tokens_per_second=427.426
```

So the remaining gap is not solved by swapping in the current local FA2
scheduler on top of MLC's existing physical paged-cache layout.

After adding vLLM-layout debug/page-copy/compact helpers and runtime-object
tests, I rebuilt the physical vLLM-layout FA2 artifact:

```bash
MODEL_SUFFIX=-fa2-physicalvllmkv-helpers \
LIB_SUFFIX=-fa2-physicalvllmkv-helpers \
QUANT=q0f16 \
CONTEXT_WINDOW_SIZE=2048 \
PREFILL_CHUNK_SIZE=320 \
MAX_BATCH_SIZE=1 \
MLC_QWEN35_FA2_PAGED_DECODE_ATTENTION=1 \
MLC_QWEN35_FA2_VLLM_CACHE_LAYOUT=1 \
TVM_FA2_PAGED_DECODE_FAST_VLLM_LAYOUT=1 \
OPT='flashinfer=1;cublas_gemm=1;faster_transformer=0;cudagraph=1;cutlass=1;ipc_allreduce_strategy=NONE' \
./compile_im.sh
```

The artifact is coherent on the Kentucky image prompt, but still not a speed
win:

```text
dist/libs/logos-multitask-qwen3.5-2026-05-03-best-q0f16-ctx2048-pc320-fa2-physicalvllmkv-helpers-cuda.so
generated_tokens_per_second=400.551
decode_tokens_per_second=429.605
```

The same 5-run control on the current best stable path measured:

```text
dist/libs/logos-multitask-qwen3.5-2026-05-03-best-q0f16-ctx2048-pc320-explicit-paged-stabledecodeinput-cuda.so
generated_tokens_per_second=403.629
decode_tokens_per_second=433.275
```

Conclusion: the physical vLLM-layout plumbing is now better guarded and valid
for simple append/fork/page-copy behavior, but the current FA2 local backend
still loses a few tokens/s to the stable FlashInfer/explicit-paged path. Keep it
as infrastructure, not the active baseline.

## 2026-05-04 Kernel Fusion Audit

To answer whether the current `.so` kernels are optimized enough, I built the
same relaxed-metadata artifact with CUDA graph disabled so Nsight can see the
internal decode kernels directly:

```bash
SKIP_GEN_CONFIG=1 SKIP_CONVERT=1 \
MODEL_SUFFIX=-fusedinproj \
LIB_SUFFIX=-explicitcrosscg-appendmetacg-relaxedmeta-nocg \
DEBUG_DUMP=debug-explicitcrosscg-appendmetacg-relaxedmeta-nocg \
QUANT=q0f16 CONTEXT_WINDOW_SIZE=2048 PREFILL_CHUNK_SIZE=320 MAX_BATCH_SIZE=1 \
OPT='flashinfer=1;cublas_gemm=1;faster_transformer=0;cudagraph=0;cutlass=1;ipc_allreduce_strategy=NONE' \
./compile_im.sh
```

The diagnostic no-graph library measured `396.528 tok/s` over 128 direct
decode-body calls without Nsight and `370.303 tok/s` under Nsight capture:

```text
dist/libs/logos-multitask-qwen3.5-2026-05-03-best-q0f16-ctx2048-pc320-explicitcrosscg-appendmetacg-relaxedmeta-nocg-cuda.so
decode_body_seconds=0.322802
decode_body_tokens_per_second=396.528
```

The exported no-graph profile is:

```text
nsys-qwen35-relaxedmeta-nocg-decodebody.sqlite
```

The reproducible summary command is:

```bash
/home/cwong/Projects/miniconda/envs/mlc/bin/python summarize_nsys_kernels.py \
  nsys-qwen35-relaxedmeta-nocg-decodebody.sqlite \
  --tokens 128 \
  --categories \
  --limit 20
```

The kernel-time attribution per decode token was:

```text
gemv_gemm        1732.960 us/tok   98.000 calls/tok
other             255.254 us/tok  169.000 calls/tok
elementwise       113.838 us/tok  112.000 calls/tok
gdn_linear         89.634 us/tok   18.000 calls/tok
flash_attention    51.158 us/tok   12.000 calls/tok
kv_cache            7.533 us/tok    6.000 calls/tok
```

The largest individual kernels were:

```text
cuBLAS hidden/projection GEMV kernels  1212.204 us/tok, 96.000 calls/tok
cuBLAS LM-head GEMV kernel              519.620 us/tok,  1.000 calls/tok
gdn_packed_decode_state_storage          89.634 us/tok, 18.000 calls/tok
rms_norm8                                76.528 us/tok, 49.000 calls/tok
causal_conv1d_decode_state_storage       72.651 us/tok, 18.000 calls/tok
FlashInfer paged attention               41.266 us/tok,  6.000 calls/tok
```

The production CUDA-graph profile confirms these kernels are used by the `.so`,
but hidden behind graph replay:

```bash
/home/cwong/Projects/miniconda/envs/mlc/bin/python summarize_nsys_kernels.py \
  nsys-qwen35-relaxedmeta-decodebody.sqlite \
  --tokens 128 \
  --categories \
  --limit 20
```

Visible work in the graph-enabled profile:

```text
total_graph_trace_ms=280.244 calls=128 per_unit_us=2189.405
cudaGraphLaunch_v10000 29.646 us/tok, 1.000 calls/tok
fused_reshape_broadcast_to6_add9_expand_dims4_broadcast_to7_reshape18_cast9_kernel
  0.978 us/tok, 1.000 calls/tok
```

So CUDA graph capture is doing the right launch-overhead optimization: nearly
all decode kernels are inside a single graph replay per token, and the only
remaining visible kernel is query-position/mrope prep at about `1 us/tok`.

The LM-head fusion microbenchmark was refreshed with:

```bash
/home/cwong/Projects/miniconda/envs/mlc/bin/python bench_lm_head_fused_argmax.py \
  --runs 200 \
  --warmup-runs 20 \
  --block-m 32 \
  --block-k 1024
```

Result:

```text
vocab=248320 hidden=1024 block_m=32
fused_ms_per_run=0.552
torch_logits_argmax_ms_per_run=2.694
```

This validates that a fused matvec plus argmax kernel is in the same cost range
as the current Nsight-observed LM-head GEMV (`~0.52 ms/token`) and much faster
than materializing logits then reducing in a naive PyTorch path. It does not
prove a speedup for normal sampling because greedy argmax can avoid logits and
probabilities while top-p/top-k sampling usually cannot.

The state-only microprofile was also refreshed on the production graph artifact:

```bash
MLC_MODEL=dist/logos-multitask-qwen3.5-2026-05-03-best-fusedinproj-q0f16-ctx2048-pc320-MLC \
MLC_MODEL_LIB=dist/libs/logos-multitask-qwen3.5-2026-05-03-best-q0f16-ctx2048-pc320-explicitcrosscg-appendmetacg-relaxedmeta-outputcapture-cuda.so \
./profile_qwen35_vm_phases.sh \
  --image kentucky.png \
  --fit-image-size 512 \
  --phase decode-state \
  --runs 512 \
  --warmup-runs 8 \
  --reuse-input-tensors
```

Result:

```text
decode_state_seconds=0.006797
decode_state_runs_per_second=75324.758
```

That is about `13 us/token`, so KV/RNN begin/end bookkeeping is not a first
order fusion target.

Fusion priority after the no-graph audit:

1. Batch-1 GEMV specialization is the highest-priority compile-time target.
   About `1.73 ms/token` is GEMV/GEMM, and the LM head alone is about
   `0.52 ms/token`.
2. LM-head plus sampling fusion is worth pursuing when the serving mode can
   avoid full logits/probabilities. It is behavior-constrained, but the
   measured cost is large enough to justify a targeted implementation.
3. GDN/linear-attention decode fusion is the next model-specific target:
   `gdn_packed_decode_state_storage`, `causal_conv1d_decode_state_storage`,
   the repeated `rms_norm8`, and adjacent split/cast/multiply kernels account
   for a few hundred microseconds per token when grouped.
4. Generic elementwise fusion is lower priority unless it removes memory
   traffic. CUDA graph already removes most launch overhead, so fusing tiny
   `~1 us` kernels only helps if the fused kernel avoids redundant global
   reads/writes.
5. FlashInfer attention is not the next target on the current 2K-context,
   batch-1 decode case. It is only about `51 us/token` in the no-graph
   attribution profile, and earlier FA2/vLLM-style backend experiments did not
   beat the stable FlashInfer/explicit-paged path.

The current `.so` is therefore optimized enough at the runtime-launch level and
already uses cuBLAS plus FlashInfer where they matter, but it is not optimized
enough at the compiler/model-kernel level. The remaining speedups should come
from generating better batch-1 projection kernels and fusing Qwen3.5's
decode-specific GDN/LM-head work, not from another attention backend swap.
