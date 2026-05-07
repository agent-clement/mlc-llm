/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/engine_actions/batch_decode.cc
 */

#include <tvm/runtime/device_api.h>
#include <tvm/runtime/nvtx.h>

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>

#include "../../support/random.h"
#include "../config.h"
#include "../model.h"
#include "../sampler/sampler.h"
#include "action.h"
#include "action_commons.h"

namespace mlc {
namespace llm {
namespace serve {

namespace {

inline void CopyArray(Tensor src, Tensor dst, TVMStreamHandle stream) {
  DLTensor dl_dst = *(dst.operator->());
  Tensor::CopyFromTo(src.operator->(), &dl_dst, stream);
}

bool SplitDecodeGetLogitsEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("MLC_SPLIT_DECODE_GET_LOGITS");
    return value != nullptr && std::string(value) == "1";
  }();
  return enabled;
}

bool FusedLMHeadArgmaxEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("MLC_FUSED_LM_HEAD_ARGMAX");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();
  return enabled;
}

bool ReuseDeviceSampledTokenEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("MLC_REUSE_DEVICE_SAMPLED_TOKEN");
    return value == nullptr || (value[0] != '\0' && value[0] != '0');
  }();
  return enabled;
}

bool StableDecodeEmbeddingWorkspaceEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("MLC_STABLE_DECODE_EMBEDDING");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();
  return enabled;
}

int DecodeBurstSteps() {
  static const int steps = [] {
    const char* value = std::getenv("MLC_DECODE_BURST_STEPS");
    if (value == nullptr || value[0] == '\0') {
      return 1;
    }
    try {
      return std::max(1, std::stoi(value));
    } catch (...) {
      return 1;
    }
  }();
  return steps;
}

bool DeferCpuTokenBurstEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("MLC_DEFER_CPU_TOKEN_BURST");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();
  return enabled;
}

bool SyncDecodeTimingEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("MLC_SYNC_DECODE_TIMING");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();
  return enabled;
}

inline double SecondsSince(std::chrono::time_point<std::chrono::high_resolution_clock> start) {
  auto end = std::chrono::high_resolution_clock::now();
  return static_cast<double>((end - start).count()) / 1e9;
}

inline void SyncTensorStreamForTiming(Tensor tensor) {
  if (!SyncDecodeTimingEnabled() || !tensor.defined()) {
    return;
  }
  Device device = tensor->device;
  DeviceAPI::Get(device)->StreamSync(device, DeviceAPI::Get(device)->GetCurrentStream(device));
}

}  // namespace

/*!
 * \brief The action that runs one-step decode for requests in the
 * `running_queue` of engine state. Preempt low-priority requests
 * accordingly when it is impossible to decode all the running requests.
 * \note The BatchDecode action **does not** take effect for speculative
 * decoding scenarios where there are multiple models. For speculative
 * decoding in the future, we will use other specific actions.
 */
class BatchDecodeActionObj : public EngineActionObj {
 public:
  explicit BatchDecodeActionObj(Array<Model> models, Tokenizer tokenizer,
                                LogitProcessor logit_processor, Sampler sampler,
                                std::vector<ModelWorkspace> model_workspaces,
                                EngineConfig engine_config,
                                Optional<EventTraceRecorder> trace_recorder)
      : models_(std::move(models)),
        tokenizer_(std::move(tokenizer)),
        logit_processor_(std::move(logit_processor)),
        sampler_(std::move(sampler)),
        model_workspaces_(std::move(model_workspaces)),
        engine_config_(std::move(engine_config)),
        trace_recorder_(std::move(trace_recorder)) {}

  Array<Request> Step(EngineState estate) final {
    auto action_tstart = std::chrono::high_resolution_clock::now();
    // - Do not run decode when there is no running request.
    if (estate->running_queue.empty()) {
      return {};
    }

    // Preempt request state entries when decode cannot apply.
    std::vector<RequestStateEntry> running_rsentries;
    {
      NVTXScopedRange nvtx_scope("BatchDecode getting requests");
      running_rsentries = estate->GetRunningRequestStateEntries();
      while (!CanDecode(running_rsentries.size())) {
        if (estate->prefix_cache->TryFreeMemory()) continue;
        RequestStateEntry preempted =
            PreemptLastRunningRequestStateEntry(estate, models_, std::nullopt, trace_recorder_);
        if (preempted.same_as(running_rsentries.back())) {
          running_rsentries.pop_back();
        }
      }
      while (running_rsentries.size() >
             std::min(static_cast<int64_t>(engine_config_->max_num_sequence),
                      engine_config_->prefill_chunk_size)) {
        running_rsentries.pop_back();
      }
    }

    // NOTE: Right now we only support decode all the running request states at a time.
    int num_rsentries = running_rsentries.size();
    TVM_FFI_ICHECK_GT(num_rsentries, 0)
        << "There should be at least one request state entry that can run decode. "
           "Possible failure reason: none of the prefill phase of the running requests is finished";
    TVM_FFI_ICHECK_LE(num_rsentries, engine_config_->max_num_sequence)
        << "The number of running requests exceeds the max number of sequence in EngineConfig. "
           "Possible failure reason: the prefill action allows new sequence in regardless of the "
           "max num sequence.";
    int burst_steps = GetSafeDecodeBurstSteps(running_rsentries);
    bool defer_cpu_tokens = CanDeferCpuTokenBurst(running_rsentries, burst_steps);
    Tensor deferred_token_ids_host{nullptr};
    Tensor deferred_token_ids_device{nullptr};
    int deferred_token_count = 0;
    Tensor deferred_next_input_device{nullptr};
    Tensor deferred_last_token_device{nullptr};
    auto prepare_tend = std::chrono::high_resolution_clock::now();
    estate->metrics.engine_batch_decode_prepare_time_sum +=
        static_cast<double>((prepare_tend - action_tstart).count()) / 1e9;
    for (int burst_idx = 0; burst_idx < burst_steps; ++burst_idx) {
      auto tstart = std::chrono::high_resolution_clock::now();

      // Collect
      // - the last committed token,
      // - the request id,
      // - the generation config,
      // - the random number generator,
      // of each request state entry.
      std::vector<int> input_tokens;
      std::vector<int> lengths;
      Array<String> request_ids;
      std::vector<int64_t> request_internal_ids;
      Array<RequestModelState> mstates;
      Array<GenerationConfig> generation_cfg;
      std::vector<RandomGenerator*> rngs;
      Tensor input_tokens_device{nullptr};

      input_tokens.reserve(num_rsentries);
      request_ids.reserve(num_rsentries);
      request_internal_ids.reserve(num_rsentries);
      mstates.reserve(num_rsentries);
      generation_cfg.reserve(num_rsentries);
      rngs.reserve(num_rsentries);

      {
        NVTXScopedRange nvtx_scope("BatchDecode setting batch info");
        for (const RequestStateEntry& rsentry : running_rsentries) {
          auto mstate = rsentry->mstates[0];
          if (defer_cpu_tokens && burst_idx > 0) {
            TVM_FFI_ICHECK(deferred_next_input_device.defined());
            input_tokens_device = deferred_next_input_device;
            lengths.push_back(1);
          } else {
            TVM_FFI_ICHECK(mstate->num_tokens_for_next_decode > 0 &&
                           mstate->num_tokens_for_next_decode <=
                               static_cast<int>(mstate->committed_tokens.size()));

            if (ReuseDeviceSampledTokenEnabled() && num_rsentries == 1 &&
                mstate->num_tokens_for_next_decode == 1 && models_[0]->SupportsDeviceTokenEmbed()) {
              const SampleResult& last_token = mstate->committed_tokens.back();
              if (last_token.sampled_token_ids_device.defined() &&
                  last_token.sampled_token_ids_device_offset == 0) {
                input_tokens_device = last_token.sampled_token_ids_device;
              }
            }

            for (auto begin = mstate->committed_tokens.end() - mstate->num_tokens_for_next_decode;
                 begin != mstate->committed_tokens.end(); ++begin) {
              input_tokens.push_back(begin->GetTokenId());
            }

            lengths.push_back(mstate->num_tokens_for_next_decode);
            mstate->num_tokens_for_next_decode = 0;
          }

          request_ids.push_back(rsentry->request->id);
          request_internal_ids.push_back(mstate->internal_id);
          mstates.push_back(mstate);
          generation_cfg.push_back(rsentry->request->generation_cfg);
          rngs.push_back(&rsentry->rng);
        }
      }

      // - Compute embeddings.
      auto model_tstart = std::chrono::high_resolution_clock::now();
      RECORD_EVENT(trace_recorder_, request_ids, "start embedding");
      ObjectRef* embedding_dst = nullptr;
      if (StableDecodeEmbeddingWorkspaceEnabled() && num_rsentries == 1 &&
          !model_workspaces_.empty() && model_workspaces_[0].embeddings.defined()) {
        embedding_dst = &model_workspaces_[0].embeddings;
      }
      ObjectRef embeddings =
          input_tokens_device.defined()
              ? models_[0]->TokenEmbed(input_tokens_device, embedding_dst)
              : models_[0]->TokenEmbed({IntTuple(input_tokens.begin(), input_tokens.end())},
                                        embedding_dst);
      RECORD_EVENT(trace_recorder_, request_ids, "finish embedding");

      // - Invoke model decode.
      // If every request only requires to process one token, batch decode kernel is called.
      // Otherwise, batch prefill kernel is called.
      bool is_every_request_single_token =
          std::all_of(lengths.begin(), lengths.end(), [](int len) { return len == 1; });
      RECORD_EVENT(trace_recorder_, request_ids, "start decode");
      Tensor logits;
      Tensor token_ids_device_from_model{nullptr};
      Tensor timing_tensor{nullptr};
      if (is_every_request_single_token) {
        if (FusedLMHeadArgmaxEnabled() && models_[0]->CanGetTokenIds() &&
            logit_processor_->CanBypassLogitsForGreedy(generation_cfg, mstates)) {
          ObjectRef hidden_states =
              models_[0]->BatchDecodeToLastHidden(embeddings, request_internal_ids);
          token_ids_device_from_model = models_[0]->GetTokenIds(hidden_states);
          TVM_FFI_ICHECK_EQ(token_ids_device_from_model->ndim, 1);
          TVM_FFI_ICHECK_EQ(token_ids_device_from_model->shape[0], num_rsentries);
          timing_tensor = token_ids_device_from_model;
        } else if (SplitDecodeGetLogitsEnabled() && models_[0]->CanGetLogits()) {
          ObjectRef hidden_states =
              models_[0]->BatchDecodeToLastHidden(embeddings, request_internal_ids);
          logits = models_[0]->GetLogits(hidden_states);
          TVM_FFI_ICHECK_EQ(logits->ndim, 2);
          TVM_FFI_ICHECK_EQ(logits->shape[0], num_rsentries);
          timing_tensor = logits;
        } else {
          logits = models_[0]->BatchDecode(embeddings, request_internal_ids);
          TVM_FFI_ICHECK_EQ(logits->ndim, 3);
          TVM_FFI_ICHECK_EQ(logits->shape[0], num_rsentries);
          TVM_FFI_ICHECK_EQ(logits->shape[1], 1);
          timing_tensor = logits;
        }
      } else {
        logits = models_[0]->BatchPrefill(embeddings, request_internal_ids, lengths);
        TVM_FFI_ICHECK_EQ(logits->ndim, 3);
        TVM_FFI_ICHECK_EQ(logits->shape[0], 1);
        TVM_FFI_ICHECK_EQ(logits->shape[1], num_rsentries);
        timing_tensor = logits;
      }
      RECORD_EVENT(trace_recorder_, request_ids, "finish decode");
      SyncTensorStreamForTiming(timing_tensor);
      estate->metrics.engine_batch_decode_model_time_sum += SecondsSince(model_tstart);

      // - Update logits.
      auto logits_update_tstart = std::chrono::high_resolution_clock::now();
      if (!token_ids_device_from_model.defined()) {
        if (logits->ndim == 3) {
          logits = logits.CreateView({num_rsentries, logits->shape[2]}, logits->dtype);
        } else {
          TVM_FFI_ICHECK_EQ(logits->ndim, 2);
          TVM_FFI_ICHECK_EQ(logits->shape[0], num_rsentries);
        }
        logit_processor_->InplaceUpdateLogits(logits, generation_cfg, mstates, request_ids);
        SyncTensorStreamForTiming(logits);
      }
      estate->metrics.engine_batch_decode_logits_update_time_sum +=
          SecondsSince(logits_update_tstart);

      // - Commit the prefix cache changes from previous round of action.
      // Note: we commit prefix cache changes here to overlap this commit with the GPU execution.
      estate->prefix_cache->CommitSequenceExtention();

      // - Sample tokens.
      std::vector<SampleResult> sample_results;
      if (defer_cpu_tokens) {
        TVM_FFI_ICHECK_EQ(num_rsentries, 1);
        Tensor token_ids_device{nullptr};
        if (token_ids_device_from_model.defined()) {
          token_ids_device = token_ids_device_from_model;
        } else if (logit_processor_->CanSampleGreedy(generation_cfg)) {
          auto sample_tstart = std::chrono::high_resolution_clock::now();
          token_ids_device = logit_processor_->SampleGreedyTokenIdsDeviceFromLogits(
              logits, generation_cfg, request_ids);
          SyncTensorStreamForTiming(token_ids_device);
          estate->metrics.engine_batch_decode_sample_time_sum += SecondsSince(sample_tstart);
        } else {
          std::vector<int> sample_indices{0};
          auto probs_tstart = std::chrono::high_resolution_clock::now();
          Tensor probs_on_device =
              logit_processor_->ComputeProbsFromLogits(logits, generation_cfg, request_ids);
          SyncTensorStreamForTiming(probs_on_device);
          estate->metrics.engine_batch_decode_probs_time_sum += SecondsSince(probs_tstart);
          auto sample_tstart = std::chrono::high_resolution_clock::now();
          Tensor renormalized_probs = sampler_->BatchRenormalizeProbsByTopP(
              probs_on_device, sample_indices, request_ids, generation_cfg);
          token_ids_device = sampler_->BatchSampleTokenIdsDeviceAfterTopP(
              renormalized_probs, sample_indices, request_ids, generation_cfg, rngs);
          SyncTensorStreamForTiming(token_ids_device);
          estate->metrics.engine_batch_decode_sample_time_sum += SecondsSince(sample_tstart);
        }
        TVM_FFI_ICHECK(token_ids_device.defined());
        deferred_next_input_device = token_ids_device;
        deferred_last_token_device = token_ids_device;
        if (!deferred_token_ids_device.defined()) {
          std::tie(deferred_token_ids_device, deferred_token_ids_host) =
              GetDeferredTokenBuffers(token_ids_device->device, burst_steps);
        }
        auto device_copy_tstart = std::chrono::high_resolution_clock::now();
        CopyDeviceTokenIdToBuffer(token_ids_device, deferred_token_ids_device,
                                  deferred_token_count);
        SyncTensorStreamForTiming(deferred_token_ids_device);
        estate->metrics.engine_batch_decode_device_token_copy_time_sum +=
            SecondsSince(device_copy_tstart);
        ++deferred_token_count;
      } else if (token_ids_device_from_model.defined() ||
                 logit_processor_->CanSampleGreedy(generation_cfg)) {
        auto sample_tstart = std::chrono::high_resolution_clock::now();
        sample_results = token_ids_device_from_model.defined()
                             ? CopyDeviceTokenIdsToSampleResults(token_ids_device_from_model,
                                                                 num_rsentries)
                             : logit_processor_->SampleGreedyFromLogits(logits, generation_cfg,
                                                                        request_ids);
        estate->metrics.engine_batch_decode_sample_time_sum += SecondsSince(sample_tstart);
      } else {
        // Fill range [0, num_rsentries) into `sample_indices`.
        std::vector<int> sample_indices(num_rsentries);
        std::iota(sample_indices.begin(), sample_indices.end(), 0);
        auto probs_tstart = std::chrono::high_resolution_clock::now();
        Tensor probs_on_device =
            logit_processor_->ComputeProbsFromLogits(logits, generation_cfg, request_ids);
        SyncTensorStreamForTiming(probs_on_device);
        estate->metrics.engine_batch_decode_probs_time_sum += SecondsSince(probs_tstart);
        auto sample_tstart = std::chrono::high_resolution_clock::now();
        Tensor renormalized_probs = sampler_->BatchRenormalizeProbsByTopP(
            probs_on_device, sample_indices, request_ids, generation_cfg);
        sample_results = sampler_->BatchSampleTokensWithProbAfterTopP(
            renormalized_probs, sample_indices, request_ids, generation_cfg, rngs);
        estate->metrics.engine_batch_decode_sample_time_sum += SecondsSince(sample_tstart);
      }
      if (!defer_cpu_tokens) {
        TVM_FFI_ICHECK_EQ(sample_results.size(), num_rsentries);
      }

      bool stop_burst = false;
      // - Update the committed tokens of states.
      if (!defer_cpu_tokens) {
        for (int i = 0; i < num_rsentries; ++i) {
          auto mstate = mstates[i];

          if (!mstate->require_retokenization_in_next_decode) {
            mstates[i]->CommitToken(sample_results[i]);
            // live update the output metrics
            running_rsentries[i]->rstate->metrics.completion_tokens += 1;
          } else {
            // Retokenize and commit tokens.
            CommitTokenMayRetokenize(running_rsentries[i], mstate, sample_results[i]);
            mstate->require_retokenization_in_next_decode = false;
            stop_burst = true;
          }

          running_rsentries[i]->rstate->metrics.decode_tokens += lengths[i];
          stop_burst = stop_burst || ShouldStopDecodeBurst(running_rsentries[i], mstate);
        }
      } else {
        running_rsentries[0]->rstate->metrics.decode_tokens += lengths[0];
      }

      double elapsed_time;
      {
        NVTXScopedRange nvtx_scope("BatchDecode get time");
        if (SyncDecodeTimingEnabled()) {
          Device device = timing_tensor->device;
          DeviceAPI::Get(device)->StreamSync(device,
                                             DeviceAPI::Get(device)->GetCurrentStream(device));
        }
        auto tend = std::chrono::high_resolution_clock::now();
        elapsed_time = static_cast<double>((tend - tstart).count()) / 1e9;
      }
      estate->metrics.engine_decode_time_sum += elapsed_time;
      estate->metrics.UpdateDecodeTimeByBatchSize(num_rsentries, elapsed_time);

      if (stop_burst) {
        break;
      }
    }

    if (defer_cpu_tokens) {
      NVTXScopedRange nvtx_scope("BatchDecode deferred token commit");
      auto commit_tstart = std::chrono::high_resolution_clock::now();
      DeferredCommitTiming commit_timing =
          CommitDeferredCpuTokens(running_rsentries[0], deferred_token_ids_device,
                                  deferred_token_ids_host, deferred_token_count,
                                  deferred_last_token_device);
      auto commit_tend = std::chrono::high_resolution_clock::now();
      estate->metrics.engine_batch_decode_deferred_commit_time_sum +=
          static_cast<double>((commit_tend - commit_tstart).count()) / 1e9;
      estate->metrics.engine_batch_decode_deferred_commit_copy_sync_time_sum +=
          commit_timing.copy_sync_seconds;
      estate->metrics.engine_batch_decode_deferred_commit_cpu_time_sum +=
          commit_timing.cpu_seconds;
    }
    auto action_tend = std::chrono::high_resolution_clock::now();
    estate->metrics.engine_batch_decode_action_time_sum +=
        static_cast<double>((action_tend - action_tstart).count()) / 1e9;

    return estate->running_queue;
  }

 private:
  void CopyDeviceTokenIdToBuffer(Tensor token_ids_device, Tensor token_ids_buffer, int offset) {
    TVM_FFI_ICHECK_EQ(token_ids_device->ndim, 1);
    TVM_FFI_ICHECK_EQ(token_ids_device->shape[0], 1);
    TVM_FFI_ICHECK(token_ids_device.DataType() == DataType::Int(32));
    TVM_FFI_ICHECK(token_ids_buffer.defined());
    TVM_FFI_ICHECK_EQ(token_ids_buffer->ndim, 1);
    TVM_FFI_ICHECK_GT(token_ids_buffer->shape[0], offset);
    TVM_FFI_ICHECK(token_ids_buffer.DataType() == DataType::Int(32));
    Device device = token_ids_device->device;
    Tensor token_id_dst = token_ids_buffer.CreateView(
        {1}, token_ids_buffer->dtype,
        static_cast<uint64_t>(offset) * token_ids_buffer.DataType().bytes());
    TVMStreamHandle compute_stream = DeviceAPI::Get(device)->GetCurrentStream(device);
    CopyArray(token_ids_device, token_id_dst, compute_stream);
  }

  std::vector<SampleResult> CopyDeviceTokenIdsToSampleResults(Tensor token_ids_device,
                                                              int num_tokens) {
    TVM_FFI_ICHECK(token_ids_device.defined());
    TVM_FFI_ICHECK_EQ(token_ids_device->ndim, 1);
    TVM_FFI_ICHECK_EQ(token_ids_device->shape[0], num_tokens);
    TVM_FFI_ICHECK(token_ids_device.DataType() == DataType::Int(32));
    Device device = token_ids_device->device;
    Tensor token_ids_host =
        Tensor::Empty({num_tokens}, DataType::Int(32), GetPreferredHostDevice(device));
    TVMStreamHandle compute_stream = DeviceAPI::Get(device)->GetCurrentStream(device);
    CopyArray(token_ids_device, token_ids_host, compute_stream);
    DeviceAPI::Get(device)->StreamSync(device, compute_stream);

    const int32_t* token_ids = static_cast<const int32_t*>(token_ids_host->data);
    std::vector<SampleResult> results;
    results.reserve(num_tokens);
    for (int i = 0; i < num_tokens; ++i) {
      SampleResult result{{token_ids[i], 1.0f}, {}};
      result.sampled_token_ids_device = token_ids_device;
      result.sampled_token_ids_device_offset = i;
      results.push_back(std::move(result));
    }
    return results;
  }

  struct DeferredCommitTiming {
    double copy_sync_seconds = 0.0;
    double cpu_seconds = 0.0;
  };

  DeferredCommitTiming CommitDeferredCpuTokens(const RequestStateEntry& rsentry,
                                               Tensor token_ids_device, Tensor token_ids_host,
                                               int token_count, Tensor last_token_ids_device) {
    if (token_count == 0) {
      return {};
    }
    TVM_FFI_ICHECK(token_ids_device.defined());
    TVM_FFI_ICHECK_EQ(token_ids_device->ndim, 1);
    TVM_FFI_ICHECK_GE(token_ids_device->shape[0], token_count);
    TVM_FFI_ICHECK(token_ids_host.defined());
    TVM_FFI_ICHECK_EQ(token_ids_host->ndim, 1);
    TVM_FFI_ICHECK_GE(token_ids_host->shape[0], token_count);
    Device device = last_token_ids_device->device;
    TVMStreamHandle compute_stream = DeviceAPI::Get(device)->GetCurrentStream(device);
    Tensor token_ids_device_view =
        token_ids_device.CreateView({token_count}, token_ids_device->dtype);
    Tensor token_ids_host_view = token_ids_host.CreateView({token_count}, token_ids_host->dtype);
    auto copy_tstart = std::chrono::high_resolution_clock::now();
    CopyArray(token_ids_device_view, token_ids_host_view, compute_stream);
    DeviceAPI::Get(device)->StreamSync(device, compute_stream);
    auto copy_tend = std::chrono::high_resolution_clock::now();

    auto cpu_tstart = std::chrono::high_resolution_clock::now();
    RequestModelState mstate = rsentry->mstates[0];
    TVM_FFI_ICHECK(!mstate->grammar_matcher.has_value());
    const int32_t* token_ids = static_cast<const int32_t*>(token_ids_host->data);
    mstate->committed_tokens.reserve(mstate->committed_tokens.size() + token_count);
    for (int i = 0; i < token_count; ++i) {
      SampleResult result{{token_ids[i], 1.0f}, {}};
      if (i + 1 == token_count) {
        result.sampled_token_ids_device = last_token_ids_device;
        result.sampled_token_ids_device_offset = 0;
      }
      mstate->committed_tokens.push_back(std::move(result));
      rsentry->rstate->metrics.completion_tokens += 1;
    }
    // All but the last deferred token have already been consumed inside the burst.
    mstate->num_tokens_for_next_decode = 1;
    auto cpu_tend = std::chrono::high_resolution_clock::now();
    return {
        static_cast<double>((copy_tend - copy_tstart).count()) / 1e9,
        static_cast<double>((cpu_tend - cpu_tstart).count()) / 1e9,
    };
  }

  std::pair<Tensor, Tensor> GetDeferredTokenBuffers(Device device, int capacity) {
    TVM_FFI_ICHECK_GT(capacity, 0);
    if (!deferred_token_ids_device_workspace_.defined() ||
        deferred_token_ids_device_workspace_->device.device_type != device.device_type ||
        deferred_token_ids_device_workspace_->device.device_id != device.device_id ||
        deferred_token_ids_device_workspace_->shape[0] < capacity) {
      deferred_token_ids_device_workspace_ =
          Tensor::Empty({capacity}, DataType::Int(32), device);
      deferred_token_ids_host_workspace_ =
          Tensor::Empty({capacity}, DataType::Int(32), GetPreferredHostDevice(device));
    }
    return {
        deferred_token_ids_device_workspace_.CreateView({capacity}, DataType::Int(32)),
        deferred_token_ids_host_workspace_.CreateView({capacity}, DataType::Int(32)),
    };
  }

  /*! \brief Return whether this request set can safely run multiple decode steps in one action. */
  int GetSafeDecodeBurstSteps(const std::vector<RequestStateEntry>& rsentries) {
    int burst_steps = DecodeBurstSteps();
    if (burst_steps <= 1) {
      return 1;
    }
    // This is an intentionally narrow fast path. General serving semantics keep one-step decode.
    if (rsentries.size() != 1 || engine_config_->speculative_mode != SpeculativeMode::kDisable) {
      return 1;
    }
    const RequestStateEntry& rsentry = rsentries[0];
    if (rsentry->request->generation_cfg->n != 1 || !rsentry->child_indices.empty()) {
      return 1;
    }
    if (!rsentry->request->generation_cfg->stop_strs.empty() ||
        rsentry->request->generation_cfg->logprobs ||
        rsentry->request->generation_cfg->top_logprobs != 0) {
      return 1;
    }
    RequestModelState mstate = rsentry->mstates[0];
    if (mstate->require_retokenization_in_next_decode || mstate->RequireNextTokenBitmask()) {
      return 1;
    }
    if (rsentry->request->generation_cfg->max_tokens >= 0) {
      int remaining = rsentry->request->generation_cfg->max_tokens -
                      static_cast<int>(mstate->committed_tokens.size());
      burst_steps = std::min(burst_steps, std::max(1, remaining));
    }
    return burst_steps;
  }

  bool CanDeferCpuTokenBurst(const std::vector<RequestStateEntry>& rsentries, int burst_steps) {
    if (!DeferCpuTokenBurstEnabled() || burst_steps <= 1 || !sampler_->SupportsDeviceTokenIds() ||
        !models_[0]->SupportsDeviceTokenEmbed() || rsentries.size() != 1) {
      return false;
    }
    const RequestStateEntry& rsentry = rsentries[0];
    GenerationConfig generation_cfg = rsentry->request->generation_cfg;
    RequestModelState mstate = rsentry->mstates[0];
    return generation_cfg->debug_config.ignore_eos && generation_cfg->stop_strs.empty() &&
           generation_cfg->frequency_penalty == 0.0 && generation_cfg->presence_penalty == 0.0 &&
           generation_cfg->repetition_penalty == 1.0 && !generation_cfg->logprobs &&
           generation_cfg->top_logprobs == 0 && generation_cfg->n == 1 &&
           !mstate->require_retokenization_in_next_decode && !mstate->RequireNextTokenBitmask();
  }

  /*! \brief Return whether an inner decode burst must stop before engine post-processing. */
  bool ShouldStopDecodeBurst(const RequestStateEntry& rsentry, const RequestModelState& mstate) {
    GenerationConfig generation_cfg = rsentry->request->generation_cfg;
    if (mstate->require_retokenization_in_next_decode || mstate->RequireNextTokenBitmask()) {
      return true;
    }
    int num_committed_tokens = static_cast<int>(mstate->committed_tokens.size());
    if (generation_cfg->max_tokens >= 0 && num_committed_tokens >= generation_cfg->max_tokens) {
      return true;
    }
    if (rsentry->request->prompt_tokens + num_committed_tokens >=
        engine_config_->max_single_sequence_length) {
      return true;
    }
    if (!generation_cfg->debug_config.ignore_eos && !mstate->committed_tokens.empty()) {
      int32_t token_id = mstate->committed_tokens.back().GetTokenId();
      return std::any_of(generation_cfg->stop_token_ids.begin(),
                         generation_cfg->stop_token_ids.end(),
                         [token_id](int32_t stop_token_id) { return token_id == stop_token_id; });
    }
    return false;
  }

  /*! \brief Check if the input request state entries can be decoded under conditions. */
  bool CanDecode(int num_rsentries) {
    int num_available_pages = models_[0]->GetNumAvailablePages();
    return num_rsentries <= num_available_pages;
  }

  /*!
   * \brief Retokenize the past tokens with a new token.
   * \param mstate The model state.
   * \param token_id The new token id.
   * \param max_rollback_tokens The maximum number of tokens to rollback.
   * \return The number of tokens to rollback and the new tokens.
   */
  std::pair<int, std::vector<int32_t>> RetokenizeWithNewToken(RequestModelState mstate,
                                                              int32_t token_id,
                                                              int max_rollback_tokens) {
    // Step 1. Get past tokens
    // past_tokens = mstate[-max_rollback_tokens:]
    // past_string = detokenize(past_tokens)
    const auto& token_table = tokenizer_->PostProcessedTokenTable();
    std::vector<int32_t> past_tokens;
    std::string past_string;
    auto past_begin_it = mstate->committed_tokens.size() >= max_rollback_tokens
                             ? mstate->committed_tokens.end() - max_rollback_tokens
                             : mstate->committed_tokens.begin();
    for (auto it = past_begin_it; it != mstate->committed_tokens.end(); ++it) {
      past_tokens.push_back(it->GetTokenId());
      past_string += token_table[it->GetTokenId()];
    }

    // Step 2. Retokenize
    // Compare tokenize(past_string + new_string) and past_tokens
    auto new_tokens = tokenizer_->EncodeNoPrependSpace(past_string + token_table[token_id]);

    int first_differ_idx = past_tokens.size();
    for (int i = 0; i < static_cast<int>(past_tokens.size()); ++i) {
      if (i == static_cast<int>(new_tokens.size()) || past_tokens[i] != new_tokens[i]) {
        first_differ_idx = i;
        break;
      }
    }

    return {past_tokens.size() - first_differ_idx,
            std::vector<int32_t>(new_tokens.begin() + first_differ_idx, new_tokens.end())};
  }

  /*!
   * \brief Commit the token and may retokenize the past tokens.
   * \param rsentry The request state entry.
   * \param mstate The model state.
   * \param sample_result The sampled token.
   */
  void CommitTokenMayRetokenize(RequestStateEntry rsentry, RequestModelState mstate,
                                const SampleResult& sample_result) {
    auto generation_cfg = rsentry->request->generation_cfg;
    // 1. If EOS token is generated, jump commit it
    if (!generation_cfg->debug_config.ignore_eos &&
        std::any_of(generation_cfg->stop_token_ids.begin(), generation_cfg->stop_token_ids.end(),
                    [&](int32_t token) { return token == sample_result.GetTokenId(); })) {
      mstate->CommitToken(sample_result);
      rsentry->rstate->metrics.completion_tokens += 1;
      return;
    }

    // 2. Check retokenization
    const auto& committed_tokens = mstate->committed_tokens;
    auto [rollback_cnt, new_tokens] =
        RetokenizeWithNewToken(mstate, sample_result.GetTokenId(), MAX_ROLLBACK_TOKENS_);

    // 3. Handle output when retokenization happens
    if (rollback_cnt >
        static_cast<int>(committed_tokens.size()) - rsentry->next_callback_token_pos) {
      const auto& token_table = tokenizer_->PostProcessedTokenTable();
      for (auto i = rsentry->next_callback_token_pos; i < committed_tokens.size(); ++i) {
        auto token_id = committed_tokens[i].GetTokenId();
        rsentry->extra_prefix_string += token_table[token_id];
      }
      rsentry->extra_prefix_string += token_table[sample_result.GetTokenId()];
      rsentry->next_callback_token_pos = static_cast<int>(committed_tokens.size()) - rollback_cnt +
                                         static_cast<int>(new_tokens.size());
    }

    if (rollback_cnt > 0) {
      mstate->RollbackTokens(rollback_cnt);
      models_[0]->PopNFromKVCache(mstate->internal_id, rollback_cnt);
    }

    for (auto token_id : new_tokens) {
      mstate->CommitToken({{token_id, 1.0}, {}});
    }

    rsentry->rstate->metrics.completion_tokens +=
        static_cast<int>(new_tokens.size()) - rollback_cnt;
  }

  /*!
   * \brief The model to run decode in. When there are multiple
   * models, the `Step` function of the created action will not take effect.
   */
  Array<Model> models_;
  /*! \brief The tokenizer of the engine. */
  Tokenizer tokenizer_;
  /*! \brief The logit processor. */
  LogitProcessor logit_processor_;
  /*! \brief The sampler to sample new tokens. */
  Sampler sampler_;
  /*! \brief Stable model workspaces owned by the engine and shared with prefill. */
  std::vector<ModelWorkspace> model_workspaces_;
  /*! \brief The engine config. */
  EngineConfig engine_config_;
  /*! \brief Event trace recorder. */
  Optional<EventTraceRecorder> trace_recorder_;
  /*! \brief Reusable token staging buffers for batch-1 decode bursts. */
  Tensor deferred_token_ids_device_workspace_{nullptr};
  Tensor deferred_token_ids_host_workspace_{nullptr};
  /*! \brief The maximum number of tokens to retokenize and may be rolled back. */
  const int MAX_ROLLBACK_TOKENS_ = 10;
};

EngineAction EngineAction::BatchDecode(Array<Model> models, Tokenizer tokenizer,
                                       LogitProcessor logit_processor, Sampler sampler,
                                       std::vector<ModelWorkspace> model_workspaces,
                                       EngineConfig engine_config,
                                       Optional<EventTraceRecorder> trace_recorder) {
  return EngineAction(tvm::ffi::make_object<BatchDecodeActionObj>(
      std::move(models), std::move(tokenizer), std::move(logit_processor), std::move(sampler),
      std::move(model_workspaces), std::move(engine_config), std::move(trace_recorder)));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
