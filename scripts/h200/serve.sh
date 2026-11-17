#!/usr/bin/env bash
# Local model servers (vLLM, OpenAI-compatible). No external API is used.
#   bash scripts/serve.sh vlm|verifier|judge <gpu> <gpu_memory_utilization> [kv_cache_bytes]
# vlm      : Qwen2.5-VL-7B-Instruct, reranker and reader (AD-Reranker setting, 512 visual tokens/page)
# verifier : Qwen3.5-9B, text-only information-gap verifier
# judge    : Qwen3.5-27B, MMLongBench-Doc-V2 semantic judge
# The GPUs are shared: an explicit KV-cache size avoids vLLM attributing other users'
# concurrent allocations to this process during memory profiling.
set -euo pipefail
ROLE=$1; GPU=$2; UTIL=${3:-0.40}; KV=${4:-}
ROOT=${ERC_ROOT:-/user_mount/home/zhukeyu/datasets/thesis-evidence-recovery}
MODELS=/user_mount/home/zhukeyu/datasets/group-memory-pilot/models
VLLM=/user_mount/home/zhukeyu/spd/envs/infer/bin/vllm
mkdir -p "$ROOT/logs"
export CUDA_VISIBLE_DEVICES=$GPU VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1
export PATH=$(dirname "$VLLM"):$PATH      # ninja for FlashInfer JIT kernels
EXTRA=()
[ -n "$KV" ] && EXTRA=(--kv-cache-memory-bytes "$KV")
case $ROLE in
  vlm)
    exec $VLLM serve $MODELS/Qwen2.5-VL-7B-Instruct --served-model-name qwen25vl7b --port 8101 \
      --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization $UTIL --max-num-seqs 64 \
      --limit-mm-per-prompt '{"image": 12}' \
      --mm-processor-kwargs '{"min_pixels": 401408, "max_pixels": 401408}' \
      --allowed-local-media-path "$ROOT" --seed 0 "${EXTRA[@]}" ;;
  verifier)
    exec $VLLM serve $MODELS/Qwen3.5-9B --served-model-name qwen35-9b --port 8102 \
      --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization $UTIL --max-num-seqs 64 \
      --limit-mm-per-prompt '{"image": 0, "video": 0}' --reasoning-parser qwen3 --gdn-prefill-backend triton --seed 0 "${EXTRA[@]}" ;;
  judge)
    exec $VLLM serve $MODELS/Qwen3.5-27B --served-model-name qwen35-27b --port 8103 \
      --dtype bfloat16 --max-model-len 20480 --gpu-memory-utilization $UTIL --max-num-seqs 64 \
      --limit-mm-per-prompt '{"image": 0, "video": 0}' --reasoning-parser qwen3 --gdn-prefill-backend triton --seed 0 "${EXTRA[@]}" ;;
  *) echo "unknown role $ROLE"; exit 1 ;;
esac
