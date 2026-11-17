#!/usr/bin/env bash
# H200 launcher for the AD-Reranker research-point-2 experiments.
#   [CONFIG=mmlb|ldu] [RUN_NAME=...] bash run_rp2_h200.sh <stage> [extra hydra overrides]
# stage: extract | retrieve | check | rerank | recover | answer | eval | probe | sense | breakdown | report
set -euo pipefail
R=/user_mount/home/zhukeyu/datasets/thesis-evidence-recovery
M=/user_mount/home/zhukeyu/datasets/group-memory-pilot/models
PY=$R/envs/mdocagent/bin/python
CFG=${CONFIG:-mmlb}
STAGE=$1; shift
cd $R/AD-Reranker
COMMON=(--config-name $CFG run-name=${RUN_NAME:-rp2-top10} dataset.top_k=10
        models.refiner.model_id=$M/Qwen2.5-VL-7B-Instruct models.answer.model_id=$M/Qwen2.5-VL-7B-Instruct
        models.refiner.backend=${BACKEND:-vllm} models.answer.backend=${BACKEND:-vllm}
        recovery.mineru_dir=$R/data/documents)
case $STAGE in
  extract)
    $PY scripts/extract.py --config-name $CFG "$@" ;;
  retrieve)
    HF_HOME=$R/hf HF_ENDPOINT=https://hf-mirror.com $PY scripts/retrieve.py --config-name $CFG retrieval=image \
      retrieval.cuda_visible_devices="'${GPU:-0}'" "$@" ;;
  check)
    CUDA_VISIBLE_DEVICES=${GPU:-0} $PY scripts/rp2_check_backends.py "${COMMON[@]}" +n=${N:-48} run-name=rp2-backend-check "$@" ;;
  *)
    $PY scripts/rp2.py "${COMMON[@]}" +stage=$STAGE "$@" ;;
esac
