#!/usr/bin/env bash
# Zero-sampling reranker uncertainty (recovery/logit.py) for both datasets and both rerankers.
# Server: VLM + AD-Reranker LoRA on 8101.
set -uo pipefail
R=/user_mount/home/zhukeyu/datasets/thesis-evidence-recovery
L=$R/logs
cd $R
rm -f $R/AD-Reranker/results/MMLongBench/rp2-top10-mmlb-abl/recovery_cache/logit_smoke.jsonl
run() {  # <config> <run name> [overrides]
  local C=$1 N=$2; shift 2
  CONFIG=$C RUN_NAME=$N bash run_rp2_h200.sh logit "$@" > $L/logit_$N.log 2>&1 && echo "$(date) $N finished" \
    || echo "$(date) $N failed"
}
run mmlb rp2-top10-mmlb-abl
run mmlb rp2-top10-mmlb-adr models.refiner.vllm_model=adr
run ldu rp2-top10-ldu
run ldu rp2-top10-ldu-adr models.refiner.vllm_model=adr
echo "$(date) LOGIT_DONE"
