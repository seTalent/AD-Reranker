#!/usr/bin/env bash
# MMLongBench group-sampling experiments (VLM on 8101):
#  1. trained AD-Reranker (LoRA "adr"): n=6, T=0.7 on its own reranked candidates
#  2. zero-shot reranker: sample size / temperature (n=10 at T=0.7, 1.0, 0.3) and permutation sampling (n=6)
set -uo pipefail
R=/user_mount/home/zhukeyu/datasets/thesis-evidence-recovery
L=$R/logs
cd $R
RUN_NAME=rp2-top10-mmlb-adr bash run_rp2_h200.sh resample models.refiner.vllm_model=adr > $L/samp_adr.log 2>&1 \
  || echo "adr resample failed"
echo "$(date) adr resample finished"
export RUN_NAME=rp2-top10-mmlb-abl
for t in 0.7 1.0 0.3; do
  bash run_rp2_h200.sh resample +resample_n=10 +resample_t=$t > $L/samp_n10_t$t.log 2>&1 || echo "n10 t$t failed"
  echo "$(date) n10 t$t finished"
done
bash run_rp2_h200.sh resample +resample_n=6 +resample_perm=true > $L/samp_perm6.log 2>&1 || echo "perm failed"
echo "$(date) SAMPLING_DONE"
