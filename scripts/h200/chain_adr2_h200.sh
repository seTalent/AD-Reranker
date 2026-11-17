#!/usr/bin/env bash
# Trained AD-Reranker (point 1) + group sampling: recovery with the text verifier and selection frequency,
# answers by the default reader (1,280 tokens/page), judge. Waits for the ADR sampling in chain_sampling_h200.sh.
set -uo pipefail
R=/user_mount/home/zhukeyu/datasets/thesis-evidence-recovery
L=$R/logs
cd $R
until grep -q "adr resample finished" $L/chain_sampling.out; do sleep 30; done
export RUN_NAME=rp2-top10-mmlb-adr
PLAN='recovery.plan=[none,ret@2,oracle@2,score10@3,score20@3,score30@3,score50@3,scoreni10@3,scoreni20@3,scoreni30@3,scoreni50@3,f50@3,"scoreni20+f50@3","score20+f50@3"]'
bash run_rp2_h200.sh recover "$PLAN" > $L/adr2_recover.log 2>&1 || { echo "adr recover failed"; exit 1; }
echo "$(date) adr recover finished"
HI=(models.answer.all_pixels=1003520)
bash run_rp2_h200.sh answer "${HI[@]}" > $L/adr2_answer.log 2>&1 || { echo "adr answer failed"; exit 1; }
echo "$(date) adr answer finished"
J=(judge.prompt=adr judge.thinking=false judge.max_tokens=256 +judge.out=judge_adr.jsonl judge.concurrency=128 +judge.batch=2048)
for sp in dev test all; do
  S=(); [ $sp != all ] && S=(+split=$sp)
  bash run_rp2_h200.sh eval "${J[@]}" "${HI[@]}" "${S[@]}" +judge.eval_out=eval_px1003520_$sp.json > $L/adr2_eval_$sp.log 2>&1 \
    || { echo "adr eval $sp failed"; exit 1; }
done
echo "$(date) ADR2_DONE"
