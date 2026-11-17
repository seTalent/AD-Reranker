#!/usr/bin/env bash
# Improvement experiments on MMLongBench, sharing the baseline run's reranking and caches:
# gap-conditioned page scoring (scoreT@b) and a high-resolution reader. Dev/test reported separately.
set -uo pipefail
R=/user_mount/home/zhukeyu/datasets/thesis-evidence-recovery
L=$R/logs
A=$R/AD-Reranker
BASE=$A/results/MMLongBench/rp2-top10-mmlb
export RUN_NAME=rp2-top10-mmlb-exp
EXP=$A/results/MMLongBench/$RUN_NAME
cd $R
until grep -a -q "recover finished" $L/chain_mmlb.log; do sleep 20; done
mkdir -p $EXP/recovery_cache
cp $BASE/rerank.json $EXP/
cp $BASE/recovery_cache/verify_*.jsonl $EXP/recovery_cache/
PLAN="recovery.plan=[none,ret@1,ret@2,ret@3,ours@2,ours_all@2,oracle@1,oracle@2,oracle@3"
for t in 10 20 30 50 70; do PLAN="$PLAN,score$t@1,score$t@2,score$t@3"; done
PLAN="$PLAN]"
bash run_rp2_h200.sh recover "$PLAN" > $L/exp_recover.log 2>&1 || { echo "exp recover failed"; exit 1; }
echo "$(date) exp recover finished"
until grep -a -q -E "answer finished|failed" $L/chain_mmlb.log; do sleep 20; done
cp $BASE/answers.jsonl $EXP/answers.jsonl
bash run_rp2_h200.sh answer > $L/exp_answer.log 2>&1 || { echo "exp answer failed"; exit 1; }
echo "$(date) exp answer (512 tokens/page) finished"
bash run_rp2_h200.sh answer models.answer.all_pixels=1003520 > $L/exp_answer_hires.log 2>&1 || { echo "exp hires answer failed"; exit 1; }
echo "$(date) exp answer (1280 tokens/page) finished"
until grep -a -q -E "eval finished|failed" $L/chain_mmlb.log; do sleep 20; done
cp -n $BASE/judge_adr.jsonl $EXP/judge_adr.jsonl 2>/dev/null
J=(judge.prompt=adr judge.thinking=false judge.max_tokens=256 +judge.out=judge_adr.jsonl judge.concurrency=128 +judge.batch=2048)
for px in 401408 1003520; do
  for sp in dev test; do
    bash run_rp2_h200.sh eval "${J[@]}" models.answer.all_pixels=$px +split=$sp +judge.eval_out=eval_px${px}_$sp.json \
      > $L/exp_eval_px${px}_$sp.log 2>&1 || { echo "exp eval $px $sp failed"; exit 1; }
  done
done
echo "$(date) EXP_DONE"
