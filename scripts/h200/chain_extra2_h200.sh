#!/usr/bin/env bash
# Steps 3-4 of chain_extra_h200.sh after fixing the variant names: insertion ablation and breakdowns.
set -uo pipefail
R=/user_mount/home/zhukeyu/datasets/thesis-evidence-recovery
L=$R/logs
A=$R/AD-Reranker/results/MMLongBench
cd $R
HI=(models.answer.all_pixels=1003520)
J=(judge.prompt=adr judge.thinking=false judge.max_tokens=256 +judge.out=judge_adr.jsonl judge.concurrency=128 +judge.batch=2048)

export RUN_NAME=rp2-top10-mmlb-ins
SRC=$A/rp2-top10-mmlb-abl
mkdir -p $A/$RUN_NAME/recovery_cache
cp $SRC/rerank.json $SRC/answers_px1003520.jsonl $SRC/judge_adr.jsonl $A/$RUN_NAME/
cp $SRC/recovery_cache/verify_score_mm.jsonl $A/$RUN_NAME/recovery_cache/
bash run_rp2_h200.sh recover "recovery.plan=[none,ret@2,oracle@2,score10@3,score10_tail@3,score20@3,score20_tail@3]" \
  > $L/${RUN_NAME}_recover.log 2>&1 || { echo "ins recover failed"; exit 1; }
bash run_rp2_h200.sh answer "${HI[@]}" > $L/${RUN_NAME}_answer_hi.log 2>&1 || { echo "ins answer failed"; exit 1; }
for sp in dev test all; do
  S=(); [ $sp != all ] && S=(+split=$sp)
  bash run_rp2_h200.sh eval "${J[@]}" "${HI[@]}" "${S[@]}" +judge.eval_out=eval_px1003520_$sp.json \
    > $L/${RUN_NAME}_eval_$sp.log 2>&1 || { echo "ins eval failed"; exit 1; }
done
echo "$(date) insertion finished"

BV="+breakdown_variants=[rerank,ret@2,score20@3,oracle@2,top10]"
for RN in rp2-top10-mmlb-abl rp2-top10-mmlb-adr; do
  export RUN_NAME=$RN
  for sp in test all; do
    S=(); [ $sp != all ] && S=(+split=$sp)
    bash run_rp2_h200.sh breakdown "${J[@]}" "${HI[@]}" "${S[@]}" "$BV" > $L/${RN}_breakdown_$sp.log 2>&1 \
      || { echo "breakdown $RN failed"; exit 1; }
  done
done
echo "$(date) EXTRA_DONE"
