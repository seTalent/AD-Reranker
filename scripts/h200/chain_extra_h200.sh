#!/usr/bin/env bash
# Additional experiments: trained AD-Reranker (LoRA "adr"), top-4 candidate pool, insertion ablation,
# and metric breakdowns. Servers: vlm+adr 8101 / verifier 8102 (GPU 2), judge 8103 / verifier_mm 8104 (GPU 3).
set -uo pipefail
R=/user_mount/home/zhukeyu/datasets/thesis-evidence-recovery
L=$R/logs
A=$R/AD-Reranker/results/MMLongBench
cd $R
HI=(models.answer.all_pixels=1003520)
J=(judge.prompt=adr judge.thinking=false judge.max_tokens=256 +judge.out=judge_adr.jsonl judge.concurrency=128 +judge.batch=2048)
SCORE="score10@3,score20@3,score30@3,score50@3"
evals() {  # run evals for dev/test/all at the given resolution(s)
  for px in "$@"; do for sp in dev test all; do
    S=(); [ $sp != all ] && S=(+split=$sp)
    bash run_rp2_h200.sh eval "${J[@]}" models.answer.all_pixels=$px "${S[@]}" +judge.eval_out=eval_px${px}_$sp.json \
      > $L/${RUN_NAME}_eval_${px}_$sp.log 2>&1 || return 1
  done; done
}
run() {  # name, stage args...
  local stage=$1; shift
  bash run_rp2_h200.sh $stage "$@" > $L/${RUN_NAME}_$stage.log 2>&1
}

# 1) trained AD-Reranker as the reranker
export RUN_NAME=rp2-top10-mmlb-adr
mkdir -p $A/$RUN_NAME && cp -n $A/rp2-top10-mmlb-exp/judge_adr.jsonl $A/$RUN_NAME/ 2>/dev/null
run rerank models.refiner.vllm_model=adr || { echo "adr rerank failed"; exit 1; }
run recover "recovery.plan=[none,ret@2,oracle@2,$SCORE]" || { echo "adr recover failed"; exit 1; }
run answer || { echo "adr answer failed"; exit 1; }
bash run_rp2_h200.sh answer "${HI[@]}" > $L/${RUN_NAME}_answer_hi.log 2>&1 || { echo "adr hires answer failed"; exit 1; }
evals 401408 1003520 || { echo "adr eval failed"; exit 1; }
echo "$(date) adr finished"

# 2) top-4 candidate pool with the zero-shot reranker
export RUN_NAME=rp2-top4-mmlb
mkdir -p $A/$RUN_NAME && cp -n $A/rp2-top10-mmlb-exp/judge_adr.jsonl $A/$RUN_NAME/ 2>/dev/null
run rerank dataset.top_k=4 || { echo "top4 rerank failed"; exit 1; }
run recover dataset.top_k=4 "recovery.plan=[none,ret@1,ret@2,oracle@2,$SCORE]" || { echo "top4 recover failed"; exit 1; }
bash run_rp2_h200.sh answer dataset.top_k=4 "${HI[@]}" > $L/${RUN_NAME}_answer_hi.log 2>&1 || { echo "top4 answer failed"; exit 1; }
for sp in dev test all; do
  S=(); [ $sp != all ] && S=(+split=$sp)
  bash run_rp2_h200.sh eval dataset.top_k=4 "${J[@]}" "${HI[@]}" "${S[@]}" +judge.eval_out=eval_px1003520_$sp.json \
    > $L/${RUN_NAME}_eval_$sp.log 2>&1 || { echo "top4 eval failed"; exit 1; }
done
echo "$(date) top4 finished"

# 3) insertion ablation on the main zero-shot run
export RUN_NAME=rp2-top10-mmlb-ins
SRC=$A/rp2-top10-mmlb-abl
mkdir -p $A/$RUN_NAME/recovery_cache
cp $SRC/rerank.json $SRC/answers_px1003520.jsonl $SRC/judge_adr.jsonl $A/$RUN_NAME/
cp $SRC/recovery_cache/verify_score_mm.jsonl $A/$RUN_NAME/recovery_cache/
run recover "recovery.plan=[none,ret@2,oracle@2,score10@3,score10@3_tail,score20@3,score20@3_tail]" || { echo "ins recover failed"; exit 1; }
bash run_rp2_h200.sh answer "${HI[@]}" > $L/${RUN_NAME}_answer_hi.log 2>&1 || { echo "ins answer failed"; exit 1; }
for sp in dev test all; do
  S=(); [ $sp != all ] && S=(+split=$sp)
  bash run_rp2_h200.sh eval "${J[@]}" "${HI[@]}" "${S[@]}" +judge.eval_out=eval_px1003520_$sp.json \
    > $L/${RUN_NAME}_eval_$sp.log 2>&1 || { echo "ins eval failed"; exit 1; }
done
echo "$(date) insertion finished"

# 4) breakdowns
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
