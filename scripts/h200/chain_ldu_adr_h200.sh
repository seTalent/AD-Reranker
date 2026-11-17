#!/usr/bin/env bash
# LongDocURL with the trained AD-Reranker (point 1, LoRA "adr"): rerank -> group sampling -> recovery ->
# both readers -> judge -> breakdown. Same thresholds as everywhere else (tau 0.2, f 0.5, b 3).
set -uo pipefail
R=/user_mount/home/zhukeyu/datasets/thesis-evidence-recovery
L=$R/logs
A=$R/AD-Reranker/results/LongDocURL
cd $R
export CONFIG=ldu RUN_NAME=rp2-top10-ldu-adr
mkdir -p $A/$RUN_NAME
cp -n $A/rp2-top10-ldu/judge_adr.jsonl $A/$RUN_NAME/   # judge cache is keyed by question and response
ADR=(models.refiner.vllm_model=adr)
bash run_rp2_h200.sh rerank "${ADR[@]}" > $L/ldu_adr_rerank.log 2>&1 || { echo "ldu adr rerank failed"; exit 1; }
echo "$(date) rerank finished"
bash run_rp2_h200.sh resample "${ADR[@]}" > $L/ldu_adr_resample.log 2>&1 || { echo "ldu adr resample failed"; exit 1; }
echo "$(date) resample finished"
PLAN='recovery.plan=[none,ret@2,oracle@2,f50@3,score20@3,scoreni20@3,"scoreni20+f50@3","score20+f50@3"]'
bash run_rp2_h200.sh recover "$PLAN" > $L/ldu_adr_recover.log 2>&1 || { echo "ldu adr recover failed"; exit 1; }
echo "$(date) recover finished"
J=(judge.prompt=adr judge.thinking=false judge.max_tokens=256 +judge.out=judge_adr.jsonl judge.concurrency=128 +judge.batch=2048)
BV='+breakdown_variants=[rerank,ret@2,scoreni20@3,score20@3,"score20+f50@3",oracle@2,top10]'
for reader in qwen25vl7b:8101 qwen3vl8b:8105; do
  NAME=${reader%:*}; PORT=${reader#*:}
  RD=(models.answer.all_pixels=1003520 models.answer.vllm_base_url=http://127.0.0.1:$PORT/v1 models.answer.vllm_model=$NAME)
  bash run_rp2_h200.sh answer "${RD[@]}" > $L/ldu_adr_answer_$NAME.log 2>&1 || { echo "ldu adr answer $NAME failed"; exit 1; }
  for sp in dev test all; do
    S=(); [ $sp != all ] && S=(+split=$sp)
    bash run_rp2_h200.sh eval "${J[@]}" "${RD[@]}" "${S[@]}" +judge.eval_out=eval_${NAME}_$sp.json > $L/ldu_adr_eval_${NAME}_$sp.log 2>&1 \
      || { echo "ldu adr eval $NAME $sp failed"; exit 1; }
  done
  bash run_rp2_h200.sh breakdown "${J[@]}" "${RD[@]}" +split=test "$BV" > $L/ldu_adr_breakdown_$NAME.log 2>&1 \
    || echo "ldu adr breakdown $NAME failed"
  echo "$(date) $NAME finished"
done
echo "$(date) LDU_ADR_DONE"
