#!/usr/bin/env bash
# End-to-end recovery with permutation sampling (n=6, greedy over shuffled candidate orders) on MMLongBench,
# answered by both readers. Copies the zero-shot run so cached verifier scores, answers and verdicts are reused.
set -uo pipefail
R=/user_mount/home/zhukeyu/datasets/thesis-evidence-recovery
L=$R/logs
A=$R/AD-Reranker/results/MMLongBench
SRC=$A/rp2-top10-mmlb-abl
export RUN_NAME=rp2-top10-mmlb-perm
DST=$A/$RUN_NAME
mkdir -p $DST/recovery_cache
cp -n $SRC/rerank.json $SRC/answers_px1003520.jsonl $SRC/answers_qwen3vl8b_px1003520.jsonl $SRC/judge_adr.jsonl $DST/
cp -n $SRC/recovery_cache/verify_score_noimg.jsonl $SRC/recovery_cache/verify_score_mm.jsonl \
      $SRC/recovery_cache/resample_perm_n6.jsonl $DST/recovery_cache/
cd $R
PLAN='recovery.plan=[none,ret@2,oracle@2,scoreni20@3,score20@3,f50@3,f16@3,"scoreni20+f50@3","scoreni20+f16@3","score20+f50@3","score20+f16@3"]'
bash run_rp2_h200.sh recover "$PLAN" +recovery.resample_file=resample_perm_n6.jsonl > $L/perm_recover.log 2>&1 \
  || { echo "perm recover failed"; exit 1; }
echo "$(date) perm recover finished"
J=(judge.prompt=adr judge.thinking=false judge.max_tokens=256 +judge.out=judge_adr.jsonl judge.concurrency=128 +judge.batch=2048)
for reader in qwen25vl7b:8101 qwen3vl8b:8105; do
  NAME=${reader%:*}; PORT=${reader#*:}
  RD=(models.answer.all_pixels=1003520 models.answer.vllm_base_url=http://127.0.0.1:$PORT/v1 models.answer.vllm_model=$NAME)
  bash run_rp2_h200.sh answer "${RD[@]}" > $L/perm_answer_$NAME.log 2>&1 || { echo "perm answer $NAME failed"; exit 1; }
  for sp in dev test; do
    bash run_rp2_h200.sh eval "${J[@]}" "${RD[@]}" +split=$sp +judge.eval_out=eval_${NAME}_$sp.json > $L/perm_eval_${NAME}_$sp.log 2>&1 \
      || { echo "perm eval $NAME $sp failed"; exit 1; }
  done
  echo "$(date) perm $NAME finished"
done
echo "$(date) PERM_DONE"
