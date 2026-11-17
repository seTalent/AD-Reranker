#!/usr/bin/env bash
# Reader generalisation on MMLongBench: the same recovered page sets answered by another reader.
#   bash chain_reader_h200.sh <served name> <port>      e.g. qwen3vl8b 8105 | qwen25vl32b 8106
set -uo pipefail
R=/user_mount/home/zhukeyu/datasets/thesis-evidence-recovery
L=$R/logs
NAME=$1; PORT=$2
cd $R
until curl -s -m 5 http://127.0.0.1:$PORT/v1/models | grep -q "\"$NAME\""; do sleep 20; done
export RUN_NAME=${RUN_NAME:-rp2-top10-mmlb-abl}; T=${RUN_NAME#rp2-top10-}
RD=(models.answer.all_pixels=1003520 models.answer.vllm_base_url=http://127.0.0.1:$PORT/v1 models.answer.vllm_model=$NAME)
V='+variants=[top4,top10,rerank,ret@2,oracle@2,f50@3,scoreni20@3,score20@3,"scoreni20+f50@3","score20+f50@3"]'
bash run_rp2_h200.sh answer "${RD[@]}" "$V" > $L/reader_${T}_${NAME}_answer.log 2>&1 || { echo "$NAME answer failed"; exit 1; }
echo "$(date) $NAME answer finished"
J=(judge.prompt=adr judge.thinking=false judge.max_tokens=256 +judge.out=judge_adr.jsonl judge.concurrency=128 +judge.batch=2048)
for sp in dev test all; do
  S=(); [ $sp != all ] && S=(+split=$sp)
  bash run_rp2_h200.sh eval "${J[@]}" "${RD[@]}" "${S[@]}" +judge.eval_out=eval_${NAME}_$sp.json > $L/reader_${T}_${NAME}_eval_$sp.log 2>&1 \
    || { echo "$NAME eval $sp failed"; exit 1; }
done
bash run_rp2_h200.sh breakdown "${RD[@]}" +split=test \
  "+breakdown_variants=[rerank,scoreni20@3,\"scoreni20+f50@3\",score20@3,top10]" > $L/reader_${T}_${NAME}_breakdown.log 2>&1
echo "$(date) READER_${T}_${NAME}_DONE"
