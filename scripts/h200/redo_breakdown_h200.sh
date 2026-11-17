#!/usr/bin/env bash
# Re-run breakdowns that were written empty because the judge file was not passed (offline, CPU only).
set -uo pipefail
R=/user_mount/home/zhukeyu/datasets/thesis-evidence-recovery
L=$R/logs
cd $R
J=(+judge.out=judge_adr.jsonl)
BV='+breakdown_variants=[rerank,ret@2,f50@3,scoreni20@3,"scoreni20+f50@3",score20@3,"score20+f50@3",oracle@2,top10]'
run() {  # <config> <run name> <reader name> <port>
  local RD=(models.answer.all_pixels=1003520)
  [ "$3" != qwen25vl7b ] && RD+=(models.answer.vllm_model=$3 models.answer.vllm_base_url=http://127.0.0.1:$4/v1)
  CONFIG=$1 RUN_NAME=$2 bash run_rp2_h200.sh breakdown "${J[@]}" "${RD[@]}" +split=test "$BV" \
    > $L/redo_breakdown_$2_$3.log 2>&1 && echo "ok $1 $2 $3" || echo "FAILED $1 $2 $3"
}
run mmlb rp2-top10-mmlb-abl qwen25vl7b 8101
run mmlb rp2-top10-mmlb-abl qwen3vl8b 8105
run mmlb rp2-top10-mmlb-adr qwen25vl7b 8101
run mmlb rp2-top10-mmlb-adr qwen3vl8b 8105
run ldu rp2-top10-ldu qwen25vl7b 8101
echo REDO_DONE
