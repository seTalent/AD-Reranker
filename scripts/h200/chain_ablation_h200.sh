#!/usr/bin/env bash
# Component ablations of gap-conditioned page scoring (high-resolution reader), thresholds chosen per
# ablation on dev; then a sequential latency probe on otherwise idle servers.
set -uo pipefail
R=/user_mount/home/zhukeyu/datasets/thesis-evidence-recovery
L=$R/logs
A=$R/AD-Reranker
SRC=$A/results/MMLongBench/rp2-top10-mmlb-exp
export RUN_NAME=rp2-top10-mmlb-abl
DST=$A/results/MMLongBench/$RUN_NAME
cd $R
mkdir -p $DST/recovery_cache
cp $SRC/rerank.json $SRC/answers_px1003520.jsonl $SRC/judge_adr.jsonl $DST/
cp $SRC/recovery_cache/verify_*.jsonl $DST/recovery_cache/
PLAN="recovery.plan=[none,ret@2,oracle@2"
for m in score scorenk scorenr scoreni; do for t in 10 20 30 50; do PLAN="$PLAN,$m$t@3"; done; done
PLAN="$PLAN]"
HI=(models.answer.all_pixels=1003520)
bash run_rp2_h200.sh recover "$PLAN" > $L/abl_recover.log 2>&1 || { echo "abl recover failed"; exit 1; }
echo "$(date) abl recover finished"
bash run_rp2_h200.sh answer "${HI[@]}" > $L/abl_answer.log 2>&1 || { echo "abl answer failed"; exit 1; }
echo "$(date) abl answer finished"
J=(judge.prompt=adr judge.thinking=false judge.max_tokens=256 +judge.out=judge_adr.jsonl judge.concurrency=128 +judge.batch=2048)
for sp in dev test all; do
  S=(); [ $sp != all ] && S=(+split=$sp)
  bash run_rp2_h200.sh eval "${J[@]}" "${HI[@]}" "${S[@]}" +judge.eval_out=eval_$sp.json > $L/abl_eval_$sp.log 2>&1 \
    || { echo "abl eval $sp failed"; exit 1; }
done
echo "$(date) abl eval finished"
bash run_rp2_h200.sh probe "${HI[@]}" +probe_n=100 "+probe_modes=[score_mm]" "+probe_variants=[rerank,score20@3,top10]" \
  > $L/abl_probe.log 2>&1 || { echo "abl probe failed"; exit 1; }
bash run_rp2_h200.sh eval "${J[@]}" "${HI[@]}" +judge.eval_out=eval_all.json > $L/abl_eval_final.log 2>&1
echo "$(date) ABL_DONE"
