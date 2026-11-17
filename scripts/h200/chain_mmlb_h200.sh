#!/usr/bin/env bash
# MMLongBench (MDocAgent/AD-Reranker version, 1,073 questions): full research-point-2 run.
# Servers: vlm 8101 + verifier 8102 (GPU 2), judge 8103 (GPU 3). Judge = the repo's binary-correctness prompt.
set -uo pipefail
R=/user_mount/home/zhukeyu/datasets/thesis-evidence-recovery
L=$R/logs
A=$R/AD-Reranker
export RUN_NAME=rp2-top10-mmlb
RD=$A/results/MMLongBench/$RUN_NAME
cd $R
while pgrep -u "$USER" -f "[s]cripts/retrieve.py" > /dev/null; do sleep 15; done
grep -a -q "Save retrieval results" $L/adr_retrieve_mmlb.log || { echo "retrieve failed"; exit 1; }
echo "$(date) retrieve finished"
mkdir -p $RD/recovery_cache
cp -n $A/results/MMLongBench/rp2-top10-v2/recovery_cache/verify_*.jsonl $RD/recovery_cache/ 2>/dev/null
bash run_rp2_h200.sh rerank  > $L/mmlb_rerank.log 2>&1  || { echo "rerank failed"; exit 1; }
echo "$(date) rerank finished"
bash run_rp2_h200.sh recover > $L/mmlb_recover.log 2>&1 || { echo "recover failed"; exit 1; }
echo "$(date) recover finished"
bash run_rp2_h200.sh answer  > $L/mmlb_answer.log 2>&1  || { echo "answer failed"; exit 1; }
echo "$(date) answer finished"
bash run_rp2_h200.sh probe +probe_n=100 > $L/mmlb_probe.log 2>&1 &
PROBE=$!
JUDGE=(judge.prompt=adr judge.thinking=false judge.max_tokens=256 +judge.out=judge_adr.jsonl
       judge.concurrency=128 +judge.batch=2048)
bash run_rp2_h200.sh eval "${JUDGE[@]}" > $L/mmlb_eval.log 2>&1 || { echo "eval failed"; exit 1; }
echo "$(date) eval finished"
wait $PROBE; echo "$(date) probe finished (exit $?)"
bash run_rp2_h200.sh eval "${JUDGE[@]}" > $L/mmlb_eval_final.log 2>&1
bash run_rp2_h200.sh report "${JUDGE[@]}" > $L/mmlb_report.md 2>&1
echo "$(date) ALL_DONE"
