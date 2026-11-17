#!/usr/bin/env bash
# Judge all responses without thinking, calibrate on a fixed 600-response subset with thinking,
# then recompute metrics with the agreement. Judge server: 8103.
set -uo pipefail
R=/user_mount/home/zhukeyu/datasets/thesis-evidence-recovery
L=$R/logs
cd $R
NOTHINK=(judge.thinking=false judge.max_tokens=512 +judge.out=judge_nothink.jsonl judge.concurrency=128 +judge.batch=2048)
bash run_rp2_h200.sh eval "${NOTHINK[@]}" > $L/rp2_eval_nothink.log 2>&1 || { echo "nothink failed"; exit 1; }
echo "$(date) nothink finished"
bash run_rp2_h200.sh eval +judge.subset=600 +judge.out=judge_think600.jsonl judge.concurrency=64 +judge.batch=600 \
  > $L/rp2_eval_think600.log 2>&1 || { echo "think600 failed"; exit 1; }
echo "$(date) think600 finished"
bash run_rp2_h200.sh eval "${NOTHINK[@]}" +judge.compare_with=judge_think600.jsonl > $L/rp2_eval_final.log 2>&1
bash run_rp2_h200.sh report > $L/rp2_report.md 2>&1
echo "$(date) ALL_DONE"
