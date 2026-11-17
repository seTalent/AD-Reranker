#!/usr/bin/env bash
# MMLongBench preprocessing with the unmodified MDocAgent scripts (H200).
#   bash scripts/prepare_mdocagent.sh <gpu>
# 1) samples.json = MMLongBench-Doc-V2 data/samples.json (same schema as V1)
# 2) extract.py: PyMuPDF pages -> tmp/MMLongBench/{doc}_{i}.png (144 dpi) and .txt
# 3) retrieve.py retrieval=image: ColPali top-10 -> data/MMLongBench/sample-with-retrieval-results.json
set -euo pipefail
GPU=${1:-0}
ROOT=${ERC_ROOT:-/user_mount/home/zhukeyu/datasets/thesis-evidence-recovery}
PY=$ROOT/envs/mdocagent/bin/python
cd "$ROOT/MDocAgent"
cp "$ROOT/data/data/samples.json" data/MMLongBench/samples.json
sha256sum data/MMLongBench/samples.json
$PY scripts/extract.py --config-name mmlb   # skips pages already extracted
ls tmp/MMLongBench/*.png | wc -l
HF_HOME=$ROOT/hf HF_ENDPOINT=https://hf-mirror.com $PY scripts/retrieve.py --config-name mmlb retrieval=image \
  retrieval.cuda_visible_devices="'$GPU'"  # quoted: os.environ needs a str
