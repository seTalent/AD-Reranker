#!/usr/bin/env bash
# MDocAgent environment (its install.sh), built under the project directory on H200.
# Differences, all forced by the current package index / network:
#  - transformers @51ed61e2 is installed from the commit archive (a full git clone of the
#    repository kept timing out); it reports version 4.50.0.dev0.
#  - sentence-transformers / mteb==1.2 are installed with --no-deps (pinned
#    sentence-transformers==3.4.1), otherwise pip replaces the pinned transformers with 5.x.
#  - ragatouille (ColBERT text retrieval) is not installed: only ColPali image retrieval is used.
set -euo pipefail
ROOT=${ERC_ROOT:-/user_mount/home/zhukeyu/datasets/thesis-evidence-recovery}
export CONDA_PKGS_DIRS=$ROOT/envs/pkgs PIP_CACHE_DIR=$ROOT/envs/pipcache
export PIP_INDEX_URL=${PIP_INDEX_URL:-https://mirrors.ustc.edu.cn/pypi/simple}
E=$ROOT/envs/mdocagent/bin
[ -x "$E/python" ] || /opt/conda/bin/conda create -y -p "$ROOT/envs/mdocagent" python=3.12
$E/pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
A=$ROOT/envs/transformers-51ed61e2.tar.gz
[ -s "$A" ] || curl -L --fail --retry 5 -o "$A" \
  https://codeload.github.com/huggingface/transformers/tar.gz/51ed61e2f05176f81fa7c9decba10cc28e138f61
$E/pip install "$A"
$E/pip install accelerate qwen_vl_utils argparse matplotlib pymupdf ipywidgets
$E/pip install pandas seaborn openai scikit-learn
$E/pip install hydra-core
$E/pip install colpali_engine==0.1.1
$E/pip install --no-deps "sentence-transformers==3.4.1" "mteb==1.2"
$E/pip install "datasets>=2.2.0,<4" jsonlines "requests>=2.26.0" "scikit_learn>=1.0.2" scipy tqdm rich pytrec_eval pydantic
$E/pip freeze > "$ROOT/logs/mdocagent_freeze.txt"
$E/python -c "import transformers, torch, mteb, pymupdf; from colpali_engine.trainer.retrieval_evaluator import CustomEvaluator; print('VERSIONS', transformers.__version__, torch.__version__, mteb.__version__)"
echo INSTALL_DONE
