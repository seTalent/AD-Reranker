"""MMLongBench-Doc-V2 semantic judging with a local judge model (official prompt/schema/metrics)."""
import importlib.util
import json
import os

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor_v2")
PROMPT = open(os.path.join(HERE, "judge_prompt.txt")).read()
SCHEMA = json.load(open(os.path.join(HERE, "judge_schema.json")))
MAX_RESPONSE_CHARS = 12000

_spec = importlib.util.spec_from_file_location("v2_metrics", os.path.join(HERE, "metrics.py"))
metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(metrics)


def judge_messages(sample, response):
    prompt = PROMPT.format(question=" ".join(sample["question"].split()), answer=sample["answer"],
                           answer_format=sample["answer_format"], response=str(response)[:MAX_RESPONSE_CHARS])
    return [{"role": "user", "content": prompt}]


def verdict(parsed, prompt="v2"):
    """Normalise a judge output to {"equivalent": bool, "abstained": bool|None}; None if invalid."""
    if not isinstance(parsed, dict):
        return None
    if prompt == "v2":
        if isinstance(parsed.get("equivalent"), bool) and isinstance(parsed.get("abstained"), bool):
            return dict(equivalent=parsed["equivalent"], abstained=parsed["abstained"], reason=parsed.get("reason", ""))
        return None
    bc = parsed.get("binary_correctness")
    if bc in (0, 1, "0", "1"):
        return dict(equivalent=int(bc) == 1, abstained=None)
    return None
