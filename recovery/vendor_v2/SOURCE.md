Verbatim copies from VectifyAI/MMLongBench-Doc-V2 (Apache-2.0), revision
3aba3c6831a432ee882f763435f9ebe92ab75ed9:

- `metrics.py` = `eval/metrics.py`
- `judge_prompt.txt`, `judge_schema.json` = constants `PROMPT` and `SCHEMA` of `eval/judge.py`

The official judge model is an OpenAI model; here the same prompt and schema are sent to a
locally served judge (see `config/judge/local.yaml`).
