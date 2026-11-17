"""Smoke test of the served models on one real sample (first pages stand in for retrieval)."""
import asyncio
import json
import sys

from erc import config as C
from erc.client import chat
from erc.data import load
from erc.prompts import GAP_SCHEMA, JUDGE_SCHEMA, judge_messages, reader_messages, rerank_messages, verifier_messages
from erc.run import parse_rerank


async def main(roles):
    s = next(x for x in load(require_retrieval=False) if len(x.gold) == 2)
    pages = list(range(min(10, s.n_pages)))
    print("Q:", s.question, "| gold", s.gold, "| answer", s.answer)
    if "vlm" in roles:
        r = await chat("vlm", rerank_messages(s.question, [s.image(p) for p in pages]), C.RERANK_MAX_TOKENS)
        print("RERANK", r["prompt_tokens"], r["completion_tokens"], round(r["latency"], 2), r["error"])
        print(r["text"][-400:])
        pos, fb = parse_rerank(r["text"], len(pages))
        print("positions", pos, "fallback", fb)
        a = await chat("vlm", reader_messages(s.question, [(i, s.image(pages[i - 1])) for i in pos]), C.READER_MAX_TOKENS)
        print("READER", a["prompt_tokens"], a["completion_tokens"], a["error"], a["text"][:300])
    if "verifier" in roles:
        msgs = verifier_messages("gap", s.question, '<page id="S1" physical_page="1">\n[S1.b1] (text) Revenue in 2019 was $5.1 billion.\n</page>',
                                 '<page id="C1" physical_page="2">\n[C1.b1] (text) Revenue in 2020 was $6.3 billion.\n</page>', "")
        msgs[1]["content"] = msgs[1]["content"].replace(s.question, "How much did revenue grow from 2019 to 2020?")
        v = await chat("verifier", msgs, C.VERIFIER_MAX_TOKENS, schema=GAP_SCHEMA, thinking=False)
        print("VERIFIER", v["prompt_tokens"], v["completion_tokens"], round(v["latency"], 2), v["error"])
        print(v["text"])
    if "judge" in roles:
        j = await chat("judge", judge_messages(s.question, s.answer, s.answer_format, f'{{"Answer": "{s.answer}"}}'),
                       C.JUDGE_MAX_TOKENS, schema=JUDGE_SCHEMA, thinking=True)
        print("JUDGE", j["prompt_tokens"], j["completion_tokens"], round(j["latency"], 2), j["error"], j["text"])
        print("reasoning chars", len(j["reasoning"] or ""))


asyncio.run(main(sys.argv[1:]))
