"""Batched requests to a locally deployed OpenAI-compatible server (vLLM on H200).

Used by the evidence-recovery verifier, the local judge and the optional `backend: vllm`
of RefinerModel / AnswerModel. No external API is involved: base_url points to a server
started with scripts/serve_vllm.sh.
"""
import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor

from openai import AsyncOpenAI


def hf_preprocessed(path):
    """The image the transformers path feeds to the processor, cached as a lossless PNG.

    RefinerModel/AnswerModel first call qwen_vl_utils.process_vision_info, whose smart_resize
    rounds the page to multiples of 28, and only then resize to all_pixels. Sending the raw page
    lets the server resize once, which changes pixels and sometimes the visual-token grid.
    """
    cached = os.path.join(os.path.dirname(os.path.abspath(path)) + "_qvl", os.path.basename(path))
    if not cached.endswith(".png"):
        cached += ".png"
    if not os.path.exists(cached):
        from qwen_vl_utils import process_vision_info
        os.makedirs(os.path.dirname(cached), exist_ok=True)
        image = process_vision_info([{"role": "user", "content": [{"type": "image", "image": path}]}])[0][0]
        tmp = cached + f".{os.getpid()}.tmp.png"
        image.save(tmp)
        os.replace(tmp, cached)
    return cached


def prepare_images(messages_list, workers=16):
    """Preprocess every distinct image of a batch in parallel; returns {path: cached path}."""
    paths = sorted({c["image"] for ms in messages_list for m in ms if isinstance(m["content"], list)
                    for c in m["content"] if c["type"] == "image"})
    with ThreadPoolExecutor(workers) as ex:
        return dict(zip(paths, ex.map(hf_preprocessed, paths)))


def to_openai_messages(messages, image_map=None):
    """qwen_vl_utils messages ({"type": "image", "image": path}) -> OpenAI chat format.
    image_map replaces image paths (e.g. with prepare_images output)."""
    out = []
    for m in messages:
        content = m["content"]
        if isinstance(content, list):
            parts = []
            for c in content:
                if c["type"] == "image":
                    path = (image_map or {}).get(c["image"], c["image"])
                    url = path if path.startswith(("file://", "http://", "https://", "data:")) \
                        else f"file://{os.path.abspath(path)}"   # dataset paths are relative to the repo root
                    parts.append({"type": "image_url", "image_url": {"url": url}})
                else:
                    parts.append({"type": "text", "text": c["text"]})
            content = parts
        out.append({"role": m["role"], "content": content})
    return out


class LocalLLM:
    def __init__(self, base_url, model, concurrency=32, timeout=1800):
        self.base_url, self.model, self.concurrency = base_url, model, concurrency
        self.timeout = timeout

    async def _one(self, client, sem, messages, max_tokens, schema, thinking, extra, logprobs=0, attempts=3):
        body = dict(extra or {})
        if thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": thinking}
        kwargs = {}
        if schema is not None:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {"name": "output", "schema": schema}}
        if logprobs:
            kwargs.update(logprobs=True, top_logprobs=logprobs)
        last = None
        async with sem:
            for attempt in range(attempts):
                t0 = time.perf_counter()
                try:
                    r = await client.chat.completions.create(
                        model=self.model, messages=messages, max_tokens=max_tokens, temperature=0.0, seed=0,
                        extra_body=body or None, **kwargs)
                    msg = r.choices[0].message
                    first = None
                    if logprobs and r.choices[0].logprobs and r.choices[0].logprobs.content:
                        first = {t.token: t.logprob for t in r.choices[0].logprobs.content[0].top_logprobs}
                    return dict(text=msg.content or "",
                                reasoning=getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None),
                                prompt_tokens=r.usage.prompt_tokens, completion_tokens=r.usage.completion_tokens,
                                finish_reason=r.choices[0].finish_reason, latency=time.perf_counter() - t0, error=None,
                                first_token_logprobs=first)
                except Exception as e:  # transport / server errors are retried, then recorded
                    last = f"{type(e).__name__}: {e}"
                    await asyncio.sleep(5 * (attempt + 1))
        return dict(text="", reasoning=None, prompt_tokens=0, completion_tokens=0, finish_reason=None,
                    latency=0.0, error=last)

    async def achat_many(self, batch, max_tokens, schema=None, thinking=None, extra=None, logprobs=0):
        client = AsyncOpenAI(base_url=self.base_url, api_key="EMPTY", timeout=self.timeout, max_retries=0)
        sem = asyncio.Semaphore(self.concurrency)
        try:
            return await asyncio.gather(*(self._one(client, sem, m, max_tokens, schema, thinking, extra, logprobs)
                                          for m in batch))
        finally:
            await client.close()

    def chat_many(self, batch, max_tokens, schema=None, thinking=None, extra=None, logprobs=0):
        """logprobs=n also returns the top-n log-probabilities of the first generated token."""
        return asyncio.run(self.achat_many(batch, max_tokens, schema, thinking, extra, logprobs))
