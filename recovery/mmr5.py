"""MM-R5 (i2vec/MM-R5, a Qwen2.5-VL-7B reasoning reranker) as the reranker in front of recovery (results doc
section 27). Its own prompt and output format (models/MMR5/examples/reranker.py): a full ranking of the
Top-10 pages; as in research point 1 the reader keeps the first k (default 2). Served by vLLM (scripts/serve_vllm.sh
mmr5); greedy decoding instead of the original T = 0.3 sampling, so runs are repeatable; pages at the same
resolution as the other rerankers (all_pixels)."""
import re

from models.MMR5.examples.reranker import SYSTEM_PROMPT, question_template


def messages(question, images):
    content = [{"type": "text", "text": question_template.format(Question=question, image_num=len(images),
                                                                 image_num_end=len(images))}]
    for i, p in enumerate(images, 1):
        content += [{"type": "text", "text": f"\nImage {i}: "}, {"type": "image", "image": p}]
    return [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content}]


def parse(text, n):
    """1-based ranking; ids missing from the answer follow in retrieval order. (order, parsed?)"""
    m = re.search(r"<answer>\s*\[(.*?)\]\s*</answer>", text or "", re.S)
    order = []
    if m:
        for x in re.findall(r"\d+", m.group(1)):
            i = int(x)
            if 1 <= i <= n and i not in order:
                order.append(i)
    return order + [i for i in range(1, n + 1) if i not in order], bool(order)
