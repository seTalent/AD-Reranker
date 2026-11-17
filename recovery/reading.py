"""Evidence-aware reading: how the reader is given the recovered pages.

The plain reader (models.qwen.AnswerModel) sees the final page list without knowing which pages
the reranker kept and which ones recovery added back. Three switches, set with `+reading=...`
("tag+text" style subset; the answers go to a separate file, see pipeline.answers_file):

  tag    provenance: recovered pages are labelled "(recovered page)" and the prompt's existing
         `reason` slot says why they were added. Costs a few prompt tokens.
  text   page text (MinerU markdown or PyMuPDF text, recovery.pagerep) is given after the image
         of every recovered page; `text_all` gives it for every page.
  conf   always on: the reader's token log-probabilities are kept, so analysis can compare the
         reader's confidence in two answers (answer-level fallback, scripts/rp2_reading_analysis.py).

Only the prompt changes; pages, their order and resolution are those of recover.json.
"""
import json
import re

REASON = ("Pages {pages} were not selected at first, but were added back because a verifier judged that they "
          "may contain information missing from the other pages. Use them if they are relevant; ignore them if not.")


def parse(spec):
    opts = set(x.strip() for x in re.split(r"[,+]", str(spec or "")) if x.strip())   # "tag+text" (hydra-safe)
    unknown = opts - {"tag", "text", "text_all", "plain"}
    if unknown:
        raise ValueError(f"unknown reading options {sorted(unknown)}")
    return opts


def file_tag(spec):
    opts = parse(spec)
    return "ear-" + "-".join(sorted(opts)) if opts else ""


def messages(model, question, images, ids, added, page_text, opts, text_chars=1500):
    """qwen_vl_utils-style messages (same layout as AnswerModel.create_image_message plus the switches).
    images: candidate image paths; ids: 1-based candidate ids to show; added: ids added by recovery;
    page_text(id) -> extracted text of that candidate page."""
    added = [i for i in ids if i in set(added)]
    reason = REASON.format(pages=", ".join(f"[{i}]" for i in added)) if "tag" in opts and added else ""
    content = [{"type": "text", "text": model.user_message.format(question=question, reason=reason)}]
    for i in dict.fromkeys(ids):
        if i - 1 >= len(images):
            continue
        label = f"[{i}] (recovered page) :" if "tag" in opts and i in added else f"[{i}] :"
        content.append({"type": "text", "text": label})
        content.append({"type": "image", "image": images[i - 1]})
        if "text_all" in opts or ("text" in opts and i in added):
            t = page_text(i).strip()
            if t:
                content.append({"type": "text", "text": f"Text extracted from page [{i}]:\n{t[:text_chars]}"})
    return [{"role": "system", "content": model.system_prompt}, {"role": "user", "content": content}]


def confidence(tokens, text):
    """Mean log-probability of the generated tokens, overall and over the value of the "Answer" field.
    tokens: [(token, logprob)] of the greedy completion; text: the completion."""
    if not tokens:
        return dict(conf_all=None, conf_ans=None, conf_min=None, n_ans=0)
    lps = [lp for _, lp in tokens]
    m = re.search(r'"Answer"\s*:\s*', text)
    start = m.end() if m else 0
    r = re.search(r',\s*"reason"\s*:', text[start:])
    end = start + r.start() if r else len(text)
    pos, ans = 0, []
    for tok, lp in tokens:
        if pos < end and pos + len(tok) > start and tok.strip():    # tokens overlapping the answer value
            ans.append(lp)
        pos += len(tok)
    return dict(conf_all=sum(lps) / len(lps), conf_ans=(sum(ans) / len(ans)) if ans else None,
                conf_min=min(ans) if ans else None, n_ans=len(ans))


def normalise(answer):
    a = str(answer).strip().lower()
    a = re.sub(r"[\s,]+", " ", a)
    return a.strip(" .'\"")


def answer_text(response):
    try:
        return json.loads(response)["Answer"]
    except Exception:
        m = re.search(r'"Answer"\s*:\s*(.*?)(?:,\s*"reason"|}\s*$)', response, re.S)
        return m.group(1) if m else response
