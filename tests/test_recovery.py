"""Synthetic tests for evidence recovery and refiner output parsing (no data, no models).
Run from the repository root: python -m pytest -q tests"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from recovery import greedy, insert
from recovery.graph import anchor, build_graph, locate
from recovery.pagerep import build, render, split_markdown
from recovery.verifier import check_provenance, gap_support, rel_ranking


def page(i, md):
    return build(i, split_markdown(md), "mineru")


def df_of(reps):
    df = Counter()
    for r in reps:
        df.update(set(r.terms))
    return df


def test_refiner_selection_parser():
    from models.qwen import parse_selection
    assert parse_selection("[3, 1, 3, 12]", 10) == ([3, 1], False)
    assert parse_selection("[10, 7]", 10) == ([10, 7], False)      # positions > 4 are kept for top-10
    assert parse_selection("[6]", 4) == ([1, 2, 3, 4], True)        # out of range -> fallback to all
    assert parse_selection("", 3) == ([1, 2, 3], True)


def test_openai_messages_use_absolute_file_urls():
    from models.local_llm import to_openai_messages
    m = to_openai_messages([{"role": "system", "content": "s"},
                            {"role": "user", "content": [{"type": "text", "text": "[1] :"},
                                                         {"type": "image", "image": "tmp/X/doc_0.png"}]}])
    assert m[0] == {"role": "system", "content": "s"}
    url = m[1]["content"][1]["image_url"]["url"]
    assert url == "file://" + os.path.abspath("tmp/X/doc_0.png")


def test_vllm_images_match_transformers_preprocessing(tmp_path):
    import numpy as np
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from models.local_llm import prepare_images
    src = tmp_path / "pages" / "doc_0.png"
    src.parent.mkdir()
    Image.fromarray((np.random.RandomState(0).rand(1583, 1225, 3) * 255).astype("uint8")).save(src)
    expected = process_vision_info([{"role": "user", "content": [{"type": "image", "image": str(src)}]}])[0][0]
    msgs = [[{"role": "user", "content": [{"type": "image", "image": str(src)}]}]]
    cached = Image.open(prepare_images(msgs)[str(src)])
    assert cached.size == expected.size and cached.size[0] % 28 == 0 and cached.size[1] % 28 == 0
    assert np.array_equal(np.asarray(cached.convert("RGB")), np.asarray(expected.convert("RGB")))


def test_markdown_blocks_and_labels():
    r = page(0, "# 3 Results\n\nAs shown in Table 2, revenue grew.\n\n"
                "<table><tr><td>Year</td><td>Revenue</td></tr><tr><td>2019</td><td>5.1</td></tr></table>\n\n"
                "Figure 3: Revenue by year\n\n![](images/a.jpg)")
    assert [b.kind for b in r.blocks] == ["heading", "text", "table", "caption"]
    assert "Year | Revenue" in r.blocks[2].text
    assert r.defs == {"figure 3": "b4"}
    assert r.mentions["table 2"] == ["b2"]


def test_graph_relations_and_location():
    reps = [
        page(0, "# Intro\n\nUnrelated opening text about the company history."),
        page(1, "# Results\n\nAs shown in Table 2, the Zorblax metric improved in the"),
        page(2, "following quarters, as the Zorblax team expanded.\n\nTable 2: Zorblax metric by quarter\n\n"
                "<table><tr><td>Q1</td><td>7</td></tr></table>"),
        page(3, "# Appendix\n\nOther material."),
    ]
    edges = build_graph([0, 1, 2, 3], reps, "What was the Zorblax metric in Q1?", df_of(reps), 4)
    assert {"ref", "cont", "qent"} <= {r.kind for r in edges[(1, 2)]}
    assert locate(edges, [1], [0, 2, 3], {0: 0, 1: 1, 2: 2, 3: 3})[0] == 2
    assert anchor(edges, 2, [1]) == 1


def test_provenance_and_gap_support():
    reps = {7: page(7, "Table 4: Net sales\n\nNet sales in 2020 were $12.5 billion, up 3%.")}
    assert check_provenance(reps[7], "b2", "net sales in 2020 were $12.5 billion")[0]
    assert check_provenance(reps[7], "b1", "Net sales in 2020 were $12.5 billion")[1] == "exact-other-block"
    assert not check_provenance(reps[7], "b2", "net income was $4 billion")[0]
    out = json.dumps({
        "requirements": [{"id": "h1", "need": "2020 net sales", "conditions": "", "status": "missing", "kept_evidence": ""},
                         {"id": "h2", "need": "2019 net sales", "conditions": "", "status": "covered", "kept_evidence": "S1.b2"}],
        "support": [{"requirement": "h1", "candidate": "C1", "block": "C1.b2", "quote": "Net sales in 2020 were $12.5 billion"},
                    {"requirement": "h2", "candidate": "C1", "block": "C1.b2", "quote": "Net sales"},
                    {"requirement": "h1", "candidate": "C1", "block": "C1.b2", "quote": "made-up text that is absent"}]})
    res = gap_support(out, {"C1": 7}, reps)
    assert res["F"] == {7: {"h1"}}
    assert [a["how"] for a in res["audit"]] == ["exact", "not-missing", "not-found"]
    assert gap_support("not json", {"C1": 7}, reps) is None
    assert rel_ranking('{"useful": [{"candidate": "C2", "reason": ""}, {"candidate": "C9", "reason": ""}]}', {"C2": 5}) == [5]


def test_multimodal_messages_and_visual_support():
    from recovery.verifier import build_input, messages_mm
    reps = {0: page(0, "Kept text about revenue."), 3: page(3, "Figure 2: Sales by region\n\n![](images/x.jpg)"),
            5: page(5, "Plain text without figures here.")}
    reps[3].visuals = 1
    vin = build_input("q?", [0], [3, 5], reps, {})
    msgs = messages_mm("q?", vin, lambda p: f"file:///img/{p}.png")
    urls = [c["image_url"]["url"] for c in msgs[1]["content"] if c["type"] == "image_url"]
    assert urls == ["file:///img/0.png", "file:///img/3.png", "file:///img/5.png"]
    out = json.dumps({"requirements": [{"id": "h1", "need": "sales", "status": "missing"}],
                      "support": [{"requirement": "h1", "candidate": "C1", "source": "visual", "block": "", "quote": "bar chart"},
                                  {"requirement": "h1", "candidate": "C2", "source": "visual", "block": "", "quote": "chart"}]})
    res = gap_support(out, vin["cand_ids"], reps)
    assert res["F"] == {3: {"h1"}} and [a["how"] for a in res["audit"]] == ["visual", "visual-no-figure"]


def test_yes_probability_and_score_selection():
    import math
    from recovery import EvidenceRecovery, mode_of
    from recovery.verifier import yes_probability
    assert abs(yes_probability({"Yes": math.log(0.6), "No": math.log(0.2), "The": math.log(0.1)}) - 0.75) < 1e-9
    assert yes_probability({"The": -0.1}) is None and yes_probability(None) is None
    assert mode_of("score40") == "score_mm" and mode_of("ours") == "gap_graph" and mode_of("ret") is None
    assert mode_of("scorenk30") == "score_nokept" and mode_of("scoreni5") == "score_noimg"
    ctx = dict(kept=[1], dropped=[4, 2, 7], rank={1: 0, 4: 1, 2: 2, 7: 3}, edges={}, located=[])
    ver = {"score_mm": {"skipped": False, "scores": {"4": 0.3, "2": 0.9, "7": 0.55}}}
    er = EvidenceRecovery.__new__(EvidenceRecovery)
    assert er.select("score50", 2, ctx, ver)[1] == [2, 7]
    assert er.select("score80", 2, ctx, ver)[1] == [2]
    assert er.select("score20", 1, ctx, ver) == ([1, 2], [2])


def test_ranking_metrics_and_tail_suffix():
    from recovery import EvidenceRecovery, mode_of
    from recovery.breakdown import _ranking
    mrr, ndcg = _ranking([5, 2, 9], {2, 9})
    assert mrr == 0.5 and 0 < ndcg < 1
    assert _ranking([2, 9], {2, 9}) == (1.0, 1.0)
    assert mode_of("score20_tail") == "score_mm"
    ctx = dict(kept=[4, 1], dropped=[2, 5], rank={4: 0, 1: 1, 2: 2, 5: 3}, edges={}, located=[])
    ver = {"score_mm": {"skipped": False, "scores": {"2": 0.9, "5": 0.8}}}
    er = EvidenceRecovery.__new__(EvidenceRecovery)
    assert er.select("score50_tail", 2, ctx, ver) == ([4, 1, 2, 5], [2, 5])


def test_greedy_and_insert():
    F = {5: {"h1"}, 6: {"h1", "h2"}, 7: {"h2"}, 8: {"h3"}}
    assert greedy(F, [5, 6, 7, 8], 2) == [6, 8]
    assert greedy(F, [5, 7], 3) == [5, 7]
    assert greedy({}, [1, 2], 2) == []
    reps = [page(i, f"Text block number {i} without terminal punctuation and long enough to continue") for i in range(6)]
    edges = build_graph([1, 2, 4, 5], reps, "q", df_of(reps), 6)
    assert insert([4, 1], [2, 5], edges) == [4, 5, 1, 2]
    assert insert([4, 1], [2, 5], edges, tail=True) == [4, 1, 2, 5]


def test_render_budget_keeps_priority_blocks():
    r = page(0, "# Head\n\n" + "\n\n".join(f"filler paragraph {i} " + "x" * 300 for i in range(20))
             + "\n\nThe key fact is 42.")
    txt, cut = render(r, "C1", 1200, keep={r.blocks[-1].bid})
    assert cut and "The key fact is 42." in txt and "[C1.b1] (heading) Head" in txt
