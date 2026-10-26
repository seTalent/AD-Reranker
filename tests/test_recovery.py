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
