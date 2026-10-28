"""Markdown tables from eval.json."""
import json

from recovery.analysis import MAIN

NAME = {"top4": "无重排 Top-4", "top10": "无重排 Top-10", "rerank": "Zero-Shot 重排（基线）",
        "ret@2": "+检索分数回填", "adj@2": "+相邻页扩展", "graph@2": "+关联图直接补回（无缺失检测）",
        "rel@2": "+关联图+相关性校验（无需求建模）", "ours_ret@2": "+缺失校验，无关联图",
        "ours_all@2": "+缺失校验，全部被筛页", "ours_noprov@2": "+本方法，无来源检查",
        "ours_tail@2": "+本方法，末尾插入", "ours@2": "**+本方法**", "oracle@2": "+金标补回（上界）"}


def pct(x):
    return "–" if x is None or x != x else f"{100 * x:.1f}"


def report(path, cfg=None):
    r = json.load(open(path))
    V, P, M, G = r["variants"], r["paired"], r["motivation"], r["graph"]
    out = ["## 动机：重排误过滤", "", "| 指标 | 数值 |", "|---|---|"]
    for k, label in [("retrieval_recall", "ColPali 检索证据页召回"), ("all_gold_retrieved", "全部证据页都被检索到的比例"),
                     ("kept_recall_in_pool", "候选池内证据页被重排保留的比例"),
                     ("wrongful_filter_rate", "误过滤率（候选池含证据的题中，≥1 个证据页被筛掉）"),
                     ("wrongful_filter_single", "　单页证据题"), ("wrongful_filter_multi", "　多页证据题"),
                     ("refiner_fallback", "重排空选择→回退全部候选")]:
        out.append(f"| {label} | {pct(M[k])}% |")
    out += [f"| 平均保留页数 | {M['kept_pages']:.2f} |",
            f"| 误过滤题数 / 被筛证据页数 | {M['wf_questions']} / {M['dropped_gold_pages']} |", "",
            "| 被筛页 | 页数 | 与保留页有关联 | 进入校验候选 |", "|---|---|---|---|",
            f"| 证据页 | {G['dropped_gold']} | {pct(G['gold_linked'])}% | {pct(G['gold_in_U'])}% |",
            f"| 非证据页 | {G['dropped_other']} | {pct(G['other_linked'])}% | {pct(G['other_in_U'])}% |", "",
            "## 主结果（b = 2）", "",
            "| 方法 | Acc | F1 | 误过滤子集 Acc | 跨页 Acc | 证据召回 | 完整覆盖 | 页数 | 补回页 | 补回精度 | 误过滤恢复率 | ΔAcc [95% CI] | 胜/负 | McNemar p |",
            "|---|" + "---|" * 13]
    for v in MAIN:
        if v not in V:
            continue
        x, p = V[v], P.get(v)
        d = "–" if not p else f"{100 * p['delta']:+.1f} [{100 * p['ci95'][0]:+.1f}, {100 * p['ci95'][1]:+.1f}]"
        wl = "–" if not p else f"{p['win']}/{p['loss']}"
        pv = "–" if not p else f"{p['mcnemar_p']:.3f}"
        out.append(f"| {NAME[v]} | {pct(x['acc'])} | {pct(x['f1'])} | {pct(x['wf_acc'])} | {pct(x['multi_acc'])} | "
                   f"{pct(x['page_recall'])} | {pct(x['full_coverage'])} | {x['pages']:.2f} | {x['added']:.2f} | "
                   f"{pct(x['added_precision'])} | {pct(x['wf_recovery'])} | {d} | {wl} | {pv} |")
    out += ["", "## 补回预算", "", "| 方法 | b | Acc | 证据召回 | 补回页 | 补回精度 | 误过滤恢复率 |", "|---|---|---|---|---|---|---|"]
    for base in ("ret", "graph", "ours", "oracle"):
        for b in (1, 2, 3):
            x = V.get(f"{base}@{b}")
            if x:
                out.append(f"| {base} | {b} | {pct(x['acc'])} | {pct(x['page_recall'])} | {x['added']:.2f} | "
                           f"{pct(x['added_precision'])} | {pct(x['wf_recovery'])} |")
    out += ["", "## 校验器与开销", "",
            "| 模式 | 调用/题数 | 需求数 | 缺失数 | 来源检查通过 | 输入 token | 输出 token | 建图 ms |", "|---|---|---|---|---|---|---|---|"]
    for m, s in r["verifier"].items():
        out.append(f"| {m} | {s['called']}/{s['questions']} | {s['requirements']:.2f} | {s['missing']:.2f} | "
                   f"{pct(s['provenance_pass'])}% | {s['prompt_tokens']:.0f} | {s['completion_tokens']:.0f} | {s['graph_ms']:.1f} |")
    if r.get("judge_agreement"):
        a = r["judge_agreement"]
        out += ["", f"裁判一致性：与 {a['reference']} 在 {a['n']} 条共同回答上，equivalent 一致 {pct(a['equivalent'])}%，"
                    f"abstained 一致 {pct(a['abstained'])}%，判对率 {pct(a['positive_rate'][0])}% / {pct(a['positive_rate'][1])}%。"]
    if r.get("latency"):
        out += ["", f"顺序时延（{r['n_probe']} 题，单请求，秒）", "", "| 环节 | 均值 | 中位数 |", "|---|---|---|"]
        out += [f"| {k} | {v['mean']:.3f} | {v['median']:.3f} |" for k, v in r["latency"].items()]
    return "\n".join(out)
