#!/usr/bin/env python3
"""Render a comparison JSON into a readable Markdown report + CSV (same directory).

  python scripts/report_benchmarks.py results/renderer/E-R0_cpu.comparison.json
"""
import csv
import json
import sys
from pathlib import Path


def fmt(v):
    if v is None:
        return "null"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def main(argv=None):
    argv = argv or sys.argv[1:]
    p = Path(argv[0])
    cmp = json.loads(p.read_text())
    src = json.loads(Path(cmp["source"]).read_text())
    env = cmp["environment"]
    lines = [f"# Benchmark report: {src['experiment']['id']} — {src['experiment']['title']}", "",
             f"- result: `{cmp['source']}`  config hash `{cmp['config_hash'][:16]}`",
             (f"- commit `{env.get('git_commit')}` · torch {env.get('torch')} (hip={env.get('torch_hip')}, "
              f"cuda={env.get('torch_cuda')}) · GPU visible: {env.get('gpu_visible')} · {env.get('platform')}"),
             f"- method: `{src['method']}` · started {src['started_at']}", ""]
    rows_csv = []
    if src["benchmark"] == "renderer":
        lines += [f"Semantics: {src['semantics']}", "",
                  f"Baseline backend **{cmp['baseline']}** vs candidate **{cmp['candidate']}**.", ""]
        for g in cmp["global_correctness"]:
            lines.append(f"- global gate `{g['name']}`: {'pass' if g.get('passed') else 'fail'} ({g.get('method', '')})")
        for grp in ("correctness", "workload", "performance", "quality"):
            rs = [r for r in cmp["rows"] if r["group"] == grp]
            if not rs:
                continue
            lines += ["", f"## {grp}", "", "| case | metric | baseline | candidate | abs diff | rel change | status |",
                      "|---|---|---|---|---|---|---|"]
            for r in rs:
                case = r["case"] if isinstance(r["case"], str) else f"N={r['case']['n']} {r['case']['res']}px"
                lines.append(f"| {case} | {r['metric']} | {fmt(r['baseline'])} | {fmt(r['candidate'])} | "
                             f"{fmt(r['abs_diff'])} | {fmt(r['rel_change'])} | {r['status']}"
                             f"{(' (' + r['reason'] + ')') if r['reason'] else ''} |")
                rows_csv.append({"case": case, **{k: r[k] for k in ("metric", "group", "baseline", "candidate",
                                                                  "abs_diff", "rel_change", "status", "reason")}})
    else:
        for c in cmp["comparisons"]:
            lines += ["", f"## {c['id']} (split `{c['split']}`): {c['baseline']} → {c['candidate']}", "",
                      "| metric | baseline | candidate | abs diff | rel change | status |", "|---|---|---|---|---|---|"]
            for r in c["rows"]:
                lines.append(f"| {r['metric']} | {fmt(r['baseline'])} | {fmt(r['candidate'])} | {fmt(r['abs_diff'])} | "
                             f"{fmt(r['rel_change'])} | {r['status']}{(' (' + r['reason'] + ')') if r['reason'] else ''} |")
                rows_csv.append({"comparison": c["id"], **{k: r[k] for k in ("metric", "group", "baseline", "candidate",
                                                                            "abs_diff", "rel_change", "status", "reason")}})
    md = p.with_name(p.name.replace(".comparison.json", ".report.md"))
    md.write_text("\n".join(lines) + "\n")
    with open(p.with_name(p.name.replace(".comparison.json", ".csv")), "w", newline="") as f:
        if rows_csv:
            w = csv.DictWriter(f, fieldnames=list(rows_csv[0]))
            w.writeheader()
            w.writerows(rows_csv)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
