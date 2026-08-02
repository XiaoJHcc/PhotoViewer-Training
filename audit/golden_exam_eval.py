"""
golden_exam_eval.py — 金标准合并考卷评估（台阶② 标准考卷：四层容差剖面 + 分层）

考卷 = D:/PhotoDB/dataset/golden_exam/（批1 48 团 + 批2 test 25 团，见 golden_exam_merge.py）。
口径（2026-07-28 用户裁定）：
  - 偏差 = 用户首选星 − 预测所选星（用户组内排序尺，只作组内序差、不作跨组数值）；
  - 四层剖面 = exact / 偏差1 / 偏差2 / 偏差≥3（对照草案目标 50/45/5/0）；
  - 前二 = 预测所选 ≥ 组内次优星（注意：星差>1 的组里"前二"比"偏差≤1"宽松，两列都报）；
  - tie/tie0 团不进判定，单列计数；
  - 分层：事件 / 批次 / 极相似带(cos≥0.96) / 潜分带 / 用户确信度(首选-次选差 1 vs ≥2)。

用法（仓根 D:/Git/PhotoViewer 下）：
    PYTHONUTF8=1 Tools/.venv/Scripts/python.exe Training/audit/golden_exam_eval.py \
        --scores ens=Training/train/out/m8_best/scores_ens.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

EXAM_DIR = Path("D:/PhotoDB/dataset/golden_exam")
COS_HI = 0.96


def main() -> int:
    ap = argparse.ArgumentParser(description="金标准合并考卷评估（台阶② 四层容差剖面）")
    ap.add_argument("--exam", default=str(EXAM_DIR))
    ap.add_argument("--scores", nargs="+", required=True, help="name=path 列表")
    args = ap.parse_args()

    exam = Path(args.exam)
    key = list(csv.DictReader(open(exam / "golden_exam_key.csv", encoding="utf-8-sig")))
    kmeta = {(r["gid"], r["anon"]): r for r in key}
    rb = {}
    for ln in open(exam / "golden_exam_readback.tsv", encoding="utf-8"):
        ln = ln.rstrip("\n")
        if not ln or ln.startswith("gid"):
            continue
        g, a, fp, us, orating, iot, iup, st = ln.split("\t")
        rb[(g, a)] = dict(gid=g, anon=a, fingerprint=fp, user_star=int(us),
                          orig_rating=int(orating), status=st)
    by_gid = defaultdict(list)
    for (g, a), r in rb.items():
        by_gid[g].append(r)

    decided, ties = [], 0
    for g, ms in sorted(by_gid.items()):
        if ms[0]["status"] != "win":
            ties += 1
            continue
        us = sorted([m["user_star"] for m in ms], reverse=True)
        winner_star = us[0]
        second_star = us[1] if len(us) > 1 else us[0]
        meta0 = kmeta[(g, ms[0]["anon"])]
        decided.append(dict(gid=g, members=ms, winner_star=winner_star,
                            second_star=second_star, gap=winner_star - second_star,
                            event=meta0["event"], batch=meta0["batch"],
                            cos=float(meta0.get("cos") or 0.9),
                            lat_q=float(meta0.get("lat_q") or 0.5)))
    print(f"考卷: {len(decided)} 判定组（tie/tie0 {ties} 不计）")

    sources = {}
    for spec in args.scores:
        name, path = spec.split("=", 1)
        sources[name] = {r["fingerprint"]: float(r["score"])
                         for r in csv.DictReader(open(path, encoding="utf-8-sig"))}

    def eval_source(s_of, groups):
        rows = []
        for d in groups:
            ms = [m for m in d["members"] if m["fingerprint"] in s_of]
            if len(ms) < 2:
                continue
            pred = max(ms, key=lambda m: s_of[m["fingerprint"]])
            dev = d["winner_star"] - pred["user_star"]
            rows.append((dev, int(pred["user_star"] >= d["second_star"]), d))
        if not rows:
            return None
        n = len(rows)
        devs = Counter(r[0] for r in rows)
        return dict(n=n, exact=devs[0] / n, dev1=devs[1] / n,
                    dev2=devs[2] / n, dev3p=sum(v for k, v in devs.items() if k >= 3) / n,
                    top2=float(np.mean([r[1] for r in rows])), rows=rows)

    def line(tag, groups):
        cells = [f"{tag:<24}"]
        for name in sources:
            r = eval_source(sources[name], groups)
            if r is None:
                cells.append(f"{'-':>34}")
                continue
            cells.append(f"{r['exact']:.2f}/{r['dev1']:.2f}/{r['dev2']:.2f}/{r['dev3p']:.2f}"
                         f" ≤1={r['exact'] + r['dev1']:.2f} 前二={r['top2']:.2f}(n={r['n']})")
        print("  ".join(cells))

    hdr = f"{'组层（偏差 exact/1/2/≥3）':<24}"
    print(hdr + "  ".join(f"{n:>34}" for n in sources))
    line("全体", decided)
    for ev in sorted({d["event"] for d in decided}):
        line(f"事件 {ev}", [d for d in decided if d["event"] == ev])
    for tag, f in (("极相似带 cos≥0.96", lambda d: d["cos"] >= COS_HI),
                   ("相似带 <0.96", lambda d: d["cos"] < COS_HI),
                   ("高确信(星差≥2)", lambda d: d["gap"] >= 2),
                   ("低确信(星差=1)", lambda d: d["gap"] == 1)):
        line(tag, [d for d in decided if f(d)])
    for name in sources:
        r = eval_source(sources[name], decided)
        hard = [row[2]["gid"] for row in r["rows"] if row[0] >= 2]
        print(f"  [{name}] 偏差≥2 硬错误 {len(hard)} 组: {hard}")
    print("\n草案目标: exact≥0.50 / ≤1累计≥0.95 / 偏差2≤0.05 / ≥3 0（n 小，±1 组≈1.7-2.4pt）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
