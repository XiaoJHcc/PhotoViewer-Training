"""
golden_exam_eval.py — 金标准合并考卷评估（台阶② 标准考卷：四层容差剖面 + 分层 + 顺序颠倒）

考卷 = D:/PhotoDB/dataset/golden_exam/（批1 48 团 + 批2 test 25 团 + 批3 test 40 团 = 113 团，
见 golden_exam_merge.py）。
口径（2026-07-28 用户裁定；2026-08-31 增补质量分层与顺序颠倒）：
  - 偏差 = 用户首选星 − 预测所选星（用户组内排序尺，只作组内序差、不作跨组数值）；
  - 四层剖面 = exact / 偏差1 / 偏差2 / 偏差≥3（对照草案目标 50/45/5/0）；
  - 前二 = 预测所选 ≥ 组内次优星（注意：星差>1 的组里"前二"比"偏差≤1"宽松，两列都报）；
  - tie/tie0 团不进判定，单列计数；
  - 分层：事件 / 极相似带(cos≥0.96) / 质量带（**好片团 = 团顶原星≥4★，2026-08-31 用户裁定唯一口径**；
    中档 3★ / 烂片 ≤2★）/ 用户确信度(首选-次选差 1 vs ≥2)。**质量分层动机（复测分层实测）**：烂片团冠军
    一致率 0.83 但属技术判别域（CV 兜底）、好片团人类冠军线 0.538 而前二 0.923
    ——混算会被烂片团稀释，好片带前二命中率才是"稳定找出好片"的产品指标；
  - 顺序颠倒（组内成对，用户同星对剔除）：硬颠倒 Δ≥2 = 方向用反的定性错误；软颠倒 Δ1 单列。

用法（仓根 D:/Git/PhotoViewer 下）：
    PYTHONUTF8=1 Tools/.venv/Scripts/python.exe Training/audit/golden_exam_eval.py \
        --scores ens=Training/train/out/m8_best/scores_ens.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from itertools import combinations
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
                            orig_cap=max(m["orig_rating"] for m in ms),
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
                   ("低确信(星差=1)", lambda d: d["gap"] == 1),
                   ("好片团(团顶原星≥4★)", lambda d: d["orig_cap"] >= 4),
                   ("中档团(团顶原星3★)", lambda d: d["orig_cap"] == 3),
                   ("烂片团(团顶原星≤2★)", lambda d: d["orig_cap"] <= 2)):
        line(tag, [d for d in decided if f(d)])
    for name in sources:
        r = eval_source(sources[name], decided)
        hard = [row[2]["gid"] for row in r["rows"] if row[0] >= 2]
        print(f"  [{name}] 偏差≥2 硬错误 {len(hard)} 组: {hard}")

    # ── 顺序颠倒剖面（组内成对，用户同星对剔除；硬 Δ≥2 / 软 Δ1 / 冠军涉入）──
    print("\n组内顺序颠倒率（pair 口径，同星=不可判剔除）")
    for name, s_of in sources.items():
        st = Counter()
        for d in decided:
            ms = [m for m in d["members"] if m["fingerprint"] in s_of]
            champ_fp = max(d["members"], key=lambda m: m["user_star"])["fingerprint"]
            for mi, mj in combinations(ms, 2):
                du = mi["user_star"] - mj["user_star"]
                if du == 0:
                    continue
                ds = s_of[mi["fingerprint"]] - s_of[mj["fingerprint"]]
                inv = ((du > 0) != (ds > 0)) and abs(ds) > 1e-9
                cls = "hard" if abs(du) >= 2 else "soft"
                st[(cls, "n")] += 1
                st[(cls, "inv")] += inv
                if champ_fp in (mi["fingerprint"], mj["fingerprint"]):
                    st[("champ", "n")] += 1
                    st[("champ", "inv")] += inv
        f_ = lambda c: f"{st[(c, 'inv')]}/{st[(c, 'n')]} = {st[(c, 'inv')] / max(st[(c, 'n')], 1):.3f}"
        print(f"  [{name}] 硬颠倒(Δ≥2) {f_('hard')} · 软颠倒(Δ1) {f_('soft')} · 冠军涉入 {f_('champ')}")
    print("\n人类线参照（2026-08-31 同卷复测）：全体 冠军 0.704 / 前二 0.926；"
          "好片团(团顶原星≥4★) 冠军 0.538 / 前二 0.923 (n=13)；烂片团(≤2★) 冠军 0.828 / 前二 0.966（技术判别域，CV 兜底）")
    print("草案目标: exact≥0.50 / ≤1累计≥0.95 / 偏差2≤0.05 / ≥3 0（n 小，±1 组≈1.1pt）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
