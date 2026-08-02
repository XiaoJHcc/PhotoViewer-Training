"""
golden_pair_gen.py — 金标准批2 train 团 → 干净团内训练对（确信度加权）

源：D:/PhotoDB/dataset/golden_star2/（key 中 split=train 的 75 团 + 用户标星读回）。
规则（2026-07-28 用户裁定）：
  - 组内两两配对、用户星高者胜（y=+1 胜者在前）；同星对剔除（0:0 不可排 / N:N 同好，不强迫）；
  - **确信度加权**：星差 1 → 0.5（低确信，如路人位置/姿态类细节）；星差 2 → 1.0；
    星差 ≥3 → 1.5（高确信，多为清晰度/水平度等技术判别）；
  - 只取 split=train 的团（test 25 团永久入考试，不入训）。

输出 audit/out/golden_pairs/pairs_train.csv（schema 与 m3 对齐，ptype=golden）+ 报告。

用法（仓根 D:/Git/PhotoViewer 下）：
    PYTHONUTF8=1 Tools/.venv/Scripts/python.exe Training/audit/golden_pair_gen.py
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GS2 = Path("D:/PhotoDB/dataset/golden_star2")
OUT = ROOT / "audit" / "out" / "golden_pairs"

W_OF_GAP = {1: 0.5, 2: 1.0}          # ≥3 → 1.5


def main() -> int:
    key = list(csv.DictReader(open(GS2 / "golden_batch2_key.csv", encoding="utf-8-sig")))
    train_gids = {r["gid"] for r in key if r["split"] == "train"}
    stars = {}
    for ln in open(GS2 / "golden_star_readback.tsv", encoding="utf-8"):
        ln = ln.rstrip("\n")
        if not ln or ln.startswith("gid"):
            continue
        g, a, fp, us, *_ = ln.split("\t")
        if g in train_gids:
            stars[(g, a)] = (fp, int(us))
    by_gid = defaultdict(list)
    for (g, a), (fp, us) in stars.items():
        by_gid[g].append((fp, us))

    OUT.mkdir(parents=True, exist_ok=True)
    n_pairs = 0
    stat = Counter()
    with open(OUT / "pairs_train.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fp_i", "fp_j", "ptype", "weight", "cos", "cv_dist", "dstar", "s_gap"])
        for g, members in sorted(by_gid.items()):
            for (fi, si), (fj, sj) in combinations(members, 2):
                if si == sj:
                    stat["tie_drop"] += 1
                    continue
                gap = abs(si - sj)
                wt = W_OF_GAP.get(gap, 1.5)
                (wi, wj) = (fi, fj) if si > sj else (fj, fi)
                w.writerow([wi, wj, "golden", f"{wt:.4f}", "nan", "nan", gap, "nan"])
                stat[f"gap{min(gap, 3)}"] += 1
                n_pairs += 1

    L = ["# 金标准批2 干净训练对报告\n",
         f"- train 团 {len(by_gid)} → 对 {n_pairs}（同星剔除 {stat['tie_drop']}）",
         f"- 星差分布: gap1 {stat['gap1']}（×0.5）· gap2 {stat['gap2']}（×1.0）· gap≥3 {stat['gap3']}（×1.5）",
         f"- 权重质量: {sum(stat[f'gap{k}'] * W_OF_GAP.get(k, 1.5) for k in (1, 2, 3)):.0f}"]
    (OUT / "golden_pairs_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[OK] {OUT}/pairs_train.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
