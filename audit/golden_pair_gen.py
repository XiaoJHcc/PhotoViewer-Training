"""
golden_pair_gen.py — 金标准 train 团 → 干净团内训练对（确信度加权）

源：标星读回后的 golden_star2/（批2）或 golden_star3/（批3，--star-dir 切换）。
规则（2026-07-28 用户裁定）：
  - 组内两两配对、用户星高者胜（y=+1 胜者在前）；同星对剔除（0:0 不可排 / N:N 同好，不强迫）；
  - **确信度加权**：星差 1 → 0.5（低确信，如路人位置/姿态类细节）；星差 2 → 1.0；
    星差 ≥3 → 1.5（高确信，多为清晰度/水平度等技术判别）；
  - 只取 split=train 的团（test 团永久入考试，不入训）。

输出 pairs_train.csv（schema 与 m3 对齐，ptype=golden）+ 报告；批3 用 --out 区分，
合并 = 两个 csv 直接 concat（fingerprint 对级，天然不冲突）。

用法（仓根 D:/Git/PhotoViewer 下）：
    PYTHONUTF8=1 Tools/.venv/Scripts/python.exe Training/audit/golden_pair_gen.py
      [--star-dir D:/PhotoDB/dataset/golden_star3 --key .../golden_batch3_key.csv
       --out Training/audit/out/golden_pairs/pairs_train_b3.csv]
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "audit" / "out" / "golden_pairs"

W_OF_GAP = {1: 0.5, 2: 1.0}          # ≥3 → 1.5


def main() -> int:
    ap = argparse.ArgumentParser(description="金标准 train 团 → 干净训练对")
    ap.add_argument("--star-dir", default="D:/PhotoDB/dataset/golden_star2")
    ap.add_argument("--key", default=None, help="默认 <star-dir>/golden_batch2_key.csv")
    ap.add_argument("--out", default=str(OUT / "pairs_train.csv"))
    ap.add_argument("--report", default=None, help="默认 <out 同目录>/golden_pairs_report.md")
    args = ap.parse_args()
    gs = Path(args.star_dir)
    key_path = Path(args.key) if args.key else gs / "golden_batch2_key.csv"
    out_csv = Path(args.out)
    report_md = Path(args.report) if args.report else out_csv.parent / "golden_pairs_report.md"

    key = list(csv.DictReader(open(key_path, encoding="utf-8-sig")))
    train_gids = {r["gid"] for r in key if r["split"] == "train"}
    stars = {}
    for ln in open(gs / "golden_star_readback.tsv", encoding="utf-8"):
        ln = ln.rstrip("\n")
        if not ln or ln.startswith("gid"):
            continue
        g, a, fp, us, *_ = ln.split("\t")
        if g in train_gids:
            stars[(g, a)] = (fp, int(us))
    by_gid = defaultdict(list)
    for (g, a), (fp, us) in stars.items():
        by_gid[g].append((fp, us))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n_pairs = 0
    stat = Counter()
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
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

    L = [f"# 金标准干净训练对报告（{gs.name}）\n",
         f"- train 团 {len(by_gid)} → 对 {n_pairs}（同星剔除 {stat['tie_drop']}）",
         f"- 星差分布: gap1 {stat['gap1']}（×0.5）· gap2 {stat['gap2']}（×1.0）· gap≥3 {stat['gap3']}（×1.5）",
         f"- 权重质量: {sum(stat[f'gap{k}'] * W_OF_GAP.get(k, 1.5) for k in (1, 2, 3)):.0f}"]
    report_md.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[OK] {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
