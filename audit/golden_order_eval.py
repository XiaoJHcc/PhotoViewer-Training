"""
golden_order_eval.py — 组内"顺序颠倒"错误率：排序正确性的成对口径（补考卷四层剖面之缺）

背景（2026-08-30 用户提议）：考卷 exact/≤1/前二只考**冠军位**；同星=不可判（差异过低、
烂片选优无意义、或多张好片本该各自为团顶）不进对。用户例：目标 0/1/3，AI 0/2/5 无问题
（保序），0/3/2 = 大问题（冠军丢 + Δ2 硬颠倒），2/0/3 = 中等问题（Δ1 软颠倒、冠军尚在）。
同样是 exact/±1/±2 各一张，结论完全不同——故增设**组内成对顺序颠倒率**，按用户星差分层：
  - 硬颠倒（Δ≥2）：2 星差通常代表硬差异，方向用反 = 定性错误；
  - 软颠倒（Δ1）：0:1 差距不大，置信度低，单列；
  - 冠军涉入颠倒：胜者被任一败者压过（≈ 冠军丢失的成对版本）。

只算**组内**（硬可比域）；组间无硬性比对依据（用户 2026-08-30 直觉，与宪法 §0.3 一致），
跨组/跨段排序正确性由台阶①（derived Δ≥2 对）与台阶③（团顶召回）承担，不在本脚本。

用法（仓根 D:/Git/PhotoViewer 下）：
    PYTHONUTF8=1 Tools/.venv/Scripts/python.exe Training/audit/golden_order_eval.py
      --readback D:/PhotoDB/dataset/golden_star3/golden_star_readback.tsv
      --key D:/PhotoDB/dataset/golden_star3/golden_batch3_key.csv
      --scores fw1ep3=Training/train/out/m5_cvfuse_hz20_fw1/scores_ep3.csv
               ens=Training/train/out/m8_best/scores_ens.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


def load_scores(spec: str) -> tuple[str, dict[str, float]]:
    name, path = spec.split("=", 1)
    d = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        d[r["fingerprint"]] = float(r["score"])
    return name, d


def main() -> int:
    ap = argparse.ArgumentParser(description="组内顺序颠倒率（成对排序口径）")
    ap.add_argument("--readback", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--scores", nargs="+", required=True, help="name=path.csv（fingerprint,score）")
    ap.add_argument("--out", default=None, help="报告 md（默认 <readback 同目录>/golden_order_report.md）")
    args = ap.parse_args()

    key = {r["fingerprint"]: r for r in csv.DictReader(open(args.key, encoding="utf-8-sig"))}
    stars: dict[str, dict[str, int]] = defaultdict(dict)
    for ln in open(args.readback, encoding="utf-8"):
        if ln.startswith("gid"):
            continue
        g, a, fp, us, *_ = ln.rstrip("\n").split("\t")
        stars[g][fp] = int(us)
    out_md = Path(args.out) if args.out else Path(args.readback).parent / "golden_order_report.md"

    models = [load_scores(s) for s in args.scores]
    L: list[str] = ["# 组内顺序颠倒率报告", ""]
    L.append(f"- 考卷：{args.readback}（{len(stars)} 组）")

    for name, sc in models:
        # 逐组逐对（用户异星对才入考；同星=不可判剔除）
        stat = Counter()          # (gap_class, kind) -> n
        g_hard = g_soft = g_champ_inv = g_total = 0
        n_cov_miss = 0
        for gid, mem in sorted(stars.items()):
            fps = [f for f in mem if f in sc]
            if len(fps) < len(mem):
                n_cov_miss += 1
            if len(fps) < 2:
                continue
            g_total += 1
            champ = max(mem, key=lambda f: mem[f])      # 用户冠军（并列取其一，仅用于涉入统计）
            has_hard = has_soft = has_champ = False
            for fi, fj in combinations(fps, 2):
                du = mem[fi] - mem[fj]
                if du == 0:
                    continue                            # 同星=不可判
                ds = sc[fi] - sc[fj]
                inv = (du > 0) != (ds > 0) and abs(ds) > 1e-9
                cls = "hard" if abs(du) >= 2 else "soft"
                stat[(cls, "pairs")] += 1
                if inv:
                    stat[(cls, "inv")] += 1
                    if cls == "hard":
                        has_hard = True
                    else:
                        has_soft = True
                    if champ in (fi, fj):
                        has_champ = True
                        stat[("champ", "inv")] += 1
                if champ in (fi, fj):
                    stat[("champ", "pairs")] += 1
            g_hard += has_hard
            g_soft += has_soft
            g_champ_inv += has_champ
        n = lambda c: stat[(c, "inv")] / max(stat[(c, "pairs")], 1)
        L += ["",
              f"## {name}",
              f"- 覆盖组 {g_total}/{len(stars)}（缺分跳过成员 {n_cov_miss} 组）",
              f"- **硬颠倒（Δ≥2）：{stat[('hard','inv')]}/{stat[('hard','pairs')]} = {n('hard'):.3f}**"
              f"（{g_hard} 组至少 1 处）",
              f"- 软颠倒（Δ1）：{stat[('soft','inv')]}/{stat[('soft','pairs')]} = {n('soft'):.3f}"
              f"（{g_soft} 组至少 1 处）",
              f"- 冠军涉入颠倒：{stat[('champ','inv')]}/{stat[('champ','pairs')]} = {n('champ'):.3f}"
              f"（{g_champ_inv} 组冠军被压）"]

        # 按 split 细分（key 有 split 列时）
        if key and "split" in next(iter(key.values())):
            for sp in ("test", "train"):
                sub = {g: m for g, m in stars.items()
                       if any(key.get(f, {}).get("split") == sp for f in m)}
                if not sub:
                    continue
                hp = hi = sp_ = si = 0
                for gid, mem in sub.items():
                    fps = [f for f in mem if f in sc]
                    for fi, fj in combinations(fps, 2):
                        du = mem[fi] - mem[fj]
                        if du == 0:
                            continue
                        ds = sc[fi] - sc[fj]
                        inv = (du > 0) != (ds > 0) and abs(ds) > 1e-9
                        if abs(du) >= 2:
                            hp += 1; hi += inv
                        else:
                            sp_ += 1; si += inv
                L.append(f"  - [{sp}] 硬 {hi}/{hp} = {hi / max(hp, 1):.3f} · "
                         f"软 {si}/{sp_} = {si / max(sp_, 1):.3f}")

    report = "\n".join(L) + "\n"
    out_md.write_text(report, encoding="utf-8")
    print(report)
    print(f"[OK] {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
