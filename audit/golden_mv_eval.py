"""
golden_mv_eval.py — 多数决真值考卷：73 团三轮标星（原星/盲选/复测）Copeland 聚合 → 干净冠军

动机（2026-08-31 三轮对齐发现）：冠军位 ~30% 不稳的主源 = 人单次判断的时间噪声，
多数决（≥2/3）可把真值噪声压到 ~5%（44/46 有多数冠军）。本脚本把三轮聚成
Copeland 多数序（对级多数投票 → 得分排序），考"模型 vs 干净真值"的冠军/前二——
回答"冠军位差距有多少是考卷噪声、多少是真实差距"。

口径：
  - 总体 = golden_exam 73 团中三轮皆决胜的 46 组（任一轮 tie/全0 即弃，宁缺毋滥）；
  - 每轮 = 组内全序（按该轮星数；同星=该轮不分胜负不投票）；对级多数 = ≥2/3 轮同向；
  - Copeland 分 = 对级多数胜场数；冠军 = 唯一最高分（不唯一/无多数冠军 → 该组弃考）；
  - 前二 = 模型所选 ∈ Copeland 前二级；
  - 对照列报"单轮盲选 vs 多数决"的人类一致性（单轮距干净真值有多远）。

用法（仓根 D:/Git/PhotoViewer 下）：
    PYTHONUTF8=1 Tools/.venv/Scripts/python.exe Training/audit/golden_mv_eval.py
      --scores ens=Training/train/out/m8_best/scores_ens.csv ...
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "audit"))
from golden_star_readback import read_rating  # noqa: E402

EXAM_DIR = Path("D:/PhotoDB/dataset/golden_exam")
RETEST_DIR = Path("D:/PhotoDB/dataset/golden_retest")


def load_rounds():
    """返回 rounds: {gid: {fp: star}} × 3 轮（原星/盲选/复测），仅 73 团。"""
    rkey = list(csv.DictReader(open(RETEST_DIR / "golden_retest_key.csv", encoding="utf-8-sig")))
    gids = {r["orig_gid"] for r in rkey}
    fp2ogid = {r["fingerprint"]: r["orig_gid"] for r in rkey}

    orig = defaultdict(dict)
    for r in csv.DictReader(open(EXAM_DIR / "golden_exam_key.csv", encoding="utf-8-sig")):
        if r["gid"] in gids:
            orig[r["gid"]][r["fingerprint"]] = int(r["rating"])
    blind = defaultdict(dict)
    for ln in open(EXAM_DIR / "golden_exam_readback.tsv", encoding="utf-8"):
        if ln.startswith("gid"):
            continue
        g, a, fp, us, *_ = ln.rstrip("\n").split("\t")
        if g in gids:
            blind[g][fp] = int(us)
    rt = defaultdict(dict)
    for r in rkey:
        cands = [f for f in RETEST_DIR.glob(f"{r['r_gid']}_{r['r_anon']}.*")
                 if f.suffix.lower() != ".xmp"]
        if cands:
            rt[r["orig_gid"]][r["fingerprint"]] = read_rating(cands[0])
    return gids, orig, blind, rt


def copeland(stars_by_round: list[dict[str, int]]):
    """对级多数投票 → {fp: 胜场分}；每轮只对异星对投票。"""
    fps = list(stars_by_round[0])
    score = Counter()
    for fi, fj in combinations(fps, 2):
        votes = Counter()
        for rd in stars_by_round:
            d = rd[fi] - rd[fj]
            if d:
                votes[fi if d > 0 else fj] += 1
        win, n = votes.most_common(1)[0] if votes else (None, 0)
        if n >= 2:
            score[win] += 1
    return score


def main() -> int:
    ap = argparse.ArgumentParser(description="多数决真值考卷（73 团三轮 Copeland 聚合）")
    ap.add_argument("--scores", nargs="+", required=True, help="name=path.csv（fingerprint,score）")
    args = ap.parse_args()

    gids, orig, blind, rt = load_rounds()
    sources = {}
    for spec in args.scores:
        name, path = spec.split("=", 1)
        sources[name] = {r["fingerprint"]: float(r["score"])
                         for r in csv.DictReader(open(path, encoding="utf-8-sig"))}

    # 逐组建多数序：要求三轮皆决胜（每轮有唯一冠军）
    groups = {}
    n_skip = Counter()
    for g in sorted(gids):
        rounds = [rd.get(g) for rd in (orig, blind, rt)]
        if any(not r for r in rounds):
            n_skip["缺轮"] += 1
            continue
        decisive = all(max(r.values()) > 0 and
                       sum(1 for v in r.values() if v == max(r.values())) == 1
                       for r in rounds)
        if not decisive:
            n_skip["任一轮非决胜"] += 1
            continue
        sc = copeland(rounds)
        fps = list(rounds[0])
        ranked = sorted(fps, key=lambda f: (-sc.get(f, 0), f))   # 全成员（零胜场=0 分）
        if len(ranked) > 1 and sc.get(ranked[0], 0) == sc.get(ranked[1], 0):
            n_skip["无多数冠军"] += 1
            continue
        groups[g] = dict(champ=ranked[0],
                         top2={f for f in fps
                               if sc.get(f, 0) >= (sc.get(ranked[1], 0) if len(ranked) > 1 else 0)},
                         rounds=rounds)
    print(f"多数决考卷: {len(groups)} 组（剔除 {dict(n_skip)}）")

    # 人类单轮 vs 多数决（参照系）
    n_b = sum(1 for g, d in groups.items()
              if max(d["rounds"][1], key=lambda f: d["rounds"][1][f]) == d["champ"])
    print(f"人类单轮（盲选）vs 多数决冠军一致: {n_b}/{len(groups)} = {n_b / max(len(groups), 1):.3f}")

    print(f"\n{'模型':<14}{'exact(冠军=多数决)':>20}{'前二':>10}")
    for name, sc in sources.items():
        n = ex = t2 = 0
        for g, d in groups.items():
            fps = [f for f in d["rounds"][0] if f in sc]
            if len(fps) < 2:
                continue
            pick = max(fps, key=lambda f: sc[f])
            n += 1
            ex += pick == d["champ"]
            t2 += pick in d["top2"]
        print(f"{name:<14}{ex}/{n} = {ex / max(n, 1):.3f}{'':>6}{t2}/{n} = {t2 / max(n, 1):.3f}")
    print("\n对照：同一模型在单轮盲选标签上的考卷成绩见 golden_exam_eval；"
          "多数决分母下 human line 约 = 单轮一致率本身（0.68-0.70 轮间）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
