"""
tops_exam_eval.py — 绝对序考卷：团顶 ∪ 孤立照（台阶③扶正版，2026-08-31 用户定口径）

定位：去重后每团只剩团顶，**全库每张照片最终都以"孤立照"身份过这把尺**——团内选优（台阶②）
只管一半链路，本考卷管另一半：团顶/孤立照之间的绝对美学序（高星标定的产品主战场）。

口径（2026-08-31 用户确认）：
  - 总体 = clusters.csv 中 is_cluster_top=1（含孤立照：cluster_size=1 者即自身团顶）；
    原始星级清洁度与代表身份绑定（宪法 v1.8），团顶/孤立照的星未受"重复压低"污染；
  - **事件内域**：只组同事件对（不同事件原星不可直接比，07-19 规则①）；异星成对，
    Δ≥2 = 硬对（方向错=定性错误）、Δ1 = 软对（参照）；指标 = 对级一致率；
  - **好片带召回**：每 test 事件内，≥4★ 团顶/孤立照被模型分排进事件内 top-N（N=该事件
    ≥4★ 数）的比例（>3★ 同报）；"稳定找出好片"的绝对序版本；
  - **跨事件域**：原星跨事件不可比，只走盲评锚点派生序——A1 abs_pairs（--abs-pairs），
    按 dstar 1 / ≥2 分层报一致率（m2_pool_ext 扩锚后新事件入域，口径不变）；
  - 考卷主报 test 事件（train/val 仅作诊断参照，泄漏铁律不变）。

用法（仓根 D:/Git/PhotoViewer 下）：
    PYTHONUTF8=1 Tools/.venv/Scripts/python.exe Training/audit/tops_exam_eval.py
      --scores ens=Training/train/out/m8_best/scores_ens.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLUSTERS = ROOT / "audit" / "out" / "clusters" / "clusters.csv"
SPLIT = ROOT / "audit" / "out" / "m3_pairs" / "split.json"
ABS_PAIRS = ROOT / "audit" / "out" / "abs_pairs" / "pairs_test.csv"
OUT = ROOT / "audit" / "out" / "tops_exam"


def main() -> int:
    ap = argparse.ArgumentParser(description="绝对序考卷：团顶∪孤立照（事件内 + 跨事件锚点两域）")
    ap.add_argument("--scores", nargs="+", required=True, help="name=path.csv（fingerprint,score）")
    ap.add_argument("--abs-pairs", default=str(ABS_PAIRS))
    ap.add_argument("--out", default=str(OUT / "tops_exam_report.md"))
    args = ap.parse_args()

    split = json.load(open(SPLIT, encoding="utf-8"))
    ev_split = {ev: s for s, evs in split.items() for ev in evs}
    # 总体：团顶∪孤立照
    pop = defaultdict(list)     # event -> [(fp, rating, cluster_size)]
    for r in csv.DictReader(open(CLUSTERS, encoding="utf-8-sig")):
        if r["is_cluster_top"] != "1":
            continue
        pop[r["event"]].append((r["fingerprint"], int(r["rating"]), int(r["cluster_size"])))
    print(f"总体（团顶∪孤立照）: {sum(len(v) for v in pop.values())} / {len(pop)} 事件")

    sources = {}
    for spec in args.scores:
        name, path = spec.split("=", 1)
        sources[name] = {r["fingerprint"]: float(r["score"])
                         for r in csv.DictReader(open(path, encoding="utf-8-sig"))}

    L: list[str] = ["# 绝对序考卷报告（团顶∪孤立照）", ""]

    # ── 域 A：事件内成对一致率（Δ 分层）──
    L.append("## 域 A · 事件内对级一致率（Δ≥2 硬对 / Δ1 软对）")
    for name, sc in sources.items():
        for sp in ("test", "val", "train"):
            st = Counter()
            for ev, members in pop.items():
                if ev_split.get(ev) != sp:
                    continue
                ms = [(fp, rt) for fp, rt, _ in members if fp in sc]
                for (fi, ri), (fj, rj) in combinations(ms, 2):
                    d = ri - rj
                    if d == 0:
                        continue
                    ds = sc[fi] - sc[fj]
                    cls = "hard" if abs(d) >= 2 else "soft"
                    st[(cls, "n")] += 1
                    st[(cls, "inv")] += (d > 0) != (ds > 0) and abs(ds) > 1e-9
            if not st:
                continue
            acc = lambda c: 1 - st[(c, "inv")] / max(st[(c, "n")], 1)
            L.append(f"- [{name}][{sp}] 硬对一致 {acc('hard'):.3f}（{st[('hard','n')]} 对）· "
                     f"软对一致 {acc('soft'):.3f}（{st[('soft','n')]} 对）")
    L.append("")

    # ── 域 A'：好片带召回（test 事件，top-N 自配额）──
    L.append("## 域 A' · 好片带召回（test 事件，top-N 自配额 = 事件内该星带数量）")
    for name, sc in sources.items():
        for band, lo in (("≥4★", 4), ("≥3★", 3)):
            hits = tot = ne = 0
            chance = []
            for ev, members in pop.items():
                if ev_split.get(ev) != "test":
                    continue
                ms = [(fp, rt) for fp, rt, _ in members if fp in sc]
                hi = [m for m in ms if m[1] >= lo]
                if not hi:
                    continue
                ne += 1
                n = len(hi)
                ranked = {fp for fp, _ in sorted(ms, key=lambda m: -sc[m[0]])[:n]}
                hits += sum(1 for fp, _ in hi if fp in ranked)
                tot += n
                chance.append(n / len(ms))
            if tot:
                L.append(f"- [{name}] {band} 召回 {hits}/{tot} = {hits / tot:.3f}（{ne} 事件；"
                         f"随机基线 {sum(chance) / len(chance):.3f}）")
    L.append("")

    # ── 域 B：跨事件锚点对（abs_pairs，盲评尺）──
    L.append("## 域 B · 跨事件锚点对一致率（abs_pairs test，盲评尺 Δstar 分层）")
    ap_path = Path(args.abs_pairs)
    if ap_path.exists():
        pairs = [r for r in csv.DictReader(open(ap_path, encoding="utf-8-sig"))
                 if r["dstar"] not in ("", "nan") and int(r["dstar"]) >= 1]
        for name, sc in sources.items():
            st = Counter()
            for r in pairs:
                if r["fp_i"] not in sc or r["fp_j"] not in sc:
                    continue
                d = int(r["dstar"])
                ok = sc[r["fp_i"]] > sc[r["fp_j"]]      # abs_pairs 约定：i 胜 j
                cls = "hard" if d >= 2 else "soft"
                st[(cls, "n")] += 1
                st[(cls, "ok")] += ok
            acc = lambda c: st[(c, "ok")] / max(st[(c, "n")], 1)
            L.append(f"- [{name}] 硬对(Δ≥2) {acc('hard'):.3f}（{st[('hard','n')]}）· "
                     f"软对(Δ1) {acc('soft'):.3f}（{st[('soft','n')]}）")
    else:
        L.append(f"- [skip] {ap_path} 不存在")
    L.append("")
    L.append("判读参照：域 A 硬对 = 定性错误率（越低越好）；域 A' ≥4★ 召回 = \"稳定找出好片\""
             "（台阶③ 扶正口径）；域 B = 跨事件绝对尺（扩锚后新事件入域）。")

    report = "\n".join(L) + "\n"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"[OK] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
