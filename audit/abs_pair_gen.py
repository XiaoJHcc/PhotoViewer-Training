"""
abs_pair_gen.py — 盲评绝对标签 → 第五监督源训练/考卷对（A1 臂）

背景（2026-07-28 方案）：模型迄今学的是 M2 压缩尺（s=g(段内星)+b_seg 派生的 derived 对），
405 张盲评（用户亲手全量程绝对尺）只用于拟合压缩、从未直接当监督。本脚本把盲评直接
转成成对监督：
  - abs_set（200）与 m2_pool（205）各自 set 内全配对（Δrating≥1），胜者在前；
  - **不跨 set 组对**（两次标注相隔一周，会话漂移未验证；set 内序各自干净）；
  - Δ=1 权重 ×0.5（整数舍入噪声教义，与 M2 同）；Δ≥2 权重 1.0；
  - split 按事件（split.json）：对两端同 split 才保留（train 对进训练，test/val 对作考卷）；
  - 高分段（两端 ≥3★）对是台阶③监督稀薄区的直接补强。

输出 audit/out/abs_pairs/：pairs_{train,val,test}.csv（schema 与 m3 对齐，ptype=abs，
cos/cv_dist/s_gap=nan）+ abs_pairs_report.md。

用法（仓根 D:/Git/PhotoViewer 下）：
    Tools/.venv/Scripts/python.exe Training/audit/abs_pair_gen.py
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # Training/
DS = Path("D:/PhotoDB/dataset")
M3 = ROOT / "audit" / "out" / "m3_pairs"
OUT = ROOT / "audit" / "out" / "abs_pairs"

SETS = [
    ("abs", DS / "abs_set_key.csv", DS / "abs_set_ratings.tsv"),
    ("m2p", DS / "m2_pool_key.csv", DS / "m2_pool_ratings.tsv"),
]


def load_set(key_path: Path, tsv_path: Path):
    key = {r["new_name"]: r for r in csv.DictReader(open(key_path, encoding="utf-8-sig"))}
    rated = {n: int(v) for n, v in
             (ln.rstrip("\n").split("\t") for ln in open(tsv_path, encoding="utf-8"))}
    out = []   # (fingerprint, rating, event)
    for name, r0 in rated.items():
        k = key.get(name)
        if k is None:
            continue
        ev = k["event_label"]
        # 旧批（D:/PhotoDB/20240212）在 photos.csv/split.json 中事件名为空串
        if ev == "20240212":
            ev = ""
        out.append((k["fingerprint"], r0, ev))
    return out


def main() -> int:
    split_of = {}
    sp = json.load(open(M3 / "split.json", encoding="utf-8"))
    for split, events in sp.items():
        for ev in events:
            split_of[ev] = split

    known_fp = {r["fingerprint"] for r in
                csv.DictReader(open(M3 / "photos.csv", encoding="utf-8-sig"))}

    OUT.mkdir(parents=True, exist_ok=True)
    writers, fh = {}, {}
    for split in ("train", "val", "test"):
        f = open(OUT / f"pairs_{split}.csv", "w", newline="", encoding="utf-8")
        w = csv.writer(f)
        w.writerow(["fp_i", "fp_j", "ptype", "weight", "cos", "cv_dist", "dstar", "s_gap"])
        writers[split], fh[split] = w, f

    stats = Counter()
    hi_band = Counter()
    for set_name, key_path, tsv_path in SETS:
        rows = [(fp, r, ev) for fp, r, ev in load_set(key_path, tsv_path) if fp in known_fp]
        stats[f"{set_name}_photos"] = len(rows)
        for (fi, ri, evi), (fj, rj, evj) in combinations(rows, 2):
            if ri == rj:
                continue
            si, sj = split_of.get(evi), split_of.get(evj)
            if si is None or si != sj:
                stats[f"{set_name}_cross_split_drop"] += 1
                continue
            d = abs(ri - rj)
            (wi, wj) = (fi, fj) if ri > rj else (fj, fi)     # 胜者在前
            writers[si].writerow([wi, wj, "abs", f"{0.5 if d == 1 else 1.0:.4f}",
                                  "nan", "nan", d, "nan"])
            stats[f"{set_name}_{si}"] += 1
            stats[f"{set_name}_{si}_d1" if d == 1 else f"{set_name}_{si}_d2p"] += 1
            if ri >= 3 and rj >= 3:
                hi_band[f"{set_name}_{si}"] += 1
    for f in fh.values():
        f.close()

    L = ["# 盲评绝对对生成报告（A1 第五监督源）\n",
         "- 源：abs_set 200 + m2_pool 205（set 内全对，不跨 set）；Δ=1 ×0.5；split 按事件\n",
         "## 计数\n"]
    for k in sorted(stats):
        L.append(f"- {k}: {stats[k]}")
    L.append("\n## 高分段对（两端 ≥3★）\n")
    for k in sorted(hi_band):
        L.append(f"- {k}: {hi_band[k]}")
    L.append("\n## GATE 自检\n")
    L.append(f"- 跨 split 对剔除: {stats['abs_cross_split_drop'] + stats['m2p_cross_split_drop']}"
             "（train/test 混搭对，不入任何文件）")
    L.append("- 跨 set 对: 0（构造上不存在）· tie: 0（构造上不存在）")
    (OUT / "abs_pairs_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[OK] {OUT}/pairs_*.csv + abs_pairs_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
