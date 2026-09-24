"""
abs_pair_gen.py — 盲评绝对标签 → 第五监督源训练/考卷对（A1 臂）

背景（2026-07-28 方案）：模型迄今学的是 M2 压缩尺（s=g(段内星)+b_seg 派生的 derived 对），
当时 405 条盲评记录主要用于拟合压缩；历史 A1 已尝试直接监督。当前读取三个原始会话：
  - abs_set（200）、原始 m2_pool（205）与扩充 m2_pool（274）各自配对，胜者在前；
  - 同一指纹复测合并为星级区间，只生成区间严格分离的偏好，不制造自配对或多数真值；
  - **不跨会话组对**（会话间尺度漂移未验证；每个会话内部单独构造）；
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
import hashlib
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # Training/
DS = Path("D:/PhotoDB/dataset")
M3 = ROOT / "audit" / "out" / "m3_pairs"
OUT = ROOT / "audit" / "out" / "abs_pairs"

SETS = [
    ("abs", DS / "abs_set_key.csv", DS / "abs_set_ratings.tsv"),
    ("m2p_original", DS / "m2_pool_key.backup-2026-09-19.csv", DS / "m2_pool_ratings.backup-2026-09-19.tsv"),
    ("m2p_extension", DS / "m2_pool_ext_key.csv", DS / "m2_pool_ext_ratings.tsv"),
]


def load_set(key_path: Path, tsv_path: Path):
    """读取一次标注会话，将复测合并为每张照片的星级区间，保留不确定性。"""
    key = {r["new_name"]: r for r in csv.DictReader(open(key_path, encoding="utf-8-sig"))}
    rated = {n: int(v) for n, v in
             (ln.rstrip("\n").split("\t") for ln in open(tsv_path, encoding="utf-8"))}
    if set(key) != set(rated) or any(rating not in range(6) for rating in rated.values()):
        raise ValueError(f"会话标注不完整或星级非法: {tsv_path}")
    by_fingerprint = defaultdict(list)
    events = {}
    for name, r0 in rated.items():
        k = key.get(name)
        if k is None:
            raise ValueError(f"标注文件含未知名称: {name}")
        ev = k["event_label"]
        # 旧批（D:/PhotoDB/20240212）在 photos.csv/split.json 中事件名为空串
        if ev == "20240212":
            ev = ""
        fingerprint = k["fingerprint"]
        if fingerprint in events and events[fingerprint] != ev:
            raise ValueError(f"复测事件不一致: {fingerprint}")
        events[fingerprint] = ev
        by_fingerprint[fingerprint].append(r0)
    return [(fingerprint, min(ratings), max(ratings), events[fingerprint])
            for fingerprint, ratings in by_fingerprint.items()]


def main() -> int:
    """按事件划分生成会话内确定偏好，输出来源哈希及去重后的配对报告。"""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m3", type=Path, default=M3)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    m3_dir, out_dir = args.m3, args.out
    split_of = {}
    sp = json.load(open(m3_dir / "split.json", encoding="utf-8"))
    for split in ("train", "val", "test"):
        events = sp[split]
        for ev in events:
            if ev in split_of:
                raise ValueError(f"事件划分重复: {ev}")
            split_of[ev] = split

    metadata = {row["fingerprint"]: row for row in
                csv.DictReader(open(m3_dir / "photos.csv", encoding="utf-8-sig"))}
    sessions = {}
    seen_fingerprints = set()
    for name, key_path, tsv_path in SETS:
        rows = load_set(key_path, tsv_path)
        fingerprints = {row[0] for row in rows}
        if fingerprints - metadata.keys() or fingerprints & seen_fingerprints:
            raise ValueError(f"会话含库外照片或与另一会话重叠，须先审计: {name}")
        for fingerprint, _, _, event in rows:
            if metadata[fingerprint]["event"] != event or metadata[fingerprint]["split"] != split_of[event]:
                raise ValueError(f"事件或划分身份不一致: {fingerprint}")
        seen_fingerprints.update(fingerprints)
        sessions[name] = rows

    out_dir.mkdir(parents=True, exist_ok=True)
    writers, fh = {}, {}
    for split in ("train", "val", "test"):
        f = open(out_dir / f"pairs_{split}.csv", "w", newline="", encoding="utf-8")
        w = csv.writer(f)
        w.writerow(["fp_i", "fp_j", "ptype", "weight", "cos", "cv_dist", "dstar", "s_gap", "source"])
        writers[split], fh[split] = w, f

    stats = Counter()
    hi_band = Counter()
    sources = {}
    for set_name, key_path, tsv_path in SETS:
        sources[set_name] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in (key_path, tsv_path)}
        rows = sessions[set_name]
        stats[f"{set_name}_photos"] = len(rows)
        for (fi, low_i, high_i, evi), (fj, low_j, high_j, evj) in combinations(rows, 2):
            if not (low_i > high_j or low_j > high_i):
                stats[f"{set_name}_overlap_drop"] += 1
                continue
            si, sj = split_of.get(evi), split_of.get(evj)
            if si is None or si != sj:
                stats[f"{set_name}_cross_split_drop"] += 1
                continue
            d = low_i - high_j if low_i > high_j else low_j - high_i
            (wi, wj) = (fi, fj) if low_i > high_j else (fj, fi)
            writers[si].writerow([wi, wj, "abs", f"{0.5 if d == 1 else 1.0:.4f}",
                                  "nan", "nan", d, "nan", set_name])
            stats[f"{set_name}_{si}"] += 1
            stats[f"{set_name}_{si}_d1" if d == 1 else f"{set_name}_{si}_d2p"] += 1
            if low_i >= 3 and low_j >= 3:
                hi_band[f"{set_name}_{si}"] += 1
    for f in fh.values():
        f.close()

    L = ["# 盲评绝对对生成报告（A1 第五监督源）\n",
         "- 源：abs_set、原始 m2_pool、扩充 m2_pool 分开配对，不跨会话；复测合并为区间，只有区间严格分离才生成偏好，Δ 取最小差；Δ=1 ×0.5；split 按事件\n",
         "## 计数\n"]
    for k in sorted(stats):
        L.append(f"- {k}: {stats[k]}")
    L.append("\n## 高分段对（两端 ≥3★）\n")
    for k in sorted(hi_band):
        L.append(f"- {k}: {hi_band[k]}")
    L.append("\n## GATE 自检\n")
    L.append(f"- 跨 split 对剔除: {sum(value for key, value in stats.items() if key.endswith('_cross_split_drop'))}"
             "（train/test 混搭对，不入任何文件）")
    L.append("- 跨 set 对: 0（构造上不存在）· tie: 0（构造上不存在）")
    (out_dir / "abs_pairs_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out_dir / "sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[OK] {out_dir}/pairs_*.csv + abs_pairs_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
