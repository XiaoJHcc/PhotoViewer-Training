"""
cluster_mine.py — 相似团挖掘与代表资格审计（M2 代表池地基；用户 2026-07-19 裁定保留复用）

背景（EXECUTION-LOG 2026-07-19"代表有效性验证"条）：时间段（>10min gap 切 120 段）是
"一串拍摄活动"而非"一堆相似照片"——段内配对 cos 中位仅 0.42、段均 9 团、段封顶照居中度
合格仅 47%。锦标赛的真实操作单位是**相似团**：3★ 是它所在相似 0-3★ 群的代表、4/5★ 是
场级评比幸存者。本脚本把该结构固化，供 M2 代表池与后续任何"按团组织"的逻辑复用。

机制：
    1. 全库按 capture_time 排序，>gap（默认 10min）切拍摄段（与 data_audit §4 / abs_probe 同口径）。
    2. 段内按 CLS 相似度 ≥ thr（默认 0.88——audit §3：cos≥0.88 几乎必然同场景，
       跨段 P99.9=0.725）union-find 聚团。
    3. 每团统计：大小、封顶星、团顶（rating 并列全标）、成员居中度（cos to 团质心）。
    4. --abs-set 给出时，审计 abs_set（200 张盲评尺）与团顶的交集：尺子里的照片多少本身
       就是团顶（可免费双职当代表）、多少只是普通成员（只能当测量/校验点，无代表资格）。

输出（--out 目录）：
    clusters.csv       每张照片一行：fingerprint,event,seg_id,cluster_id,cluster_size,
                       rating,is_cluster_top,centrality
    cluster_summary.md 团结构摘要（大小分布 / 封顶×大小 / 段内团数 / 代表居中度 / 孤立照）

用法：
    Tools/.venv/Scripts/python.exe Training/audit/cluster_mine.py
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "probes"))
from feature_probe import l2_normalize, split_segments          # noqa: E402
from datetime import datetime                                    # noqa: E402

MODEL_ORIG = "dinov3_vits16_f32_518_v1"     # 场景内容视图用原片 CLS（与 audit §3 同）


def load_photos(db_path: str):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = [(fp, int(r), datetime.fromisoformat(str(ct)), str(ev or ""))
                for fp, r, ct, ev in conn.execute(
                    "SELECT fingerprint, rating, capture_time, event_label FROM photos "
                    "WHERE rating IS NOT NULL AND capture_time IS NOT NULL")]
        feats = {fp: np.frombuffer(blob, dtype="<f4").astype(np.float32)
                 for fp, blob in conn.execute(
                     "SELECT fingerprint, cls_vector FROM photo_features WHERE model_id = ?",
                     (MODEL_ORIG,))}
    finally:
        conn.close()
    rows = [r for r in rows if r[0] in feats]
    rows.sort(key=lambda x: x[2])
    return rows, feats


def _union_find(n: int, G: np.ndarray, thr: float) -> np.ndarray:
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    iu = np.triu_indices(n, 1)
    hit = G[iu] >= thr
    for a, b in zip(iu[0][hit], iu[1][hit]):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    return np.array([find(i) for i in range(n)])


def mine(rows, feats, gap: float, thr: float):
    """返回 records：每张照片一行 dict（含团归属/团顶标记/居中度）。"""
    ts = [r[2] for r in rows]
    seg = split_segments(ts, gap)
    R = np.array([r[1] for r in rows])
    E = np.array([r[3] for r in rows])
    X = l2_normalize(np.stack([feats[r[0]] for r in rows]))
    records: list[dict] = []
    cid_seq = 0
    for s in range(int(seg.max()) + 1):
        ii = np.where(seg == s)[0]
        n = len(ii)
        if n == 1:
            lab = np.zeros(1, dtype=int)
        else:
            lab = _union_find(n, X[ii] @ X[ii].T, thr)
        for c in np.unique(lab):
            jj = ii[lab == c]
            m = len(jj)
            r = R[jj]
            if m > 1:
                cen = l2_normalize(X[jj].mean(axis=0))
                centr = X[jj] @ cen
            else:
                centr = np.ones(1)
            cap = int(r.max())
            for k in range(m):
                records.append(dict(
                    fingerprint=rows[jj[k]][0], event=E[jj[k]], seg_id=int(s),
                    cluster_id=cid_seq, cluster_size=m, rating=int(r[k]),
                    is_cluster_top=int(r[k] == cap), centrality=float(centr[k]),
                    cluster_cap=cap))
            cid_seq += 1
    return records, int(seg.max()) + 1


def summarize(records: list[dict], n_seg: int, thr: float) -> str:
    from collections import defaultdict
    clus: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        clus[r["cluster_id"]].append(r)
    sizes = np.array([len(v) for v in clus.values()])
    caps = np.array([v[0]["cluster_cap"] for v in clus.values()])
    per_seg = np.bincount([v[0]["seg_id"] for v in clus.values()], minlength=n_seg)
    singles = [v[0] for v in clus.values() if len(v) == 1]
    single_hi = [r for r in singles if r["rating"] >= 3]

    def central(v):
        tops = [r["centrality"] for r in v if r["is_cluster_top"]]
        med = float(np.median([r["centrality"] for r in v]))
        return float(np.mean(tops)) >= med

    big2 = [v for v in clus.values() if len(v) >= 2]
    big16 = [v for v in clus.values() if len(v) >= 16]
    L = []
    L.append(f"# 相似团结构摘要（cos≥{thr} 段内 union-find，{n_seg} 段）\n")
    L.append(f"- 团总数 **{len(clus)}** | 大小中位 {int(np.median(sizes))} / P90 {int(np.percentile(sizes, 90))} / max {sizes.max()}")
    L.append(f"- 单张团（无近重复=孤立照）**{len(singles)}**（占团 {len(singles)/len(clus):.1%}、占照片 {len(singles)/len(records):.1%}）"
             f"；其中 ≥3★ **{len(single_hi)}** 张（{sum(1 for r in single_hi if r['rating']==5)} 张 5★）")
    L.append(f"- 每段团数：中位 {int(np.median(per_seg))} / P90 {int(np.percentile(per_seg, 90))} / max {per_seg.max()}；单团段 {(per_seg==1).sum()}/{n_seg}")
    L.append(f"- 团代表居中度（团顶居中度≥团中位）：≥2 张团 {sum(map(central, big2))}/{len(big2)} = {np.mean([central(v) for v in big2]):.0%}；"
             f"≥16 张团 {sum(map(central, big16))}/{len(big16)} = {np.mean([central(v) for v in big16]):.0%}")
    L.append("\n## 团封顶星数 × 团大小\n\n| 大小桶 | 0★ | 1★ | 2★ | 3★ | 4★ | 5★ |\n|---|---|---|---|---|---|---|")
    for lo, hi, lab in [(1, 1, "单张"), (2, 4, "2-4"), (5, 15, "5-15"), (16, 10**9, "16+")]:
        m = (sizes >= lo) & (sizes <= hi)
        L.append(f"| {lab} | " + " | ".join(str(int(((caps == k) & m).sum())) for k in range(6)) + " |")
    if big16:
        dist = np.zeros(6)
        for v in big16:
            for r in v:
                dist[r["rating"]] += 1
        L.append(f"\n≥16 张团平均星分布（金字塔校验）：{np.round(dist/len(big16), 1).tolist()}")
    return "\n".join(L) + "\n"


def audit_abs_set(records: list[dict], key_path: str) -> str:
    by_fp = {r["fingerprint"]: r for r in records}
    key = list(csv.DictReader(open(key_path, encoding="utf-8-sig")))
    hit = [by_fp[k["fingerprint"]] for k in key if k["fingerprint"] in by_fp]
    L = [f"\n=== abs_set（{len(key)} 张盲评尺）× 团顶交集审计 ==="]
    L.append(f"回库命中 {len(hit)}/{len(key)}")
    buckets = {"单张团": lambda r: r["cluster_size"] == 1,
               "2-4 张团": lambda r: 2 <= r["cluster_size"] <= 4,
               "5-15 张团": lambda r: 5 <= r["cluster_size"] <= 15,
               "16+ 张团": lambda r: r["cluster_size"] >= 16}
    L.append("所在团大小 | 张数 | 其中团顶（可免费双职当代表）| 非团顶（仅测量/校验点）")
    for lab, f in buckets.items():
        sub = [r for r in hit if f(r)]
        top = sum(r["is_cluster_top"] for r in sub)
        L.append(f"  {lab:<8} | {len(sub):>3} | {top:>3} | {len(sub)-top:>3}")
    top_all = sum(r["is_cluster_top"] for r in hit)
    L.append(f"  合计     | {len(hit):>3} | {top_all:>3} | {len(hit)-top_all:>3}")
    big16_tops = {(r["cluster_id"]) for r in records if r["cluster_size"] >= 16 and r["is_cluster_top"]}
    big16_hit = {(r["cluster_id"]) for r in hit if r["cluster_size"] >= 16}
    big16_top_hit = {(r["cluster_id"]) for r in hit if r["cluster_size"] >= 16 and r["is_cluster_top"]}
    L.append(f"16+ 大团共 {len(big16_tops)} 个：abs_set 触及 {len(big16_hit)} 个，其中团顶在尺内 {len(big16_top_hit)} 个")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="D:/PhotoDB/dataset/photos_dataset.db")
    ap.add_argument("--gap", type=float, default=10.0)
    ap.add_argument("--thr", type=float, default=0.88)
    ap.add_argument("--abs-set-key", default="D:/PhotoDB/dataset/abs_set_key.csv")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "out" / "clusters"))
    args = ap.parse_args()

    rows, feats = load_photos(args.db)
    records, n_seg = mine(rows, feats, args.gap, args.thr)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "clusters.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    summary = summarize(records, n_seg, args.thr)
    (out / "cluster_summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    if args.abs_set_key and Path(args.abs_set_key).exists():
        print(audit_abs_set(records, args.abs_set_key))
    print(f"\n[OK] 写出 {out / 'clusters.csv'} / cluster_summary.md（{len(records)} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
