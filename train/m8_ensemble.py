"""
m8_ensemble.py — 系综合成 + 三级台阶评估（可复现固化，替代 2026-07-27 临时脚本）

输入：任意多个 scores.csv（fingerprint,event,seg_id,split,rating,score，全库 9418 行），
每个成员 z-score（默认全库口径）后按权重合成。权重缺省均权；--search 时在 val 上
做贪心前向选择（可重复选同一成员 = 整数权重），目标 = val 台阶①综合（derived dstar==2
与 abs 对 Δ≥2 的均值）。

三级台阶考卷（test，口径与 EXECUTION-LOG 2026-07-27 对齐 + A1 新考卷）：
  ① 高分段任意两张谁强谁弱：
     - derived 对（两端皆团顶/孤立，构造即清洁）：dstar==2 / dstar>=3 / 全体；
     - 全体干净对（window+global+derived 限两端团顶）：dstar>=2 / >=3；
     - global 对 dstar==2；
     - **abs 盲评对（A1 新考卷，audit/out/abs_pairs）**：全体 / Δ==1 / Δ>=2，分 set 拆报；
  ② 团内选优：test 有星差团 top-1（模型最高分成员是否真最高星），chance 对照；
     另报 Δ>=2「真胜负」团子集；
  ③ 团顶排序：test 事件内候选=团顶，取 Top 12.5%，≥3★ 团顶召回（pooled）。

验证基线：lora+lorap+laion+l518 均权 → derived dstar==2 应复现 0.755（n=1422）。

用法（仓根 D:/Git/PhotoViewer 下）：
    PYTHONUTF8=1 Tools/.venv/Scripts/python.exe Training/train/m8_ensemble.py \
        --scores lora=Training/train/out/m5_lora/scores_ep3.csv \
                 lorap=Training/train/out/m5_lora_patch/scores_ep3.csv \
                 laion=Training/train/out/m7_extprobe/scores.csv \
                 l518=Training/train/out/m4_l518_cls2/scores.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

M3_DEFAULT = "D:/Git/PhotoViewer/Training/audit/out/m3_pairs"
ABS_PAIRS_DEFAULT = "D:/Git/PhotoViewer/Training/audit/out/abs_pairs"


def load_meta(m3_dir: str):
    meta = {}
    for r in csv.DictReader(open(f"{m3_dir}/photos.csv", encoding="utf-8-sig")):
        meta[r["fingerprint"]] = dict(
            event=r["event"], seg=int(r["seg_id"]), cluster=int(r["cluster_id"]),
            csize=int(r["cluster_size"]), is_top=int(r["is_cluster_top"]),
            rating=int(r["rating_raw"]), s=float(r["abs_score"]), split=r["split"])
    return meta


def load_scores(path: str):
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        out[r["fingerprint"]] = float(r["score"])
    return out


def load_m3_pairs(m3_dir: str, split: str, meta):
    """(fi, fj, y, ptype, dstar)；derived 的 y 取潜分符号，其余取原始星符号（同 m4 load_pairs）。"""
    out = []
    for r in csv.DictReader(open(f"{m3_dir}/pairs_{split}.csv", encoding="utf-8-sig")):
        pi, pj = meta.get(r["fp_i"]), meta.get(r["fp_j"])
        if pi is None or pj is None:
            continue
        if r["ptype"] == "derived":
            y = np.sign(pi["s"] - pj["s"])
        else:
            y = np.sign(pi["rating"] - pj["rating"])
        if y == 0:
            continue
        out.append((r["fp_i"], r["fp_j"], int(y), r["ptype"], int(r["dstar"])))
    return out


def load_abs_pairs(abs_dir: str, split: str):
    """(fi, fj, y=+1 胜者在前, dstar, set_tag)；set_tag 从文件顺序不可知，按 id 段拆：
    abs 对与 m2p 对分文件生成、前后相接——改为读报告不可靠，这里直接两趟按已知集合划分。"""
    out = []
    p = Path(abs_dir) / f"pairs_{split}.csv"
    if not p.exists():
        return out
    for r in csv.DictReader(open(p, encoding="utf-8-sig")):
        out.append((r["fp_i"], r["fp_j"], 1, int(r["dstar"])))
    return out


def acc_pairs(pairs, s_of, filt=None):
    ok, n = 0, 0
    for fi, fj, y, *rest in pairs:
        if filt and not filt(rest):
            continue
        if fi not in s_of or fj not in s_of:
            continue
        ok += int(np.sign(s_of[fi] - s_of[fj]) == y)
        n += 1
    return (ok / n if n else float("nan"), n)


def step1(s_of, pairs_m3, pairs_abs):
    derived = [p for p in pairs_m3 if p[3] == "derived"]
    out = {}
    out["derived_d2"] = acc_pairs(derived, s_of, lambda r: r[1] == 2)
    out["derived_d3p"] = acc_pairs(derived, s_of, lambda r: r[1] >= 3)
    out["derived_all"] = acc_pairs(derived, s_of)
    out["abs_all"] = acc_pairs(pairs_abs, s_of)
    out["abs_d1"] = acc_pairs(pairs_abs, s_of, lambda r: r[0] == 1)
    out["abs_d2p"] = acc_pairs(pairs_abs, s_of, lambda r: r[0] >= 2)
    return out


def step2(s_of, meta, split):
    by_cluster = defaultdict(list)
    for fp, m in meta.items():
        if m["split"] == split and m["csize"] >= 2:
            by_cluster[(m["event"], m["seg"], m["cluster"])].append(fp)
    res = {}
    for tag, min_gap in (("all", 1), ("d2", 2)):
        hits = ch = n = 0
        for members in by_cluster.values():
            rs = np.array([meta[fp]["rating"] for fp in members])
            if rs.max() - rs.min() < min_gap:
                continue
            sc = np.array([s_of.get(fp, np.nan) for fp in members])
            if np.isnan(sc).any():
                continue
            if meta[members[int(sc.argmax())]]["rating"] == rs.max():
                hits += 1
            ch += (rs == rs.max()).mean()
            n += 1
        res[f"cluster_top1_{tag}"] = (hits / n if n else float("nan"),
                                      ch / n if n else float("nan"), n)
    return res


def step3(s_of, meta, split):
    by_ev = defaultdict(list)
    for fp, m in meta.items():
        if m["split"] == split and m["is_top"]:
            by_ev[m["event"]].append(fp)
    got = need = 0
    for ev, tops in by_ev.items():
        k = max(1, round(len(tops) * 0.125))
        sel = sorted(tops, key=lambda fp: -s_of[fp])[:k]
        got += sum(1 for fp in sel if meta[fp]["rating"] >= 3)
        need += sum(1 for fp in tops if meta[fp]["rating"] >= 3)
    return {"top_recall_125": (got / need if need else float("nan"), got, need)}


def combine(members, weights, fps, zscope, meta, split):
    z = {}
    for name, s in members.items():
        v = np.array([s[fp] for fp in fps])
        if zscope == "split":
            mask = np.array([meta[fp]["split"] == split for fp in fps])
            mu, sd = v[mask].mean(), v[mask].std()
        else:
            mu, sd = v.mean(), v.std()
        z[name] = (v - mu) / (sd or 1.0)
    wsum = sum(weights) or 1.0
    mix = sum(w * z[name] for name, w in zip(members, weights)) / wsum
    return dict(zip(fps, mix))


def fmt_block(title, d):
    L = [f"## {title}\n"]
    for k, v in d.items():
        if len(v) == 2:
            L.append(f"- {k}: **{v[0]:.3f}**（n={v[1]}）")
        else:
            L.append(f"- {k}: **{v[0]:.3f}**（chance {v[1]:.3f}，n={v[2]}）")
    return L


def main() -> int:
    ap = argparse.ArgumentParser(description="系综合成 + 三级台阶评估")
    ap.add_argument("--scores", nargs="+", required=True, help="name=path 列表")
    ap.add_argument("--weights", default=None, help="逗号分隔权重，缺省均权")
    ap.add_argument("--zscope", default="all", choices=["all", "split"],
                    help="z-score 统计域：全库 / 仅被评 split")
    ap.add_argument("--m3", default=M3_DEFAULT)
    ap.add_argument("--abs-pairs", default=ABS_PAIRS_DEFAULT)
    ap.add_argument("--search", action="store_true", help="val 上贪心前向选择权重")
    ap.add_argument("--search-iters", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    meta = load_meta(args.m3)
    members = {}
    for spec in args.scores:
        name, path = spec.split("=", 1)
        members[name] = load_scores(path)
    fps = [fp for fp in meta if all(fp in m for m in members.values())]
    print(f"成员: {list(members)} · 对齐照片 {len(fps)}/{len(meta)}", flush=True)

    pairs_te = load_m3_pairs(args.m3, "test", meta)
    # 干净对标记：两端皆团顶/孤立
    pairs_te = [(fi, fj, y, pt, d, meta[fi]["is_top"] and meta[fj]["is_top"])
                for fi, fj, y, pt, d in pairs_te]
    abs_te = load_abs_pairs(args.abs_pairs, "test")
    pairs_va = load_m3_pairs(args.m3, "val", meta)
    pairs_va = [(fi, fj, y, pt, d, meta[fi]["is_top"] and meta[fj]["is_top"])
                for fi, fj, y, pt, d in pairs_va]
    abs_va = load_abs_pairs(args.abs_pairs, "val")

    names = list(members)
    if args.weights:
        weights = [float(x) for x in args.weights.split(",")]
        assert len(weights) == len(names)
    else:
        weights = [1.0] * len(names)

    if args.search:
        chosen = []          # 成员索引多重集（整数权重）
        best_obj = -1.0
        for _ in range(args.search_iters):
            cand_best, cand_obj = None, -1.0
            for ix in range(len(names)):
                trial = chosen + [ix]
                w = [trial.count(i) for i in range(len(names))]
                s_of = combine(members, w, fps, args.zscope, meta, "val")
                m = step1(s_of, pairs_va, abs_va)
                obj = np.nanmean([m["derived_d2"][0], m["abs_d2p"][0]])
                if obj > cand_obj:
                    cand_best, cand_obj = ix, obj
            if cand_obj <= best_obj + 1e-4:
                break
            chosen.append(cand_best)
            best_obj = cand_obj
            print(f"search +{names[cand_best]} → val obj {best_obj:.4f}", flush=True)
        weights = [chosen.count(i) for i in range(len(names))] or weights
        print(f"search 结果权重: {dict(zip(names, weights))}", flush=True)

    s_of = combine(members, weights, fps, args.zscope, meta, "test")

    L = ["# 系综三级台阶评估（m8_ensemble）\n",
         f"成员×权重: {dict(zip(names, weights))} · zscope={args.zscope}\n"]
    m1 = step1(s_of, pairs_te, abs_te)
    clean_te = [p for p in pairs_te if p[5]]
    m1["clean_d2"] = acc_pairs(clean_te, s_of, lambda r: r[1] >= 2)
    m1["clean_d3"] = acc_pairs(clean_te, s_of, lambda r: r[1] >= 3)
    glob = [p for p in pairs_te if p[3] == "global"]
    m1["global_d2"] = acc_pairs(glob, s_of, lambda r: r[1] == 2)
    L.append("## 台阶① 高分段任意两张（test）\n")
    for k, (a, n) in m1.items():
        L.append(f"- {k}: **{a:.3f}**（n={n}）")
    m2 = step2(s_of, meta, "test")
    L.append("\n## 台阶② 团内选优（test）\n")
    for k, (a, c, n) in m2.items():
        L.append(f"- {k}: **{a:.3f}**（chance {c:.3f}，n={n} 团）")
    m3r = step3(s_of, meta, "test")
    L.append("\n## 台阶③ 团顶≥3★召回@12.5%（test，候选=团顶）\n")
    a, g, nd = m3r["top_recall_125"]
    L.append(f"- recall: **{a:.3f}**（{g}/{nd}）")

    text = "\n".join(L) + "\n"
    print(text)
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        tag = "m8_report.md"
        (Path(args.out) / tag).write_text(text, encoding="utf-8")
        with open(Path(args.out) / "scores_ens.csv", "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["fingerprint", "score"])
            for fp in fps:
                wr.writerow([fp, f"{s_of[fp]:.4f}"])
        (Path(args.out) / "weights.json").write_text(
            json.dumps(dict(zip(names, weights)), ensure_ascii=False), encoding="utf-8")
        print(f"[OK] {args.out}/{tag} + scores_ens.csv + weights.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
