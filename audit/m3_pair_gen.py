"""
m3_pair_gen.py — M3 训练对生成（plan-3-2 §6.2 v1.0 + 宪法 §0.3 v1.8 清洁度规则）

输入：clusters.csv（团结构）+ photos 表（capture_time/rating/cv_grid/is_retouched）
     + 干净潜分（m2_offset_fit --exclude-events 对 test/val 锚点剔除后的 latent_scores，
       决策 9 反泄漏条款）+ S-orig CLS（cos 相似度）。

split（决策 9，草案待用户确认）：test = {2026-3-14 茶博, 2026-4-19 虎跑, 2026-4-25 良渚版本}，
val = {2026-1-10 祥睦桥}，train = 其余 7 事件。**金标准集后续必须从 test 事件取**（决策 11）。
所有配对两端同 split（test/val 事件的对只作 M4 评估集）。

配对类型（参数全冻结，§6.2 决策 5-8）：
    A. window  段内滑窗对（高权）：段内时间序滑窗 20 张，rating 不同；cos<0.73 → w=0 排除，
       否则 w = 相似度映射(cos) × Δ降权；tie：cos≥0.98 且 CV 距离 < τ_cv=0.078 → 丢弃。
       相似度映射（audit §9）：过点 (0.73,0.3)→(0.786,0.5)→(0.918,0.8)→(0.973,1.0) 分段线性。
       全体照片参与（非团顶的唯一合法配对域，清洁度规则）。
    B. global  ≥3★ 事件内全局对（中权 ×0.5，草案值）：仅清洁照片（团顶/孤立，is_cluster_top=1），
       rating 不同，w = 0.5 × Δ降权；同 tie 规则；不做相似度映射（该比较当年事件全局真实发生，
       宪法决策 2 例外）。
    C. derived 潜分派生对（低权 ×0.2）：仅清洁照片、同 split、跨 cluster、|Δs| > 显著阈值
       （= 训练事件锚点对中 |Δnew|≥2 对的 |Δs| 中位数）；每照片最多 8 个 partner（seed 定），
       跨 segment/跨事件同权。
    Δ降权（决策 7 草案）：|Δrating|=3 → ×0.5；≥4 → ×0.25（仅 A/B 原始星级对）。

输出（--out 目录）：
    photos.csv        主表（决策 10 格式 + split）
    pairs_{train,val,test}.csv   配对（fp_i,fp_j,ptype,weight,cos,cv_dist,dstar,s_gap）
    m3_report.md      计数 / 权重分布 / 孤儿照片 / M3 GATE 自检（无跨 split、窗域正确、
                      无 tie 残留、derived 全清洁、无重复杂糅断言）
    split.json        事件 → split 映射

用法（仓根 D:/Git/PhotoViewer 下）：
    PYTHONUTF8=1 Tools/.venv/Scripts/python.exe Training/audit/m3_pair_gen.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "probes"))
from feature_probe import _cv_aggregate, l2_normalize, pairwise_cv_distance  # noqa: E402

DB_DEFAULT = "D:/PhotoDB/dataset/photos_dataset.db"
CLUSTERS_DEFAULT = "D:/Git/PhotoViewer/Training/audit/out/clusters/clusters.csv"
LATENT_CLEAN_DEFAULT = "D:/Git/PhotoViewer/Training/audit/out/m2_offset_clean/latent_scores.csv"
POOL_KEY_DEFAULT = "D:/PhotoDB/dataset/m2_pool_key.csv"
POOL_TSV_DEFAULT = "D:/PhotoDB/dataset/m2_pool_ratings.tsv"
ABS_KEY_DEFAULT = "D:/PhotoDB/dataset/abs_set_key.csv"
ABS_TSV_DEFAULT = "D:/PhotoDB/dataset/abs_set_ratings.tsv"
MODEL_ORIG = "dinov3_vits16_f32_518_v1"

TEST_EVENTS = ["2026-3-14 茶博", "2026-4-19 虎跑", "2026-4-25 良渚版本"]
# 2026-08-16 用户裁定"按比例重新划分"：23 事件下 val 补至 2 场（原单事件太薄不能做系综选择——已证）；
# 选西湖北外滩 = 新批最大、非六月口径、≥3★ 最瘦（划出 train 代价最小）。test 不动（金标准考卷完整性）。
VAL_EVENTS = ["2026-1-10 祥睦桥", "2025-4-28 西湖北外滩"]

WINDOW = 20
TAU_CV = 0.078                # 保守主口径（决策 6）
W_GLOBAL = 0.5                # 全局段中权（草案值）
W_DERIVED = 0.2               # 派生对低权（决策 8）
DERIVED_K = 8                 # 每照片派生 partner 上限
SEED = 0


def sim_weight(cos: np.ndarray) -> np.ndarray:
    """相似度→权重映射（audit §9 冻结锚点的分段线性实现；cos<0.73 → 0 排除）。"""
    xs = np.array([0.73, 0.786, 0.918, 0.973])
    ys = np.array([0.3, 0.5, 0.8, 1.0])
    w = np.interp(cos, xs, ys)
    return np.where(cos < 0.73, 0.0, w)


def dstar_factor(d: np.ndarray) -> np.ndarray:
    """大星级差降权（决策 7 草案）：Δ=3→0.5，Δ≥4→0.25，其余 1。"""
    return np.where(d == 3, 0.5, np.where(d >= 4, 0.25, 1.0))


# ---------------------------------------------------------------------------

def load_all(args):
    clus = {r["fingerprint"]: dict(seg=int(r["seg_id"]), cluster=int(r["cluster_id"]),
                                   csize=int(r["cluster_size"]),
                                   is_top=int(r["is_cluster_top"]), event=r["event"])
            for r in csv.DictReader(open(args.clusters, encoding="utf-8-sig"))}
    lat = {r["fingerprint"]: (float(r["b_seg"]), float(r["score"]))
           for r in csv.DictReader(open(args.latent_clean, encoding="utf-8-sig"))}
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        meta = {fp: dict(time=ct, rating=int(r), cv=_cv_aggregate(cv_blob),
                         ret=int(rt or 0))
                for fp, ct, r, cv_blob, rt in conn.execute(
                    "SELECT fingerprint, capture_time, rating, cv_grid, "
                    "COALESCE(is_retouched,0) FROM photos WHERE rating IS NOT NULL")}
        feats = {fp: np.frombuffer(blob, dtype="<f4").astype(np.float32)
                 for fp, blob in conn.execute(
                     "SELECT fingerprint, cls_vector FROM photo_features WHERE model_id=?",
                     (MODEL_ORIG,))}
    finally:
        conn.close()
    from datetime import datetime
    photos = []
    for fp, m in meta.items():
        if fp not in clus or fp not in feats or fp not in lat:
            continue
        try:
            t = datetime.fromisoformat(str(m["time"]))
        except (TypeError, ValueError):
            continue
        photos.append(dict(fp=fp, t=t, **m, **clus[fp], b=lat[fp][0], s=lat[fp][1]))
    for p in photos:
        p["split"] = ("test" if p["event"] in TEST_EVENTS
                      else "val" if p["event"] in VAL_EVENTS else "train")
        p["feat"] = l2_normalize(feats[p["fp"]])
    return photos


def load_anchor_gap_threshold(photos, args) -> float:
    """派生对显著阈值 = 训练事件锚点对中 |Δnew|≥2 对的 |Δs| 中位数。"""
    s_of = {p["fp"]: p["s"] for p in photos}
    split_of = {p["fp"]: p["split"] for p in photos}
    pairs = []
    for key_p, tsv_p in ((args.pool_key, args.pool_tsv), (args.abs_key, args.abs_tsv)):
        key = {r["new_name"]: r["fingerprint"]
               for r in csv.DictReader(open(key_p, encoding="utf-8-sig"))}
        rated = [(k, int(v)) for k, v in
                 (line.rstrip("\n").split("\t") for line in open(tsv_p, encoding="utf-8"))]
        fps = [(key[n], r) for n, r in rated if n in key and key[n] in s_of]
        for a in range(len(fps)):
            for b in range(a + 1, len(fps)):
                (fa, ra), (fb, rb) = fps[a], fps[b]
                if abs(ra - rb) >= 2 and split_of[fa] == "train" and split_of[fb] == "train":
                    pairs.append(abs(s_of[fa] - s_of[fb]))
    thr = float(np.median(pairs)) if pairs else 1.0
    print(f"派生对显著阈值 |Δs| > {thr:.3f}（训练事件锚点 |Δnew|≥2 对的 |Δs| 中位，n={len(pairs)}）")
    return thr


# ---------------------------------------------------------------------------

def gen_pairs(photos, thr):
    """生成三类对。返回 list[dict]（含 split/ptype/weight/诊断列）。"""
    idx_of = {p["fp"]: i for i, p in enumerate(photos)}
    F = np.stack([p["feat"] for p in photos])
    rating = np.array([p["rating"] for p in photos])
    seg = np.array([p["seg"] for p in photos])
    sharp = np.array([p["cv"][0] for p in photos])
    shake = np.array([p["cv"][1] for p in photos])
    contrast = np.array([p["cv"][2] for p in photos])
    stds = [np.nanstd(sharp) or 1.0, np.nanstd(shake) or 1.0, np.nanstd(contrast) or 1.0]
    pairs: list[dict] = []

    def emit(i, j, ptype, w):
        cos = float(F[i] @ F[j])
        pairs.append(dict(i=i, j=j, ptype=ptype, weight=w, cos=cos,
                          cv=float("nan"), dstar=int(abs(rating[i] - rating[j])),
                          s_gap=float("nan"), split=photos[i]["split"]))

    # ---- A. window（段内滑窗，全体照片） ----
    by_seg: dict[int, list[int]] = defaultdict(list)
    for i, p in enumerate(photos):
        by_seg[p["seg"]].append(i)
    win_ii, win_jj = [], []
    for s, ixs in by_seg.items():
        ixs.sort(key=lambda i: photos[i]["t"])
        for a in range(len(ixs)):
            for b in range(a + 1, min(a + WINDOW + 1, len(ixs))):
                i, j = ixs[a], ixs[b]
                if rating[i] != rating[j]:
                    win_ii.append(i)
                    win_jj.append(j)
    if win_ii:
        ii = np.array(win_ii)
        jj = np.array(win_jj)
        cos = (F[ii] * F[jj]).sum(axis=1)
        cv, _ = pairwise_cv_distance(sharp, shake, contrast, *stds, ii, jj)
        w = sim_weight(cos) * dstar_factor(np.abs(rating[ii] - rating[jj]))
        tie = (cos >= 0.98) & (cv < TAU_CV)
        keep = (w > 0) & ~tie
        for k in np.where(keep)[0]:
            i, j = int(ii[k]), int(jj[k])
            if photos[i]["split"] != photos[j]["split"]:
                continue                      # 理论上同段必同 split，防御
            pairs.append(dict(i=i, j=j, ptype="window", weight=float(w[k]),
                              cos=float(cos[k]), cv=float(cv[k]),
                              dstar=int(abs(rating[i] - rating[j])), s_gap=float("nan"),
                              split=photos[i]["split"]))

    # ---- B. global（≥3★ 事件内全局，仅清洁） ----
    by_ev: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(photos):
        if p["is_top"] and p["rating"] >= 3:
            by_ev[p["event"]].append(i)
    for ev, ixs in by_ev.items():
        for a in range(len(ixs)):
            for b in range(a + 1, len(ixs)):
                i, j = ixs[a], ixs[b]
                if rating[i] == rating[j]:
                    continue
                emit(i, j, "global", 0.0)
    g_ix = [k for k, pr in enumerate(pairs) if pr["ptype"] == "global"]
    if g_ix:
        gi = np.array([pairs[k]["i"] for k in g_ix])
        gj = np.array([pairs[k]["j"] for k in g_ix])
        cos = (F[gi] * F[gj]).sum(axis=1)
        cv, _ = pairwise_cv_distance(sharp, shake, contrast, *stds, gi, gj)
        tie = (cos >= 0.98) & (cv < TAU_CV)
        for n, k in enumerate(g_ix):
            if tie[n]:
                pairs[k]["ptype"] = "global_tie_drop"
                continue
            pairs[k]["weight"] = W_GLOBAL * float(dstar_factor(np.array([pairs[k]["dstar"]]))[0])
            pairs[k]["cos"] = float(cos[n])
            pairs[k]["cv"] = float(cv[n])
        pairs = [pr for pr in pairs if pr["ptype"] != "global_tie_drop"]

    # ---- C. derived（潜分派生，仅清洁、同 split、跨 cluster） ----
    rng = np.random.default_rng(SEED)
    by_split: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(photos):
        if p["is_top"]:
            by_split[p["split"]].append(i)
    seen = set()
    for sp, ixs in by_split.items():
        s = np.array([photos[i]["s"] for i in ixs])
        for a, i in enumerate(ixs):
            cand = [ixs[b] for b in range(len(ixs))
                    if b != a and abs(s[b] - s[a]) > thr
                    and photos[ixs[b]]["cluster"] != photos[i]["cluster"]]
            if not cand:
                continue
            take = rng.choice(cand, size=min(DERIVED_K, len(cand)), replace=False)
            for j in take:
                key = (min(i, int(j)), max(i, int(j)))
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(dict(i=key[0], j=key[1], ptype="derived", weight=W_DERIVED,
                                  cos=float(F[key[0]] @ F[key[1]]), cv=float("nan"),
                                  dstar=int(abs(rating[key[0]] - rating[key[1]])),
                                  s_gap=float(abs(photos[key[0]]["s"] - photos[key[1]]["s"])),
                                  split=sp))
    return pairs, idx_of


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="M3 训练对生成（plan-3-2 §6.2）")
    ap.add_argument("--db", default=DB_DEFAULT)
    ap.add_argument("--clusters", default=CLUSTERS_DEFAULT)
    ap.add_argument("--latent-clean", default=LATENT_CLEAN_DEFAULT)
    ap.add_argument("--pool-key", default=POOL_KEY_DEFAULT)
    ap.add_argument("--pool-tsv", default=POOL_TSV_DEFAULT)
    ap.add_argument("--abs-key", default=ABS_KEY_DEFAULT)
    ap.add_argument("--abs-tsv", default=ABS_TSV_DEFAULT)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "out" / "m3_pairs"))
    args = ap.parse_args()

    photos = load_all(args)
    print(f"照片 {len(photos)} | train/val/test = "
          f"{sum(p['split']=='train' for p in photos)}/{sum(p['split']=='val' for p in photos)}"
          f"/{sum(p['split']=='test' for p in photos)}")
    thr = load_anchor_gap_threshold(photos, args)
    pairs, idx_of = gen_pairs(photos, thr)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "photos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fingerprint", "event", "seg_id", "cluster_id", "cluster_size",
                    "is_cluster_top", "rating_raw", "b_seg", "abs_score", "is_retouched", "split"])
        for p in photos:
            w.writerow([p["fp"], p["event"], p["seg"], p["cluster"], p["csize"], p["is_top"],
                        p["rating"], f"{p['b']:.4f}", f"{p['s']:.4f}", p["ret"], p["split"]])
    for sp in ("train", "val", "test"):
        with open(out / f"pairs_{sp}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["fp_i", "fp_j", "ptype", "weight", "cos", "cv_dist", "dstar", "s_gap"])
            for pr in pairs:
                if pr["split"] == sp:
                    w.writerow([photos[pr["i"]]["fp"], photos[pr["j"]]["fp"], pr["ptype"],
                                f"{pr['weight']:.4f}", f"{pr['cos']:.4f}", f"{pr['cv']:.4f}",
                                pr["dstar"], f"{pr['s_gap']:.4f}"])

    # ---------------- M3 GATE 自检 ----------------
    n_cross = sum(1 for pr in pairs if photos[pr["i"]]["split"] != photos[pr["j"]]["split"])
    win = [pr for pr in pairs if pr["ptype"] == "window"]
    n_win_seg = sum(1 for pr in win if photos[pr["i"]]["seg"] != photos[pr["j"]]["seg"])
    n_tie_left = sum(1 for pr in pairs if pr["ptype"] in ("window", "global")
                     and pr["cos"] >= 0.98 and pr["cv"] < TAU_CV)
    der = [pr for pr in pairs if pr["ptype"] == "derived"]
    n_der_dirty = sum(1 for pr in der
                      if not (photos[pr["i"]]["is_top"] and photos[pr["j"]]["is_top"]))
    n_der_sameclu = sum(1 for pr in der
                        if photos[pr["i"]]["cluster"] == photos[pr["j"]]["cluster"])
    have_pair = set()
    for pr in pairs:
        have_pair.add(pr["i"])
        have_pair.add(pr["j"])
    orphans = [p for i, p in enumerate(photos) if i not in have_pair]
    orb_by_rating = defaultdict(int)
    for p in orphans:
        orb_by_rating[p["rating"]] += 1

    cnt = defaultdict(lambda: defaultdict(int))
    wsum = defaultdict(lambda: defaultdict(float))
    for pr in pairs:
        cnt[pr["split"]][pr["ptype"]] += 1
        wsum[pr["split"]][pr["ptype"]] += pr["weight"]

    L = ["# M3 训练对生成报告（plan-3-2 §6.2）\n"]
    L.append(f"- split：test={TEST_EVENTS} · val={VAL_EVENTS} · train=其余；照片 {len(photos)}")
    L.append(f"- 派生阈值 |Δs|>{thr:.3f}；τ_cv={TAU_CV}；window={WINDOW}；w_global={W_GLOBAL}；w_derived={W_DERIVED}\n")
    L.append("## 配对计数（按 split × 类型）\n| split | window | global | derived | 合计 |\n|---|---|---|---|---|")
    for sp in ("train", "val", "test"):
        c = cnt[sp]
        L.append(f"| {sp} | {c['window']} | {c['global']} | {c['derived']} | {sum(c.values())} |")
    L.append("\n## M3 GATE 自检\n")
    L.append(f"- 跨 split 对：**{n_cross}**（应 0）")
    L.append(f"- window 对跨段：**{n_win_seg}**（应 0）")
    L.append(f"- tie 残留（cos≥0.98 且 cv<{TAU_CV}）：**{n_tie_left}**（应 0）")
    L.append(f"- derived 非清洁 / 同团：**{n_der_dirty} / {n_der_sameclu}**（应 0/0）")
    L.append(f"- 孤儿照片（无任何对）：**{len(orphans)}**（{dict(sorted(orb_by_rating.items()))}）")
    L.append(f"- 金标准集约束：待攒集，**必须落在 test 事件**（{TEST_EVENTS}）——决策 11")
    gate_ok = (n_cross == 0 and n_win_seg == 0 and n_tie_left == 0
               and n_der_dirty == 0 and n_der_sameclu == 0)
    L.append(f"\n**M3 GATE：{'PASS' if gate_ok else 'FAIL（见上）'}**")
    (out / "m3_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "split.json").write_text(json.dumps(
        {"test": TEST_EVENTS, "val": VAL_EVENTS,
         "train": sorted({p["event"] for p in photos if p["split"] == "train"})},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] {out}/m3_report.md + photos.csv + pairs_*.csv")
    for sp in ("train", "val", "test"):
        c = cnt[sp]
        print(f"  {sp}: window {c['window']} · global {c['global']} · derived {c['derived']}")
    print(f"GATE 自检: 跨split {n_cross} · 跨段window {n_win_seg} · tie残留 {n_tie_left} · "
          f"derived违规 {n_der_dirty + n_der_sameclu} · 孤儿 {len(orphans)} → {'PASS' if gate_ok else 'FAIL'}")
    return 0 if gate_ok else 1


if __name__ == "__main__":
    sys.exit(main())
