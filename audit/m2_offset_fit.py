"""
m2_offset_fit.py — M2 排序制校准拟合（plan-3-2 §6 v1.0 决策 4）+ GATE 三条

宪法 §0.3 v1.7：星级只有排序、无数值意义。本脚本只做**排序进、排序出**：
    潜分 s(x) = g(段内局部星级) + b_seg。g 单调（参数化为正增量累加，保段内老序不破），
    b_seg 逐段一个；拟合 = 加权逐对 logistic 损失（Bradley-Terry 风格）+ 事件内收缩先验。
    输出的潜分只用于排序，数值本身无意义。

约束集（§6 v1.0）：
    ① 锚点跨段/跨事件排序对（m2_pool 183 代表 高权 w=1；isolated_lo 低权 w=0.3；
       abs_set 141 团顶双职 高权 w=1）——评级只取等级差方向，同级=并列不出约束；
    ② 大段封顶约束（中权 w=0.5）：同事件内 ≥32 张且出过 4★+ 的段顶 > ≥32 张且止步 ≤3★ 的段顶；
    ③ 精修顶端约束（弱权 w=0.2）：被精修者 > 同事件 0-2★ 未精修者（每精修抽 ≤5 个对照）。
    暗放重复件不进拟合，只做 GATE 1。

GATE（全排序口径）：
    1. 重复件对级一致率（dup 两次评级对第三方方向一致比例）≥90%（首跑草案阈值）；
    2. 留出锚点回测：80/20 分层留出重拟合，留出锚点的预测序 vs 实评序对级一致率
       （另报 abs_set 59 张非团顶纯外验集——全程未参与拟合）；
    3. 大段封顶约束满足率（sanity）。

输出（--out 目录）：
    m2_offset_report.md  报告（g 曲线 / b_seg 表 / GATE 结果 / 事件偏移地图 / 残差诊断）
    latent_scores.csv    全库 9418 行：fingerprint,event,seg_id,cluster_id,rating,g,b_seg,score

用法（仓根 D:/Git/PhotoViewer 下）：
    PYTHONUTF8=1 Tools/.venv/Scripts/python.exe Training/audit/m2_offset_fit.py
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

DB_DEFAULT = "D:/PhotoDB/dataset/photos_dataset.db"
CLUSTERS_DEFAULT = "D:/Git/PhotoViewer/Training/audit/out/clusters/clusters.csv"
POOL_KEY_DEFAULT = "D:/PhotoDB/dataset/m2_pool_key.csv"
POOL_TSV_DEFAULT = "D:/PhotoDB/dataset/m2_pool_ratings.tsv"
ABS_KEY_DEFAULT = "D:/PhotoDB/dataset/abs_set_key.csv"
ABS_TSV_DEFAULT = "D:/PhotoDB/dataset/abs_set_ratings.tsv"

W_NORMAL, W_LOW, W_CAP, W_RET = 1.0, 0.3, 0.5, 0.2
W_D1_DEFAULT = 0.5              # Δ=1 锚点对权重因子（方向常是整数舍入噪声，降权不学噪声）
LAM_B = 0.02                    # 事件内收缩先验强度（每段每张照片计）
BIG_SEG = 32                    # 大段阈值（§6）
SEED = 0


# ---------------------------------------------------------------------------
# 数据装载
# ---------------------------------------------------------------------------

def load_library(clusters_path: str, db_path: str):
    rows = list(csv.DictReader(open(clusters_path, encoding="utf-8-sig")))
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        ret = {fp: int(v or 0) for fp, v in conn.execute(
            "SELECT fingerprint, COALESCE(is_retouched,0) FROM photos")}
    finally:
        conn.close()
    lib = []
    for r in rows:
        fp = r["fingerprint"]
        lib.append(dict(fp=fp, event=r["event"], seg=int(r["seg_id"]),
                        cluster=int(r["cluster_id"]), csize=int(r["cluster_size"]),
                        rating=int(r["rating"]), centrality=float(r["centrality"]),
                        ret=ret.get(fp, 0)))
    return lib


def load_anchor_ratings(key_path: str, tsv_path: str, skip_dups: bool) -> list[dict]:
    key = {r["new_name"]: r for r in csv.DictReader(open(key_path, encoding="utf-8-sig"))}
    out = []
    for line in open(tsv_path, encoding="utf-8"):
        name, rating = line.rstrip("\n").split("\t")
        k = key[name]
        if skip_dups and k.get("is_dup_of"):
            continue                    # 重复件不进拟合（GATE 1 专用）
        out.append(dict(fp=k["fingerprint"], new=int(rating),
                        weight=W_LOW if k.get("weight_class") == "low" else W_NORMAL,
                        origin="pool" if "m2_pool" in key_path else "abs"))
    return out


def load_dup_pairs(key_path: str, tsv_path: str) -> list[tuple[str, int, int]]:
    """暗放重复件：(fingerprint, 原件新评级, 重复件新评级)。"""
    key = {r["new_name"]: r for r in csv.DictReader(open(key_path, encoding="utf-8-sig"))}
    ratings = {n: int(r) for n, r in
               (line.rstrip("\n").split("\t") for line in open(tsv_path, encoding="utf-8"))}
    by_fp: dict[str, list[str]] = defaultdict(list)
    for name, k in key.items():
        by_fp[k["fingerprint"]].append(name)
    pairs = []
    for fp, names in by_fp.items():
        if len(names) == 2:
            a, b = sorted(names)
            pairs.append((fp, ratings[a], ratings[b]))
    return pairs


# ---------------------------------------------------------------------------
# 模型：s = g(rating) + b[seg]；g(r) = Σ_{k≤r} exp(a_k)，g(0)=0
# ---------------------------------------------------------------------------

class OffsetModel:
    def __init__(self, n_seg: int):
        self.n_seg = n_seg
        self.a = np.zeros(5)                  # log 增量：g(1..5)-g(0..4)
        self.b = np.zeros(n_seg)

    def pack(self):
        return np.concatenate([self.a, self.b])

    def unpack(self, th):
        self.a, self.b = th[:5], th[5:]

    def g(self, r: np.ndarray) -> np.ndarray:
        inc = np.exp(self.a)                  # 5 个正增量
        gcum = np.concatenate([[0.0], np.cumsum(inc)])
        return gcum[r]

    def score(self, rating: np.ndarray, seg: np.ndarray) -> np.ndarray:
        return self.g(rating) + self.b[seg]


def fit(lib, constraints, seg_sizes, seg_event, n_seg, seed=SEED):
    """L-BFGS 拟合。constraints = (i, j, w)：s_i > s_j、权重 w（库行号）。"""
    from scipy.optimize import minimize
    rating = np.array([p["rating"] for p in lib])
    seg = np.array([p["seg"] for p in lib])
    ci = np.array([c[0] for c in constraints])
    cj = np.array([c[1] for c in constraints])
    cw = np.array([c[2] for c in constraints])
    ev_of_seg = np.array([seg_event[s] for s in range(n_seg)])
    model = OffsetModel(n_seg)

    def loss_grad(th):
        model.unpack(th)
        inc = np.exp(model.a)
        gcum = np.concatenate([[0.0], np.cumsum(inc)])
        s = gcum[rating] + model.b[seg]
        d = s[ci] - s[cj]
        lse = np.logaddexp(0, -d)
        loss = float((cw * lse).sum())
        sig = -cw * (1.0 / (1.0 + np.exp(np.clip(d, -30, 30))))   # dloss/dd
        grad_a = np.zeros(5)
        for k in range(1, 6):
            coef = (sig * (rating[ci] >= k)).sum() - (sig * (rating[cj] >= k)).sum()
            grad_a[k - 1] = coef * inc[k - 1]
        grad_b = np.bincount(seg[ci], weights=sig, minlength=n_seg) \
            - np.bincount(seg[cj], weights=sig, minlength=n_seg)
        # 事件内收缩先验：LAM_B * n_seg * (b_seg - mean_b_event)
        for ev in np.unique(ev_of_seg):
            segs_ev = np.where(ev_of_seg == ev)[0]
            m_b = np.average(model.b[segs_ev], weights=seg_sizes[segs_ev])
            dev = model.b[segs_ev] - m_b
            loss += float(LAM_B * (seg_sizes[segs_ev] * dev ** 2).sum())
            grad_b[segs_ev] += 2 * LAM_B * seg_sizes[segs_ev] * dev
        # 全局定心（消平移不定性）
        loss += float((seg_sizes * model.b).sum() ** 2)
        grad_b += 2 * (seg_sizes * model.b).sum() * seg_sizes
        return loss, np.concatenate([grad_a, grad_b])

    res = minimize(loss_grad, model.pack(), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=500))
    model.unpack(res.x)
    return model, res


# ---------------------------------------------------------------------------
# 约束构建
# ---------------------------------------------------------------------------

def anchor_constraints(anchor_idx: list[int], anchor_new: list[int],
                       anchor_w: list[float], w_d1: float = W_D1_DEFAULT) -> list[tuple[int, int, float]]:
    """锚点两两（等级不同才出约束，方向=新评级大者在前，权重=w_i*w_j；
    Δ=1 对乘 w_d1——整数分级边界舍入是量化噪声，降权不学噪声（用户 07-19 原则））。"""
    cons = []
    n = len(anchor_idx)
    for a in range(n):
        for b in range(a + 1, n):
            if anchor_new[a] == anchor_new[b]:
                continue
            i, j = (a, b) if anchor_new[a] > anchor_new[b] else (b, a)
            w = anchor_w[i] * anchor_w[j]
            if abs(anchor_new[a] - anchor_new[b]) == 1:
                w *= w_d1
            cons.append((anchor_idx[i], anchor_idx[j], w))
    return cons


def cap_constraints(lib, n_seg) -> list[tuple[int, int, float]]:
    """大段封顶约束：同事件内（≥32张 & cap≥4）段顶 >（≥32张 & cap≤3）段顶。"""
    by_seg: dict[int, list[int]] = defaultdict(list)
    for ix, p in enumerate(lib):
        by_seg[p["seg"]].append(ix)
    tops: dict[int, int] = {}
    for s, ixs in by_seg.items():
        tops[s] = max(ixs, key=lambda ix: (lib[ix]["rating"], lib[ix]["centrality"]))
    info = {s: dict(n=len(ixs), cap=lib[tops[s]]["rating"], event=lib[tops[s]]["event"])
            for s, ixs in by_seg.items()}
    cons = []
    for a in info:
        if not (info[a]["n"] >= BIG_SEG and info[a]["cap"] >= 4):
            continue
        for b in info:
            if (info[b]["n"] >= BIG_SEG and info[b]["cap"] <= 3
                    and info[a]["event"] == info[b]["event"]):
                cons.append((tops[a], tops[b], W_CAP))
    return cons


def retouch_constraints(lib, seed=SEED) -> list[tuple[int, int, float]]:
    rng = np.random.default_rng(seed)
    low_by_ev: dict[str, list[int]] = defaultdict(list)
    for ix, p in enumerate(lib):
        if p["rating"] <= 2 and not p["ret"]:
            low_by_ev[p["event"]].append(ix)
    cons = []
    for ix, p in enumerate(lib):
        if not p["ret"]:
            continue
        pool = low_by_ev.get(p["event"], [])
        for jx in rng.choice(pool, size=min(5, len(pool)), replace=False):
            cons.append((ix, int(jx), W_RET))
    return cons


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------

def pair_agreement_by_delta(model, lib, idxs_a, idxs_b, new_a, new_b) -> dict[int, tuple[float, int]]:
    """预测序 vs 实评序对级一致率，按 |Δ新评级| 分层（跳过并列）。
    返回 {|Δ|: (一致率, 对数)}——闸门只看 Δ≥2（真实差异）；Δ=1 是舍入噪声区，仅参考。"""
    s = model.score(np.array([p["rating"] for p in lib]), np.array([p["seg"] for p in lib]))
    ok: dict[int, int] = defaultdict(int)
    tot: dict[int, int] = defaultdict(int)
    for i, ra in zip(idxs_a, new_a):
        for j, rb in zip(idxs_b, new_b):
            if ra == rb:
                continue
            d = min(abs(ra - rb), 3)          # 3 = Δ≥3 合并档
            tot[d] += 1
            ok[d] += int((s[i] - s[j]) * (ra - rb) > 0)
    return {d: (ok[d] / tot[d] if tot[d] else float("nan"), tot[d]) for d in sorted(tot)}


def pair_agreement(model, lib, idxs_a, idxs_b, new_a, new_b) -> tuple[float, int]:
    by = pair_agreement_by_delta(model, lib, idxs_a, idxs_b, new_a, new_b)
    n = sum(t for _, t in by.values())
    ok = sum(a * t for a, t in by.values() if not np.isnan(a))
    return (ok / n if n else float("nan")), n


def dup_consistency(dups) -> tuple[int, int, list[int]]:
    """GATE 1（跳档口径，用户 07-19 原则）：±1 = 整数分级粒度不存在的边界舍入，不算不一致；
    |两次评级差| ≥2（跳档）才算真不一致。返回 (跳档数, dup 数, 差值列表)。"""
    diffs = [abs(a - b) for _, a, b in dups]
    return sum(1 for d in diffs if d >= 2), len(dups), diffs


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="M2 排序制校准拟合 + GATE（plan-3-2 §6 v1.0）")
    ap.add_argument("--db", default=DB_DEFAULT)
    ap.add_argument("--clusters", default=CLUSTERS_DEFAULT)
    ap.add_argument("--pool-key", default=POOL_KEY_DEFAULT)
    ap.add_argument("--pool-tsv", default=POOL_TSV_DEFAULT)
    ap.add_argument("--abs-key", default=ABS_KEY_DEFAULT)
    ap.add_argument("--abs-tsv", default=ABS_TSV_DEFAULT)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "out" / "m2_offset"))
    ap.add_argument("--w-d1", type=float, default=W_D1_DEFAULT,
                    help="Δ=1 锚点对权重因子（默认 0.5；1.0 = 不降权对照）")
    args = ap.parse_args()

    lib = load_library(args.clusters, args.db)
    fp2ix = {p["fp"]: i for i, p in enumerate(lib)}
    n_seg = max(p["seg"] for p in lib) + 1
    seg_sizes = np.bincount([p["seg"] for p in lib], minlength=n_seg).astype(float)
    seg_event = {}
    for p in lib:
        seg_event.setdefault(p["seg"], p["event"])

    # 锚点：pool 183 + abs 双职（团顶 141）；abs 非团顶 59 = 纯外验
    anchors = load_anchor_ratings(args.pool_key, args.pool_tsv, skip_dups=True)
    abs_all = load_anchor_ratings(args.abs_key, args.abs_tsv, skip_dups=False)
    clus_top = {r["fingerprint"]: int(r["is_cluster_top"])
                for r in csv.DictReader(open(args.clusters, encoding="utf-8-sig"))}
    abs_top = [a for a in abs_all if clus_top.get(a["fp"], 0) == 1]
    abs_val = [a for a in abs_all if clus_top.get(a["fp"], 0) == 0]
    for a in abs_top:
        a["origin"] = "abs_top"
    print(f"锚点: pool {len(anchors)} + abs双职 {len(abs_top)} | 纯外验 abs非团顶 {len(abs_val)}")

    fit_anchors = anchors + abs_top
    a_idx = [fp2ix[a["fp"]] for a in fit_anchors]
    a_new = [a["new"] for a in fit_anchors]
    a_w = [a["weight"] for a in fit_anchors]

    cons_a = anchor_constraints(a_idx, a_new, a_w, w_d1=args.w_d1)
    cons_cap = cap_constraints(lib, n_seg)
    cons_ret = retouch_constraints(lib)
    print(f"约束: 锚点对 {len(cons_a)}（Δ=1 权重×{args.w_d1}）· 大段封顶 {len(cons_cap)} · 精修 {len(cons_ret)}")

    # ---------------- 全量拟合 ----------------
    model, res = fit(lib, cons_a + cons_cap + cons_ret, seg_sizes, seg_event, n_seg)
    print(f"拟合收敛: {res.message} (loss={res.fun:.1f})")

    # GATE 1：重复件跳档检查（±1 = 粒度不存在的边界舍入，不算不一致）
    dups = load_dup_pairs(args.pool_key, args.pool_tsv)
    g1_bad, g1_n, dup_diffs = dup_consistency(dups)
    dist = {int(k): int(v) for k, v in zip(*np.unique(dup_diffs, return_counts=True))}
    print(f"\nGATE 1 重复件跳档检查: 跳档(≥2级) {g1_bad}/{g1_n} "
          f"{'PASS' if g1_bad == 0 else 'FAIL'}；两次评级差分布 {dist}")

    # GATE 2：80/20 分层留出重拟合（按 |Δ| 分层报；闸门只卡 Δ≥2——真实差异处必须准，
    # Δ=1 是整数舍入噪声区，只作参考不闸门，用户 07-19 原则）
    rng = np.random.default_rng(SEED)
    hold = np.zeros(len(fit_anchors), bool)
    by_ev: dict[str, list[int]] = defaultdict(list)
    for k, a in enumerate(fit_anchors):
        by_ev[lib[fp2ix[a["fp"]]]["event"]].append(k)
    for ev, ks in by_ev.items():
        ks = np.array(ks)
        hold[rng.choice(ks, size=max(1, round(len(ks) * 0.2)), replace=False)] = True
    tr = ~hold
    cons_tr = anchor_constraints([a_idx[k] for k in np.where(tr)[0]],
                                 [a_new[k] for k in np.where(tr)[0]],
                                 [a_w[k] for k in np.where(tr)[0]],
                                 w_d1=args.w_d1)
    model_ho, _ = fit(lib, cons_tr + cons_cap + cons_ret, seg_sizes, seg_event, n_seg)
    hold_idx = [a_idx[k] for k in np.where(hold)[0]]
    hold_new = [a_new[k] for k in np.where(hold)[0]]
    tr_idx = [a_idx[k] for k in np.where(tr)[0]]
    tr_new = [a_new[k] for k in np.where(tr)[0]]
    val_idx = [fp2ix[a["fp"]] for a in abs_val]
    val_new = [a["new"] for a in abs_val]
    g2_sets = {
        "留出×留出": pair_agreement_by_delta(model_ho, lib, hold_idx, hold_idx, hold_new, hold_new),
        "留出×训练": pair_agreement_by_delta(model_ho, lib, hold_idx, tr_idx, hold_new, tr_new),
        "外验59×训练": pair_agreement_by_delta(model_ho, lib, val_idx, tr_idx, val_new, tr_new),
        "[对照]训练内": pair_agreement_by_delta(model_ho, lib, tr_idx, tr_idx, tr_new, tr_new),
    }

    def _ge2(by: dict) -> tuple[float, int]:
        ok = sum(a * t for d, (a, t) in by.items() if d >= 2 and not np.isnan(a))
        n = sum(t for d, (_, t) in by.items() if d >= 2)
        return (ok / n if n else float("nan"), n)

    def _fmt(by: dict) -> str:
        parts = [f"Δ1 {by[1][0]:.3f}(n={by[1][1]})" if 1 in by else "Δ1 —"]
        for d in (2, 3):
            if d in by:
                parts.append(f"Δ{'≥3' if d == 3 else '2'} {by[d][0]:.3f}(n={by[d][1]})")
        return " · ".join(parts)

    for lab, by in g2_sets.items():
        ge2, n2 = _ge2(by)
        print(f"GATE 2 {lab:<12}: {_fmt(by)}  ⇒  Δ≥2 合计 {ge2:.3f} (n={n2})")

    # GATE 3：大段封顶满足率
    s_all = model.score(np.array([p["rating"] for p in lib]), np.array([p["seg"] for p in lib]))
    g3 = float(np.mean([s_all[i] > s_all[j] for i, j, _ in cons_cap])) if cons_cap else float("nan")
    print(f"GATE 3 大段封顶约束满足率 = {g3:.3f} (n={len(cons_cap)})")

    # ---------------- 报告与产出 ----------------
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inc = np.exp(model.a)
    gcum = np.concatenate([[0.0], np.cumsum(inc)])
    ev_b: dict[str, float] = {}
    for ev in sorted(set(seg_event.values())):
        segs_ev = [s for s in range(n_seg) if seg_event[s] == ev]
        ev_b[ev] = float(np.average(model.b[segs_ev], weights=seg_sizes[segs_ev]))

    with open(out / "latent_scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fingerprint", "event", "seg_id", "cluster_id", "rating", "g", "b_seg", "score"])
        for i, p in enumerate(lib):
            w.writerow([p["fp"], p["event"], p["seg"], p["cluster"], p["rating"],
                        f"{gcum[p['rating']]:.4f}", f"{model.b[p['seg']]:.4f}", f"{s_all[i]:.4f}"])

    L = []
    L.append("# M2 排序制校准报告（plan-3-2 §6 v1.0 决策 4）\n")
    L.append(f"- 锚点：m2_pool {len(anchors)} + abs_set 双职 {len(abs_top)}；纯外验 {len(abs_val)}")
    L.append(f"- 约束：锚点对 {len(cons_a)}（Δ=1 权重×{args.w_d1}，不学整数舍入噪声）/ 大段封顶 {len(cons_cap)} / 精修 {len(cons_ret)}")
    L.append(f"- 收敛：{res.message}（loss={res.fun:.1f}）\n")
    L.append("## GATE（全排序口径，chance=50%；闸门只卡真实差异处——Δ≥2 对与跳档）\n")
    L.append(f"1. **重复件跳档检查 = {g1_bad}/{g1_n} 跳档（≥2级）** → **{'PASS' if g1_bad == 0 else 'FAIL'}**；"
             f"两次评级差分布 {dist}（±1 = 整数分级粒度不存在的边界舍入，不算不一致）")
    L.append("2. **留出锚点回测（按 |Δ| 分层，闸门看 Δ≥2）**：")
    for lab, by in g2_sets.items():
        ge2, n2 = _ge2(by)
        L.append(f"   - {lab}：{_fmt(by)} ⇒ **Δ≥2 合计 {ge2:.3f}**（n={n2}）")
    L.append(f"3. **大段封顶约束满足率 = {g3:.3f}**（n={len(cons_cap)}；锦标赛路径依赖噪声信号，仅参考）\n")
    L.append("## g 曲线（段内局部星级的单调映射，g(0)=0）\n")
    L.append("| 0★ | 1★ | 2★ | 3★ | 4★ | 5★ |\n|---|---|---|---|---|---|")
    L.append("| " + " | ".join(f"{v:.3f}" for v in gcum) + " |\n")
    L.append("## 事件偏移地图（b_seg 的照片数加权均值，降序）\n")
    L.append("| 事件 | 段数 | 加权 b 均值 |")
    L.append("|---|---|---|")
    for ev, v in sorted(ev_b.items(), key=lambda t: -t[1]):
        n_ev = sum(1 for s in range(n_seg) if seg_event[s] == ev)
        L.append(f"| {ev} | {n_ev} | {v:+.3f} |")
    (out / "m2_offset_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n[OK] 写出 {out / 'm2_offset_report.md'} / latent_scores.csv（{len(lib)} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
