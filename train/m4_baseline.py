"""
m4_baseline.py — M4 基线训练（plan-3-3 §6.1 轻量起点 v0）+ 段内验收

v0 特征（patch 主体权重留 v1，先跑通管线拿参照）：
    CLS(S-enh, 384) + CV 网格统计（7 标量 × [nanmean/nanstd/p10/worst] = 28 + 7 NaN-mask）
    + EXIF（log 等效焦距 / log 光圈 / log 快门 / crop_factor + 4 mask）≈ 427 维 → 小 MLP → score。

训练：pairs_train（60180：window 高 / global 中 / derived 低）加权逐对 logistic（软偏好，
决策 4）；Adam + val 早停（val 对级一致率）。val/test 仅作评估。

评估（val/test，全部排序口径 + 容差三层 v1.8）：
    1. 对级一致率（按 ptype 拆，附 |Δ| 分层——Δ=1 是舍入噪声区只参考，闸门看 Δ≥2）；
    2. 段内 top-1 命中（模型段内第一名是否真最高星；对照 chance 与 M2 潜分基线）；
    3. 段内 Spearman（≥5 张且有星方差的段）；
    4. ≥3★ 召回 @ Top 12.5%（事件级 + 段配额保底版，决策 10）；
    5. 0-5 容差三层（score→0-5 阈值仅作度量映射、train 分位拟合）：exact / ±1 / ≥2，
       ≥2 区分团顶清洁/非团顶污染样本（v1.8）；
    6. 绝对涌现代理（金标准未到）：test 三事件 abs_set 盲评照片（40 张）模型分 vs 用户序；
       事件均分排序 vs 用户 abs 序（茶博<虎跑≈良渚）。

输出（--out）：model.pt · m4_report.md · scores.csv（全库模型分）

用法（仓根 D:/Git/PhotoViewer 下）：
    PYTHONUTF8=1 Tools/.venv/Scripts/python.exe Training/train/m4_baseline.py
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
from feature_probe import CV_SCALAR_COUNT, CV_PLANE_LEN, l2_normalize  # noqa: E402

DB_DEFAULT = "D:/PhotoDB/dataset/photos_dataset.db"
M3_DIR_DEFAULT = "D:/Git/PhotoViewer/Training/audit/out/m3_pairs"
ABS_KEY_DEFAULT = "D:/PhotoDB/dataset/abs_set_key.csv"
ABS_TSV_DEFAULT = "D:/PhotoDB/dataset/abs_set_ratings.tsv"
MODEL_ENH = "dinov3_vits16_f32_518_v1+clhe2.0ycc1.0"
FEAT_SLICES = {"cls": (0, 384), "cv": (384, 419), "exif": (419, 427), "all": (0, 427)}

HIDDEN = (128, 64)
EPOCHS = 60
PATIENCE = 8
BATCH = 2048
LR = 1e-3
WD = 1e-4
SEED = 0


# ---------------------------------------------------------------------------

def load_photos(m3_dir: str, db: str, model_id: str):
    photos = {r["fingerprint"]: dict(
        event=r["event"], seg=int(r["seg_id"]), cluster=int(r["cluster_id"]),
        csize=int(r["cluster_size"]), is_top=int(r["is_cluster_top"]),
        rating=int(r["rating_raw"]), b=float(r["b_seg"]), s=float(r["abs_score"]),
        ret=int(r["is_retouched"]), split=r["split"])
        for r in csv.DictReader(open(f"{m3_dir}/photos.csv", encoding="utf-8-sig"))}
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        exif = {fp: (fl or 0.0, ap or 0.0, ss or 0.0, cf or 0.0)
                for fp, fl, ap, ss, cf in conn.execute(
                    "SELECT fingerprint, focal_length, aperture, shutter_speed, crop_factor FROM photos")}
        cls = {fp: np.frombuffer(blob, dtype="<f4").astype(np.float32)
               for fp, blob in conn.execute(
                   "SELECT fingerprint, cls_vector FROM photo_features WHERE model_id=?",
                   (model_id,))}
        cv = {}
        for fp, blob in conn.execute("SELECT fingerprint, cv_grid FROM photos"):
            if blob is None:
                cv[fp] = None
                continue
            arr = np.frombuffer(blob, dtype="<f4")
            if arr.size != CV_SCALAR_COUNT * CV_PLANE_LEN:
                cv[fp] = None
                continue
            grid = arr.reshape(CV_SCALAR_COUNT, CV_PLANE_LEN).astype(np.float64)
            with np.errstate(all="ignore"):
                stats = np.concatenate([np.nanmean(grid, axis=1), np.nanstd(grid, axis=1),
                                        np.nanpercentile(grid, 10, axis=1), np.nanmin(grid, axis=1)])
            cv[fp] = stats.astype(np.float32)     # 28 维，可能含 NaN
    finally:
        conn.close()
    fps = [fp for fp in photos if fp in cls]
    X, metas = [], []
    for fp in fps:
        p = photos[fp]
        cvs = cv.get(fp)
        cv_feat = np.zeros(35, np.float32) if cvs is None else np.concatenate(
            [np.nan_to_num(cvs, nan=0.0), np.isnan(cvs).astype(np.float32)[:7]])
        fl, ap, ss, cf = exif.get(fp, (0, 0, 0, 0))
        ex = np.array([np.log(max(fl * (cf or 1.0), 1e-3)), np.log(max(ap, 1e-3)),
                       np.log(max(ss, 1e-6)), cf], np.float32)
        ex_mask = (np.array([fl, ap, ss, cf]) == 0).astype(np.float32)
        X.append(np.concatenate([l2_normalize(cls[fp]), cv_feat, ex, ex_mask]))
        metas.append(p)
    X = np.stack(X)
    return fps, X, metas


def load_pairs(m3_dir: str, split: str, photos):
    out = []
    for r in csv.DictReader(open(f"{m3_dir}/pairs_{split}.csv", encoding="utf-8-sig")):
        pi, pj = photos.get(r["fp_i"]), photos.get(r["fp_j"])
        if pi is None or pj is None:
            continue
        if r["ptype"] == "derived":
            y = np.sign(pi["s"] - pj["s"])
        else:
            y = np.sign(pi["rating"] - pj["rating"])
        if y == 0:
            continue
        out.append((r["fp_i"], r["fp_j"], int(y), float(r["weight"]), r["ptype"],
                    abs(pi["rating"] - pj["rating"])))
    return out


# ---------------------------------------------------------------------------

def train_model(Xtr, pairs_tr, fp2ix, Xva, pairs_va):
    import torch
    torch.manual_seed(SEED)
    D = Xtr.shape[1]
    net = torch.nn.Sequential(
        torch.nn.Linear(D, HIDDEN[0]), torch.nn.ReLU(), torch.nn.Dropout(0.1),
        torch.nn.Linear(HIDDEN[0], HIDDEN[1]), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN[1], 1))
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    ii = torch.tensor([fp2ix[p[0]] for p in pairs_tr])
    jj = torch.tensor([fp2ix[p[1]] for p in pairs_tr])
    yy = torch.tensor([p[2] for p in pairs_tr], dtype=torch.float32)
    ww = torch.tensor([p[3] for p in pairs_tr], dtype=torch.float32)
    Xt = torch.tensor(Xtr)
    vi = torch.tensor([fp2ix[p[0]] for p in pairs_va])
    vj = torch.tensor([fp2ix[p[1]] for p in pairs_va])
    vy = np.array([p[2] for p in pairs_va])
    Xv = torch.tensor(Xva)

    def va_acc():
        net.eval()
        with torch.no_grad():
            s = net(Xv).squeeze(-1).numpy()
        return float((np.sign(s[vi.numpy()] - s[vj.numpy()]) == vy).mean())

    best, best_state, bad = -1.0, None, 0
    n = len(ii)
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(n)
        for k in range(0, n, BATCH):
            b = perm[k:k + BATCH]
            si = net(Xt[ii[b]]).squeeze(-1)
            sj = net(Xt[jj[b]]).squeeze(-1)
            loss = (torch.nn.functional.softplus(-(si - sj) * yy[b]) * ww[b]).sum() / ww[b].sum()
            opt.zero_grad(); loss.backward(); opt.step()
        acc = va_acc()
        if acc > best:
            best, best_state, bad = acc, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(best_state)
    print(f"训练 {ep + 1} epoch 早停 | val 对级一致率 {best:.3f}")
    return net


# ---------------------------------------------------------------------------

def evaluate(net, X, fps, metas, pairs_by_split, abs_probe):
    import torch
    net.eval()
    with torch.no_grad():
        scores = net(torch.tensor(X)).squeeze(-1).numpy()
    s_of = dict(zip(fps, scores))
    rating_of = dict(zip(fps, (m["rating"] for m in metas)))
    R = {}

    def seg_groups(split):
        by_seg = defaultdict(list)
        for fp, m in zip(fps, metas):
            if m["split"] == split:
                by_seg[(m["event"], m["seg"])].append(fp)
        return by_seg

    for split, pairs in pairs_by_split.items():
        sub = {}
        by_type = defaultdict(lambda: [0, 0])
        by_delta = defaultdict(lambda: [0, 0])
        for fi, fj, y, w, pt, dstar in pairs:
            ok = int(np.sign(s_of[fi] - s_of[fj]) == y)
            by_type[pt][0] += ok
            by_type[pt][1] += 1
            d = min(dstar, 3)
            by_delta[d][0] += ok
            by_delta[d][1] += 1
        sub["pairs"] = {k: (v[0] / v[1], v[1]) for k, v in by_type.items()}
        sub["delta"] = {k: (v[0] / v[1], v[1]) for k, v in sorted(by_delta.items())}

        hits = ch = n_top = 0
        rhos = []
        for _, group in seg_groups(split).items():
            if len(group) < 2:
                continue
            rs = np.array([rating_of[fp] for fp in group])
            if rs.max() == rs.min():
                continue
            sc = np.array([s_of[fp] for fp in group])
            top_pred = group[int(sc.argmax())]
            if rating_of[top_pred] == rs.max():
                hits += 1
            ch += (rs == rs.max()).mean()
            n_top += 1
            if len(group) >= 5:
                from scipy.stats import spearmanr
                rho = spearmanr(rs, sc).statistic
                if not np.isnan(rho):
                    rhos.append(float(rho))
        sub["top1"] = (hits / n_top if n_top else float("nan"),
                       ch / n_top if n_top else float("nan"), n_top)
        sub["seg_rho"] = (float(np.mean(rhos)), float(np.std(rhos)), len(rhos))

        # ≥3★ 召回 @ Top12.5%（事件级 + 段配额版）
        for mode in ("event", "seg_quota"):
            got = need = 0
            if mode == "event":
                by_ev = defaultdict(list)
                for fp, m in zip(fps, metas):
                    if m["split"] == split:
                        by_ev[m["event"]].append(fp)
                for ev, group in by_ev.items():
                    k = max(1, round(len(group) * 0.125))
                    top = sorted(group, key=lambda fp: -s_of[fp])[:k]
                    got += sum(1 for fp in top if rating_of[fp] >= 3)
                    need += sum(1 for fp in group if rating_of[fp] >= 3)
            else:
                for _, group in seg_groups(split).items():
                    k = max(1, round(len(group) * 0.125))
                    top = sorted(group, key=lambda fp: -s_of[fp])[:k]
                    got += sum(1 for fp in top if rating_of[fp] >= 3)
                    need += sum(1 for fp in group if rating_of[fp] >= 3)
            sub[f"recall_{mode}"] = (got / need if need else float("nan"), got, need)

        # 0-5 容差三层（train 分位阈值映射）
        sub["star"] = star_tolerance(s_of, fps, metas, split)
        R[split] = sub

    # 绝对涌现代理：test abs_set 盲评照片
    emerg = {}
    for ev, group in abs_probe.items():
        rs = np.array([g[1] for g in group])
        sc = np.array([s_of[g[0]] for g in group])
        from scipy.stats import spearmanr
        if len(set(rs)) > 1 and len(group) >= 4:
            emerg[ev] = (float(spearmanr(rs, sc).statistic), len(group))
    return R, emerg, s_of


def star_tolerance(s_of, fps, metas, split):
    """score→0-5 阈值（train 分位拟合，仅度量用）→ exact / ±1 / ≥2（区分团顶清洁）。"""
    tr_scores = np.array([s_of[fp] for fp, m in zip(fps, metas) if m["split"] == "train"])
    tr_rates = np.array([m["rating"] for fp, m in zip(fps, metas) if m["split"] == "train"])
    qs = [np.mean(tr_rates <= k) for k in range(5)]
    th = [np.quantile(tr_scores, q) for q in qs]
    def to_star(v):
        return int(sum(v > t for t in th))
    res = {"clean": np.zeros(4), "dirty": np.zeros(4)}   # [exact, ±1, ±2, ≥3]
    for fp, m in zip(fps, metas):
        if m["split"] != split:
            continue
        err = abs(to_star(s_of[fp]) - m["rating"])
        bucket = res["clean"] if m["is_top"] else res["dirty"]
        bucket[min(err, 3)] += 1
    out = {}
    for k, v in res.items():
        n = v.sum() or 1
        out[k] = (v[0] / n, (v[0] + v[1]) / n, (v[0] + v[1] + v[2]) / n, int(n))
    return out


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="M4 基线训练 + 段内验收（plan-3-3）")
    ap.add_argument("--db", default=DB_DEFAULT)
    ap.add_argument("--m3", default=M3_DIR_DEFAULT)
    ap.add_argument("--abs-key", default=ABS_KEY_DEFAULT)
    ap.add_argument("--abs-tsv", default=ABS_TSV_DEFAULT)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "out" / "m4_baseline"))
    ap.add_argument("--model-id", default=MODEL_ENH, help="CLS 视图（梯2 对照可换 ViT-L 同后缀）")
    ap.add_argument("--feat-group", default="all", choices=list(FEAT_SLICES),
                    help="特征组消融：cls/cv/exif/all")
    args = ap.parse_args()

    fps, X, metas = load_photos(args.m3, args.db, args.model_id)
    fp2ix = {fp: i for i, fp in enumerate(fps)}
    tr_mask = np.array([m["split"] == "train" for m in metas])
    mu = X[tr_mask].mean(axis=0)
    sd = X[tr_mask].std(axis=0)
    sd[sd == 0] = 1.0
    X = (X - mu) / sd
    lo, hi = FEAT_SLICES[args.feat_group]
    X = X[:, lo:hi]
    print(f"特征 {X.shape}（train 标准化；model_id={args.model_id} · feat={args.feat_group}）")

    pairs_tr = load_pairs(args.m3, "train", {fp: metas[fp2ix[fp]] for fp in fps})
    pairs_va = load_pairs(args.m3, "val", {fp: metas[fp2ix[fp]] for fp in fps})
    pairs_te = load_pairs(args.m3, "test", {fp: metas[fp2ix[fp]] for fp in fps})
    print(f"对: train {len(pairs_tr)} · val {len(pairs_va)} · test {len(pairs_te)}")

    net = train_model(X, pairs_tr, fp2ix, X, pairs_va)

    abs_probe = defaultdict(list)
    abs_key = {r["new_name"]: r for r in csv.DictReader(open(args.abs_key, encoding="utf-8-sig"))}
    abs_rated = {n: int(r) for n, r in
                 (line.rstrip("\n").split("\t") for line in open(args.abs_tsv, encoding="utf-8"))}
    meta_of = dict(zip(fps, metas))
    for name, k in abs_key.items():
        if k["fingerprint"] in meta_of and meta_of[k["fingerprint"]]["split"] == "test":
            abs_probe[k["event_label"]].append((k["fingerprint"], abs_rated[name]))

    R, emerg, s_of = evaluate(net, X, fps, metas,
                              {"train": pairs_tr, "val": pairs_va, "test": pairs_te},
                              abs_probe)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save(net.state_dict(), out / "model.pt")
    with open(out / "scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fingerprint", "event", "seg_id", "split", "rating", "score"])
        for fp, m in zip(fps, metas):
            w.writerow([fp, m["event"], m["seg"], m["split"], m["rating"], f"{s_of[fp]:.4f}"])

    L = ["# M4 基线报告（plan-3-3，v0 轻量特征）\n"]
    for split, sub in R.items():
        L.append(f"## {split}\n")
        L.append("- 对级一致率（按类型）: " + " · ".join(
            f"{k} {a:.3f}(n={n})" for k, (a, n) in sub["pairs"].items()))
        L.append("- 对级一致率（按 |Δ星级|）: " + " · ".join(
            f"Δ{'≥3' if k == 3 else k} {a:.3f}(n={n})" for k, (a, n) in sub["delta"].items()))
        t1, ch, nt = sub["top1"]
        L.append(f"- 段内 top-1 命中: **{t1:.3f}**（chance {ch:.3f}，n={nt} 段）")
        mr, sr, nr = sub["seg_rho"]
        L.append(f"- 段内 Spearman: **{mr:.3f}±{sr:.3f}**（n={nr} 段）")
        for mode in ("event", "seg_quota"):
            rc, g, nd = sub[f"recall_{mode}"]
            L.append(f"- ≥3★ 召回@Top12.5%（{mode}）: **{rc:.3f}**（{g}/{nd}）")
        st = sub["star"]
        L.append(f"- 0-5 容差（exact/±1/±2 累计）: 团顶清洁 {st['clean'][0]:.3f}/{st['clean'][1]:.3f}/"
                 f"{st['clean'][2]:.3f}(n={st['clean'][3]}) · 非团顶 {st['dirty'][0]:.3f}/"
                 f"{st['dirty'][1]:.3f}/{st['dirty'][2]:.3f}(n={st['dirty'][3]})")
        L.append("")
    L.append("## 绝对涌现代理（test 事件 abs_set 盲评照片，金标准未到）\n")
    for ev, (rho, n) in sorted(emerg.items()):
        L.append(f"- {ev}: Spearman {rho:.3f}（n={n}）")
    (out / "m4_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[OK] {out}/m4_report.md + model.pt + scores.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
