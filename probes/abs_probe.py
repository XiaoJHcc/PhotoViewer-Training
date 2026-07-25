"""
abs_probe.py — 绝对星级线性探针（Plan-3-1 §1.5 门槛，M1 关门判据）

与 feature_probe.py（相对对级、段级 split）互补：本脚本测"绝对标尺"——
用户对 abs_set 200 张（原库 ≥3★ 子集）按全量程 0-5 重标的新星级，
能否由 CLS 特征线性读出。新 0★ = 该子集内最弱（不是全库意义上的废片），
因此本探针测的是**顶部 band 内部的细粒度排序能力**，正是锦标赛要用的区间。

口径设计（与任务书 §1.5 对齐）：
    1. 样本 = abs_set_key.csv 中 200 个指纹，标签 = abs_set_ratings.tsv 读回的新星级；
       两两配对，只保留新星级不同的对（设计上跨事件/跨段，不做窗口限制）。
    2. 主口径 = 事件级留出（LOEO）：留一事件的所有照片及其对内对做测试，
       训练用其余事件的对——测的正是挑战 3（跨事件迁移），对应锦标赛"新拍一场"场景。
       对级 5 折（同一事件可能跨 train/test，泄漏）只作乐观上界对照。
    3. 辅助：留一事件的 Ridge 回归（照片级特征 → 新星级）Spearman 相关；
       按 |Δrating| 与相邻边界（0-1..4-5）拆解对内准确率。
    4. 视图 = {ViT-S 原片, ViT-S 增强, ViT-L 原片, ViT-L 增强} 四路 CLS（库里现成），
       主视图 = ViT-S 增强（v1 冻结 backbone），ViT-L 只作免费对照。

判读（§1.5）：主视图 LOEO 明显超 50% chance → 特征泛化腿强，M2 偏移监督可轻量；
贴 chance → M2 人工偏移（3a）加重、M5 大概率触发。

用法：
    Training/.venv/Scripts/python.exe Training/probes/abs_probe.py
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_probe import l2_normalize, _make_clf, probe_accuracy   # noqa: E402

MODELS = {
    "S-orig": "dinov3_vits16_f32_518_v1",
    "S-enh":  "dinov3_vits16_f32_518_v1+clhe2.0ycc1.0",
    "L-orig": "dinov3_vitl16_f32_518_v1",
    "L-enh":  "dinov3_vitl16_f32_518_v1+clhe2.0ycc1.0",
}
PRIMARY_VIEW = "S-enh"
MIN_TEST_PAIRS = 10        # 留一事件内少于该对数则该事件不参与 LOEO 准确率统计
RATING_LEVELS = 6          # 0..5


def load_abs_labels(key_path: str, ratings_path: str):
    """key（new_name→fingerprint/event/old_rating/seg_id）⨝ 读回的新星级。"""
    key = {r["new_name"]: r for r in csv.DictReader(open(key_path, encoding="utf-8-sig"))}
    rows = []
    for line in open(ratings_path, encoding="utf-8"):
        name, rating = line.rstrip("\n").split("\t")
        if name not in key:
            raise SystemExit(f"[ERROR] 读回文件 {name} 不在 key 表中")
        rows.append((name, key[name]["fingerprint"], key[name]["event_label"],
                     int(key[name]["old_rating"]), int(key[name]["seg_id"]), int(rating)))
    missing = set(key) - {r[0] for r in rows}
    if missing:
        raise SystemExit(f"[ERROR] key 表中 {len(missing)} 个文件缺读回星级：{sorted(missing)[:5]}")
    return rows


def load_cls_views(db_path: str, fingerprints: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """四个模型 × 每个指纹的 CLS（L2 归一化）。缺任何一路的指纹直接剔除并报数。"""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        feats: dict[str, dict[str, np.ndarray]] = {}
        q = ("SELECT fingerprint, model_id, cls_vector FROM photo_features "
             f"WHERE fingerprint IN ({','.join('?' * len(fingerprints))})")
        for fp, model_id, blob in conn.execute(q, fingerprints):
            feats.setdefault(fp, {})[model_id] = l2_normalize(
                np.frombuffer(blob, dtype="<f4").astype(np.float32))
    finally:
        conn.close()
    views: dict[str, dict[str, np.ndarray]] = {}
    for view, model_id in MODELS.items():
        views[view] = {fp: f[model_id] for fp, f in feats.items() if model_id in f}
    return views


def make_pairs(ratings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """所有 i<j 且新星级不同的对。"""
    ii, jj = np.triu_indices(len(ratings), k=1)
    m = ratings[ii] != ratings[jj]
    return ii[m], jj[m]


def loeo_oof(feat: np.ndarray, ii: np.ndarray, jj: np.ndarray, ratings: np.ndarray,
             events: np.ndarray, clf_kind: str = "linear"):
    """
    留一事件 out-of-fold：训练 = 两端都不在该事件的对；逐对返回
    (correct_within, evaluated_within, correct_touch, evaluated_touch)——
    within = 两端同处留出事件（对内排序）；touch = 恰好一端在留出事件（跨事件定标）。
    """
    diff = feat[ii] - feat[jj]
    label = np.where(ratings[ii] > ratings[jj], 1, -1)
    ev_i, ev_j = events[ii], events[jj]
    cw = np.zeros(len(ii), bool); ew = np.zeros(len(ii), bool)
    ct = np.zeros(len(ii), bool); et = np.zeros(len(ii), bool)
    for ev in np.unique(events):
        tr = (ev_i != ev) & (ev_j != ev)
        te_w = (ev_i == ev) & (ev_j == ev)
        te_t = ((ev_i == ev) != (ev_j == ev))
        if tr.sum() < MIN_TEST_PAIRS:
            continue
        clf = _make_clf(clf_kind).fit(
            np.concatenate([diff[tr], -diff[tr]]), np.concatenate([label[tr], -label[tr]]))
        if hasattr(clf, "decision_function"):          # linear：对称决策面
            dec = clf.decision_function(diff) * label > 0
        else:                                          # mlp 等：±diff 两票平均判向
            p = clf.predict_proba(np.vstack([diff, -diff]))[:, 1]
            dec = (p[:len(ii)] - p[len(ii):]) * label > 0
        if te_w.sum() >= MIN_TEST_PAIRS:
            cw[te_w] = dec[te_w]; ew[te_w] = True
        if te_t.sum() >= MIN_TEST_PAIRS:
            ct[te_t] = dec[te_t]; et[te_t] = True
    return cw, ew, ct, et


def loeo_ridge_oof(feat: np.ndarray, ratings: np.ndarray, events: np.ndarray):
    """照片级 Ridge：训练 = 其余事件照片（feat→rating），预测留出事件每张。返回 OOF 预测。"""
    from sklearn.linear_model import Ridge
    pred = np.full(len(ratings), np.nan)
    for ev in np.unique(events):
        tr = events != ev
        if tr.sum() < 20 or (~tr).sum() < 4:
            continue
        reg = Ridge(alpha=1.0).fit(feat[tr], ratings[tr])
        pred[~tr] = reg.predict(feat[~tr])
    return pred


def _acc(correct: np.ndarray, evaluated: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
    m = evaluated & mask
    return (float(correct[m].mean()) if m.sum() else float("nan"), int(m.sum()))


def label_structure(new: np.ndarray, old: np.ndarray, events: np.ndarray,
                    segs: np.ndarray) -> dict:
    """
    标签侧结构（不碰特征）——回答"M2 要补的绝对尺缺口有多大、长什么样"：
      1. 老↔新序相关（Spearman）：重标是盲评（抹星级中性名副本），老↔新的序相关不是构造产物，
         它测"老锦标赛结果携带多少真实的（跨事件）绝对序信息"。注意：等级映射无意义
         （新量程是用户自选的构造，新 0★ ≠ 库 0★），只有**序**可比。
      2. 嵌套方差分解 事件/段/张：新绝对尺的方差有多少在事件间、多少在事件内段间、
         多少在段内张间——直接量出挑战 3 的体量与形状，也就是 M2 b_seg 偏移要吸收的份额
         （事件间+段间）vs f(x)/段内排序必须自己解释的份额（段内张间）。
         注意适用域：abs_set 是 ≥3★ 顶部 band，份额只对顶部 band 成立。
      3. 每事件新星级均值表：跨事件尺度错位的直接地图（"好天 3★ vs 坏天 3★"的实测）。
    """
    from scipy.stats import spearmanr
    out: dict = {}
    out["rho_old_new"] = float(spearmanr(old, new).statistic)

    grand = new.mean()
    ss_tot = float(((new - grand) ** 2).sum())
    ss_event = 0.0
    ss_seg = 0.0
    seg_mean: dict[int, float] = {}
    ev_of_seg: dict[int, str] = {}
    for s in np.unique(segs):
        ms = segs == s
        seg_mean[int(s)] = float(new[ms].mean())
        ev_of_seg[int(s)] = str(events[ms][0])
    ev_mean: dict[str, float] = {}
    for e in np.unique(events):
        me = events == e
        ev_mean[str(e)] = float(new[me].mean())
        ss_event += float(me.sum() * (ev_mean[str(e)] - grand) ** 2)
    for s in np.unique(segs):
        ms = segs == s
        ss_seg += float(ms.sum() * (seg_mean[int(s)] - ev_mean[ev_of_seg[int(s)]]) ** 2)
    ss_within = ss_tot - ss_event - ss_seg
    out["var_share"] = dict(
        event=ss_event / ss_tot, seg=ss_seg / ss_tot, within=ss_within / ss_tot,
        n_event=len(ev_mean), n_seg=len(seg_mean))
    out["event_table"] = sorted(
        ((e, int((events == e).sum()), m) for e, m in ev_mean.items()),
        key=lambda t: t[2], reverse=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="D:/PhotoDB/dataset/photos_dataset.db")
    ap.add_argument("--key", default="D:/PhotoDB/dataset/abs_set_key.csv")
    ap.add_argument("--ratings", default="D:/PhotoDB/dataset/abs_set_ratings.tsv")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "out_abs"))
    ap.add_argument("--clf", default="linear", choices=["linear", "mlp"],
                    help="读出头：linear=全局线性方向（主口径）；mlp=非线性头（检验'方向非线性可救'）")
    ap.add_argument("--views", default=",".join(MODELS.keys()),
                    help="逗号分隔的视图子集（默认全部四路）")
    args = ap.parse_args()
    use_views = [v for v in MODELS if v in args.views.split(",")]
    if not use_views:
        raise SystemExit(f"[ERROR] --views 无有效项（可选 {list(MODELS)}）")
    suffix = "" if args.clf == "linear" else f"_{args.clf}"

    rows = load_abs_labels(args.key, args.ratings)
    names = [r[0] for r in rows]; fps = [r[1] for r in rows]
    events_all = np.array([r[2] for r in rows])
    old = np.array([r[3] for r in rows]); new = np.array([r[5] for r in rows])
    segs_all = np.array([r[4] for r in rows])

    views_fp = load_cls_views(args.db, fps)
    keep = [i for i, fp in enumerate(fps) if all(fp in views_fp[v] for v in use_views)]
    if len(keep) < len(fps):
        print(f"[WARN] {len(fps) - len(keep)} 个指纹缺特征，剔除")
    idx = np.array(keep)
    ratings = new[idx]; events = events_all[idx]; segs = segs_all[idx]
    feats = {v: np.stack([views_fp[v][fps[i]] for i in idx]) for v in use_views}

    print(f"样本 {len(idx)} 张 | 事件 {len(set(events))} 个 | 读出头 {args.clf} | "
          f"新星级分布 {dict(sorted(Counter(ratings.tolist()).items()))}")

    # 老↔新星级对照（数据本身；等级映射是用户自选量程的构造，只有"序"可比）
    print("\n老星级(行) × 新星级(列) 对照（等级映射无意义，看序）：")
    header = "      " + "".join(f"新{k:d}★".ljust(6) for k in range(RATING_LEVELS))
    print(header)
    for o in sorted(set(old[idx].tolist())):
        row = [int(((old[idx] == o) & (ratings == k)).sum()) for k in range(RATING_LEVELS)]
        print(f"老{o}★  " + "".join(str(c).ljust(6) for c in row))

    # 标签侧结构：挑战 3 的体量与形状（M2 b_seg 要吸收的份额）
    ls = label_structure(ratings, old[idx], events, segs)
    print(f"\n=== 标签侧结构（挑战 3 实测）===")
    print(f"老↔新盲评序相关 Spearman = {ls['rho_old_new']:.3f}（重标盲评，序相关非构造产物）")
    vs = ls["var_share"]
    print(f"新星级方差分解（事件/段/张，共 {vs['n_event']} 事件 {vs['n_seg']} 段）："
          f"事件间 {vs['event']:.1%} | 事件内段间 {vs['seg']:.1%} | 段内张间 {vs['within']:.1%}")
    print("每事件新星级均值（跨事件尺度错位地图，降序）：")
    for e, n, m in ls["event_table"]:
        print(f"  {e:<28} n={n:<4} new_mean={m:.2f}")

    ii, jj = make_pairs(ratings)
    print(f"\n可用对（新星级不同）: {len(ii)} / 全对 {len(idx) * (len(idx) - 1) // 2}")

    # ------------------------------------------------------------------
    report: dict[str, dict] = {}
    for view, feat in feats.items():
        # 对级 5 折（泄漏上界）
        pair_acc, pair_std, _ = probe_accuracy(feat, ii, jj, ratings, clf_kind=args.clf)
        # LOEO 对内 / 跨事件
        cw, ew, ct, et = loeo_oof(feat, ii, jj, ratings, events, clf_kind=args.clf)
        w_acc, w_n = _acc(cw, ew, np.ones(len(ii), bool))
        t_acc, t_n = _acc(ct, et, np.ones(len(ii), bool))
        # 按 |Δ| 与相邻边界拆（对内 OOF）
        delta = np.abs(ratings[ii] - ratings[jj])
        by_delta = {d: _acc(cw, ew, delta == d) for d in range(1, RATING_LEVELS)}
        bnd = np.minimum(ratings[ii], ratings[jj])
        by_bnd = {b: _acc(cw, ew, (delta == 1) & (bnd == b)) for b in range(RATING_LEVELS - 1)}
        # Ridge Spearman
        from scipy.stats import spearmanr
        pred = loeo_ridge_oof(feat, ratings, events)
        m = ~np.isnan(pred)
        rho_all = float(spearmanr(ratings[m], pred[m]).statistic)
        rhos = [float(spearmanr(ratings[events == ev], pred[events == ev]).statistic)
                for ev in np.unique(events)
                if m[events == ev].sum() >= 4 and len(set(ratings[events == ev])) > 1]
        report[view] = dict(pair=(pair_acc, pair_std), within=(w_acc, w_n), touch=(t_acc, t_n),
                            by_delta=by_delta, by_bnd=by_bnd,
                            rho=(rho_all, float(np.mean(rhos)), float(np.std(rhos))), pred=pred)

    # ------------------------------------------------------------------
    primary = PRIMARY_VIEW if PRIMARY_VIEW in report else use_views[0]
    print("\n=== 主表（chance=50%）===")
    print(f"{'视图':<8}{'对级5折(泄漏)':<18}{'LOEO 对内':<18}{'LOEO 跨事件':<18}{'Spearman 池化/事件均±std'}")
    for view in use_views:
        r = report[view]
        tag = " *主" if view == primary else ""
        print(f"{view + tag:<8}"
              f"{r['pair'][0]:.3f}±{r['pair'][1]:.3f}    "
              f"{r['within'][0]:.3f} (n={r['within'][1]:<5})  "
              f"{r['touch'][0]:.3f} (n={r['touch'][1]:<5})  "
              f"{r['rho'][0]:.3f} / {r['rho'][1]:.3f}±{r['rho'][2]:.3f}")

    print(f"\n=== 主视图 {primary} LOEO 对内：按 |Δrating| 拆 ===")
    for d, (a, n) in report[primary]["by_delta"].items():
        print(f"  Δ={d}: {a:.3f} (n={n})")
    print(f"=== 主视图 {primary} LOEO 对内：Δ=1 相邻边界拆 ===")
    for b, (a, n) in report[primary]["by_bnd"].items():
        print(f"  {b}★ vs {b + 1}★: {a:.3f} (n={n})")

    # ------------------------------------------------------------------
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / f"abs_probe{suffix}.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write(f"# abs_probe — 绝对星级探针（Plan-3-1 §1.5，读出头 {args.clf}）\n\n")
        f.write(f"- 样本：{len(idx)} 张 abs_set（原库 ≥3★ 子集全量程重标），事件 {len(set(events))} 个\n")
        f.write(f"- 新星级分布：{dict(sorted(Counter(ratings.tolist()).items()))}\n")
        f.write(f"- 可用对：{len(ii)}；chance = 50%；主口径 = LOEO 对内（事件级留出）\n\n")
        f.write(f"## 标签侧结构（挑战 3 实测，不碰特征）\n\n"
                f"- 老↔新盲评序相关 Spearman = **{ls['rho_old_new']:.3f}**（等级映射是用户自选量程的构造，只有序可比）\n"
                f"- 新星级方差分解：事件间 **{vs['event']:.1%}** | 事件内段间 **{vs['seg']:.1%}** | 段内张间 **{vs['within']:.1%}**"
                f"（{vs['n_event']} 事件 {vs['n_seg']} 段；≥3★ 顶部 band 口径）\n"
                f"- 每事件新星级均值（降序）："
                + "；".join(f"{e} {m:.2f}(n={n})" for e, n, m in ls["event_table"]) + "\n\n")
        f.write("| 视图 | 对级5折(泄漏) | LOEO 对内 | LOEO 跨事件 | Spearman 池化 | Spearman 事件均±std |\n")
        f.write("|---|---|---|---|---|---|\n")
        for view in use_views:
            r = report[view]
            star = " **(主)**" if view == primary else ""
            f.write(f"| {view}{star} | {r['pair'][0]:.3f}±{r['pair'][1]:.3f} "
                    f"| {r['within'][0]:.3f} (n={r['within'][1]}) "
                    f"| {r['touch'][0]:.3f} (n={r['touch'][1]}) "
                    f"| {r['rho'][0]:.3f} | {r['rho'][1]:.3f}±{r['rho'][2]:.3f} |\n")
        f.write(f"\n## 主视图 {primary} 拆解\n\n按 |Δrating|（LOEO 对内）：\n\n")
        for d, (a, n) in report[primary]["by_delta"].items():
            f.write(f"- Δ={d}: {a:.3f} (n={n})\n")
        f.write("\nΔ=1 相邻边界（LOEO 对内）：\n\n")
        for b, (a, n) in report[primary]["by_bnd"].items():
            f.write(f"- {b}★ vs {b + 1}★: {a:.3f} (n={n})\n")
    print(f"\n[OK] 报告写出 {md}")

    # 图：主视图 Ridge OOF 预测 vs 新星级（散点+抖动），给 analysis-story 用
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pred = report[primary]["pred"]; m = ~np.isnan(pred)
        rng = np.random.default_rng(0)
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        ax.scatter(ratings[m] + rng.uniform(-0.12, 0.12, m.sum()),
                   pred[m], s=22, alpha=0.65, edgecolors="none")
        means = [pred[m & (ratings == k)].mean() for k in range(RATING_LEVELS)]
        ax.plot(range(RATING_LEVELS), means, "o-", color="crimson", label="per-level mean")
        ax.set_xlabel("new rating (abs_set re-label)"); ax.set_ylabel(f"LOEO ridge prediction ({primary})")
        ax.set_xticks(range(RATING_LEVELS)); ax.legend(); ax.grid(alpha=0.3)
        ax.set_title(f"abs_probe LOEO ridge: Spearman={report[primary]['rho'][0]:.3f}")
        fig.tight_layout(); fig.savefig(out_dir / f"abs_probe_oof{suffix}.png", dpi=140)
        print(f"[OK] 图写出 {out_dir / f'abs_probe_oof{suffix}.png'}")
    except Exception as e:                                   # noqa: BLE001
        print(f"[WARN] 画图失败（不影响数值结果）: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
