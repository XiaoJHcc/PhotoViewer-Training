"""
horizon_eval.py — 水平度信号考卷评估（EXIF 横滚 × CV 地平线 × 组内共识，四变体）

信号变体（值越小越"平"，pick = argmin）：
  exif_abs : |roll 到最近 90° 倍数|（绝对水平，EXIF）
  exif_rel : |roll 偏离组内共识（mod 90 中位）|（近拍 burst 的"异类即歪"假设）
  cv_abs   : |CV 图像倾斜角|（视觉地平线，render518 梯度方向直方图）
  cv_rel   : |CV 倾斜角偏离组内共识|

产出：
  1. audit/out/horizon/scores_lvl.csv（exif_abs 单信号，可喂 golden_exam_eval.py）
  2. 考卷判定组：四变体 × 系综对照 四层剖面；按 Δerr 分层；按用户确信度 × 极相似带分层；
  3. 组合规则扫描（Δerr≥θ 改选最平）；
  4. golden_pairs：『更平者胜』胜率分桶（exif_abs / cv_abs）。

用法（仓根 D:/Git/PhotoViewer）：
    PYTHONUTF8=1 Tools/.venv/Scripts/python.exe Training/audit/horizon_eval.py \
        --scores-ens Training/train/out/m8_best/scores_ens.csv \
        --cv Training/audit/out/horizon/cv_horizon.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

EXAM_DIR = Path("D:/PhotoDB/dataset/golden_exam")
PAIRS = Path("Training/audit/out/golden_pairs/pairs_train.csv")
OUT_DIR = Path("Training/audit/out/horizon")
SIGNALS = ["exif_abs", "exif_rel", "cv_abs", "cv_rel"]


def dev90(x: float) -> float:
    """折到 [-45,45)：相对最近 0/90 轴的偏差。"""
    return (x + 45.0) % 90.0 - 45.0


def level_err(roll_deg: float) -> float:
    return abs(dev90(roll_deg))


def load_accel(path: str) -> dict[str, dict]:
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        if r["roll_deg"] == "":
            continue
        out[r["fingerprint"]] = dict(
            roll=float(r["roll_deg"]), pitch=float(r["pitch_deg"]),
            model=r["camera_model"], event=r["event_label"])
    return out


def load_cv(path: str) -> dict[str, float]:
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        if r["cv_tilt_deg"] != "":
            out[r["fingerprint"]] = float(r["cv_tilt_deg"])
    return out


def load_exam(exam: Path):
    key = list(csv.DictReader(open(exam / "golden_exam_key.csv", encoding="utf-8-sig")))
    kmeta = {(r["gid"], r["anon"]): r for r in key}
    rb = {}
    for ln in open(exam / "golden_exam_readback.tsv", encoding="utf-8"):
        ln = ln.rstrip("\n")
        if not ln or ln.startswith("gid"):
            continue
        g, a, fp, us, orating, iot, iup, st = ln.split("\t")
        rb[(g, a)] = dict(gid=g, anon=a, fingerprint=fp, user_star=int(us), status=st)
    by_gid = defaultdict(list)
    for (g, a), r in rb.items():
        by_gid[g].append(r)
    decided = []
    for g, ms in sorted(by_gid.items()):
        if ms[0]["status"] != "win":
            continue
        us = sorted([m["user_star"] for m in ms], reverse=True)
        meta0 = kmeta[(g, ms[0]["anon"])]
        decided.append(dict(gid=g, members=ms, winner_star=us[0],
                            second_star=us[1] if len(us) > 1 else us[0],
                            gap=us[0] - (us[1] if len(us) > 1 else us[0]),
                            event=meta0["event"], batch=meta0["batch"],
                            cos=float(meta0.get("cos") or 0.9)))
    return decided


def profile(rows):
    n = len(rows)
    if n == 0:
        return None
    devs = [r[0] for r in rows]
    return dict(n=n, exact=devs.count(0) / n,
                le1=sum(1 for d in devs if d <= 1) / n,
                top2=float(np.mean([r[1] for r in rows])))


def fmt(p):
    if p is None:
        return "-"
    return f"exact={p['exact']:.2f} ≤1={p['le1']:.2f} 前二={p['top2']:.2f} (n={p['n']})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--accel", default="D:/PhotoDB/dataset/accel_export.csv")
    ap.add_argument("--exam", default=str(EXAM_DIR))
    ap.add_argument("--pairs", default=str(PAIRS))
    ap.add_argument("--cv", default=None, help="horizon_cv.py 产出的 cv_horizon.csv")
    ap.add_argument("--scores-ens", default=None)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    accel = load_accel(args.accel)
    cv = load_cv(args.cv) if args.cv else {}
    print(f"EXIF 横滚: {len(accel)} 指纹 · CV 倾斜: {len(cv)} 指纹")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "scores_lvl.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fingerprint", "score"])
        for fp, a in accel.items():
            w.writerow([fp, f"{-level_err(a['roll']):.4f}"])

    decided = load_exam(Path(args.exam))
    ens = None
    if args.scores_ens:
        ens = {r["fingerprint"]: float(r["score"])
               for r in csv.DictReader(open(args.scores_ens, encoding="utf-8-sig"))}

    # 组装每成员四变体信号
    groups = []
    for d in decided:
        ms = []
        for m in d["members"]:
            fp = m["fingerprint"]
            sig = {}
            if fp in accel:
                sig["exif_abs"] = level_err(accel[fp]["roll"])
                sig["_exif_dev"] = dev90(accel[fp]["roll"])
            if fp in cv:
                sig["cv_abs"] = abs(cv[fp])
                sig["_cv_dev"] = cv[fp]
            if sig:
                ms.append(m | dict(sig=sig))
        if len(ms) < 2:
            continue
        # 组内共识（mod 90 中位），≥3 成员才有意义
        for src in ("_exif_dev", "_cv_dev"):
            have = [m["sig"][src] for m in ms if src in m["sig"]]
            if len(have) >= 3:
                med = float(np.median(have))
                rel = src.replace("_dev", "_rel").strip("_")  # exif_rel / cv_rel
                for m in ms:
                    if src in m["sig"]:
                        m["sig"][rel] = abs(dev90(m["sig"][src] - med))
        d = d | dict(members=ms)
        groups.append(d)
    print(f"考卷判定组: {len(groups)}/{len(decided)} 有任一信号")

    def pick(d, sig):
        cand = [m for m in d["members"] if sig in m["sig"]]
        if len(cand) < 2:
            return None
        return min(cand, key=lambda m: m["sig"][sig])

    def rows_for(sig, subset):
        rows = []
        for d in subset:
            if sig == "ens":
                cand = [m for m in d["members"] if ens and m["fingerprint"] in ens]
                if len(cand) < 2:
                    continue
                p = max(cand, key=lambda m: ens[m["fingerprint"]])
            else:
                p = pick(d, sig)
                if p is None:
                    continue
            rows.append((d["winner_star"] - p["user_star"],
                         int(p["user_star"] >= d["second_star"]), d))
        return rows

    # ── A. 全量判定组：四变体 vs 系综 ────────────────────────────────
    print("\n── A. 全量判定组（四变体 vs 系综）──")
    for sig in SIGNALS + (["ens"] if ens else []):
        print(f"  {sig:<9}:", fmt(profile(rows_for(sig, groups))))

    # ── B. 分层：极相似带 × 用户确信度 ──────────────────────────────
    print("\n── B. 分层（极相似带 cos≥0.96 / 低确信 gap=1）──")
    slices = [
        ("极相似带 cos≥0.96", [d for d in groups if d["cos"] >= 0.96]),
        ("相似带 <0.96", [d for d in groups if d["cos"] < 0.96]),
        ("低确信 gap=1", [d for d in groups if d["gap"] == 1]),
        ("高确信 gap≥2", [d for d in groups if d["gap"] >= 2]),
        ("极相似×低确信", [d for d in groups if d["cos"] >= 0.96 and d["gap"] == 1]),
    ]
    for tag, sub in slices:
        print(f"  [{tag}] n={len(sub)}")
        for sig in SIGNALS + (["ens"] if ens else []):
            p = profile(rows_for(sig, sub))
            if p:
                print(f"    {sig:<9}:", fmt(p))

    # ── C. 按组内 Δerr 分层（每变体自己的"有把握区间"）──────────────
    print("\n── C. 按组内 Δerr 分层（Δerr=次平−最平；检验信号是否'越明显越准'）──")
    for sig in SIGNALS:
        print(f"  [{sig}]")
        print(f"  {'Δerr≥':>6} {'n':>4} {'exact':>7} {'前二':>7}")
        for th in (0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0):
            sub = []
            for d in groups:
                vals = sorted(m["sig"][sig] for m in d["members"] if sig in m["sig"])
                if len(vals) >= 2 and vals[1] - vals[0] >= th:
                    sub.append(d)
            p = profile(rows_for(sig, sub))
            if p:
                print(f"  {th:>6.1f} {p['n']:>4} {p['exact']:>7.3f} {p['top2']:>7.3f}")

    # ── D. 组合规则扫描（系综 + 每变体 tiebreak）─────────────────────
    if ens:
        print("\n── D. 组合规则：组内 Δerr ≥ θ 时选最平者，否则跟系综 ──")
        for sig in SIGNALS:
            line = [f"  [{sig}]"]
            for th in (1e9, 1.5, 1.0, 0.7, 0.5, 0.3):
                rows = []
                for d in groups:
                    vals = sorted(m["sig"][sig] for m in d["members"] if sig in m["sig"])
                    use = len(vals) >= 2 and vals[1] - vals[0] >= th
                    if use:
                        p = pick(d, sig)
                    else:
                        cand = [m for m in d["members"] if m["fingerprint"] in ens]
                        p = max(cand, key=lambda m: ens[m["fingerprint"]]) if len(cand) >= 2 else None
                    if p is None:
                        continue
                    rows.append((d["winner_star"] - p["user_star"],
                                 int(p["user_star"] >= d["second_star"]), d))
                p = profile(rows)
                tag = "纯系综" if th > 1e8 else f"θ={th}"
                line.append(f"{tag}:{p['exact']:.2f}/{p['top2']:.2f}" if p else f"{tag}:-")
            print(" ".join(line))

    # ── E. 候选规则：极相似带条件水平度（band-conditional horizon rule）──────
    # 先验（用户 2026-08-02 注册）：极相似技术对的判别因子=清晰度/水平度；带外内容主导。
    # 门限依据：EXIF↔CV 交叉验证残差 MAD≈1° → Δerr<0.5° 视为噪声区不介入。
    if ens:
        print("\n── E. 候选规则考卷剖面（极相似带 cos≥0.96 内按 Δerr 门限介入，带外跟系综）──")
        for gate in (0.0, 0.3, 0.5, 0.7, 1.0):
            rows = []
            for d in groups:
                cand_e = [m for m in d["members"] if "exif_abs" in m["sig"]]
                if d["cos"] >= 0.96 and len(cand_e) >= 2:
                    errs = sorted(m["sig"]["exif_abs"] for m in cand_e)
                    if errs[1] - errs[0] >= gate:
                        p = min(cand_e, key=lambda m: m["sig"]["exif_abs"])
                        rows.append((d["winner_star"] - p["user_star"],
                                     int(p["user_star"] >= d["second_star"]), d))
                        continue
                cand = [m for m in d["members"] if m["fingerprint"] in ens]
                if len(cand) < 2:
                    continue
                p = max(cand, key=lambda m: ens[m["fingerprint"]])
                rows.append((d["winner_star"] - p["user_star"],
                             int(p["user_star"] >= d["second_star"]), d))
            print(f"  门限 {gate:.1f}°: {fmt(profile(rows))}")
        print("  对照  纯系综:", fmt(profile(rows_for("ens", groups))),
              " · 带内纯水平度:", end=" ")
        rows = []
        for d in groups:
            cand_e = [m for m in d["members"] if "exif_abs" in m["sig"]]
            if d["cos"] >= 0.96 and len(cand_e) >= 2:
                p = min(cand_e, key=lambda m: m["sig"]["exif_abs"])
            else:
                cand = [m for m in d["members"] if m["fingerprint"] in ens]
                if len(cand) < 2:
                    continue
                p = max(cand, key=lambda m: ens[m["fingerprint"]])
            rows.append((d["winner_star"] - p["user_star"],
                         int(p["user_star"] >= d["second_star"]), d))
        print(fmt(profile(rows)))

    # ── F. golden_pairs：『更平者胜』胜率分桶 ────────────────────────
    pp = Path(args.pairs)
    if pp.exists():
        buckets = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0), (1.0, 1.5), (1.5, 45.0)]
        for name, lut, key in (("exif_abs",
                                {fp: level_err(a["roll"]) for fp, a in accel.items()}, None),
                               ("cv_abs", {fp: abs(t) for fp, t in cv.items()}, None)):
            stat = {b: [0.0, 0.0] for b in buckets}
            n_use = 0
            for r in csv.DictReader(open(pp, encoding="utf-8-sig")):
                if r["fp_i"] not in lut or r["fp_j"] not in lut:
                    continue
                n_use += 1
                ei, ej = lut[r["fp_i"]], lut[r["fp_j"]]
                d = abs(ei - ej)
                w = float(r["weight"])
                for lo, hi in buckets:
                    if lo <= d < hi:
                        stat[(lo, hi)][1] += w
                        if ei < ej:
                            stat[(lo, hi)][0] += w
                        break
            print(f"\n── golden_pairs『更平者胜』[{name}]（n={n_use}）──")
            for lo, hi in buckets:
                wsum, tsum = stat[(lo, hi)]
                if tsum > 0:
                    print(f"  {lo:.1f}-{hi:.1f}° 胜率={wsum / tsum:.3f} 加权n={tsum:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
