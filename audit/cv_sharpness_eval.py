"""
cv_sharpness_eval.py — 清晰度（cv_grid edge_width_p20）全局选择规则考卷评估（负知识存证）

动机：2026-08-02 H 组错误分析发现 H005/H022 锐度边宽支持用户 → 试"极相似带内选最锐"规则。
结论（2026-08-05）：全局 p20 均值选择在带内 exact 0.33（n=9），弱于水平度（0.78），
与水平度组合 0.67 也不及纯水平度 —— 全局锐度聚合是错口径（天空/主体等权），
"哪块模糊该计分"须由 cvfuse 分块融合学习（用户处方），不做 CV 特判。存此为负知识。

口径：公共支撑（组内全体成员非 NaN 格子）上 edge_width_p20 均值，小=锐，选最小者。
另含逐格配对投票变体（2026-08-05 补）：公共支撑格内逐格比较边宽，Copeland 计票
（对每个对手，格级胜率 = 更锐格占比；组内取累计胜率最高者）——比全局均值更贴近
"区块对齐"思想，但仍无学习到的空间权重。平面 3（drag_width 抖动）同法另测。
用法（仓根 D:/Git/PhotoViewer）：
    PYTHONUTF8=1 Tools/.venv/Scripts/python.exe Training/audit/cv_sharpness_eval.py \
        --scores-ens Training/train/out/m8_best/scores_ens.csv
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from horizon_eval import load_accel, load_exam, level_err, profile, fmt  # noqa: E402

DB = "file:D:/PhotoDB/dataset/photos_dataset.db?mode=ro"
EXAM_DIR = Path("D:/PhotoDB/dataset/golden_exam")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exam", default=str(EXAM_DIR))
    ap.add_argument("--scores-ens", default=None)
    args = ap.parse_args()

    con = sqlite3.connect(DB, uri=True)

    def grid_plane(fp, plane):
        r = con.execute("SELECT cv_grid FROM photos WHERE fingerprint=?", (fp,)).fetchone()
        if not r or r[0] is None:
            return None
        d = np.frombuffer(r[0], dtype="<f4")
        return d[plane * 1024:(plane + 1) * 1024] if d.size == 7168 else None

    accel = load_accel("D:/PhotoDB/dataset/accel_export.csv")
    decided = load_exam(Path(args.exam))
    ens = ({r["fingerprint"]: float(r["score"]) for r in
            csv.DictReader(open(args.scores_ens, encoding="utf-8-sig"))}
           if args.scores_ens else None)

    def sharp_pick(ms, grids):
        mask = None
        for m in ms:
            g = grids.get(m["fingerprint"])
            if g is None:
                return None
            v = ~np.isnan(g)
            mask = v if mask is None else (mask & v)
        if mask is None or mask.sum() < 20:
            return None
        return min(ms, key=lambda m: float(np.mean(grids[m["fingerprint"]][mask])))

    def vote_pick(ms, grids, plane):
        """Copeland 计票：成员 m 对每对手 o 的格级胜率 = 公共支撑格中 m 更锐(边宽更小)占比；
        组内取平均胜率最高者。"""
        mask = None
        for m in ms:
            g = grids.get(m["fingerprint"])
            if g is None:
                return None
            v = ~np.isnan(g)
            mask = v if mask is None else (mask & v)
        if mask is None or mask.sum() < 20:
            return None
        W = {m["fingerprint"]: grids[m["fingerprint"]][mask] for m in ms}
        fps = [m["fingerprint"] for m in ms]
        score = {}
        for a in fps:
            wins = []
            for b in fps:
                if a == b:
                    continue
                d = W[a] - W[b]  # <0 = a 更锐
                valid = np.isfinite(d)
                if valid.sum() < 20:
                    return None
                wins.append(float(np.mean(d[valid] < 0)))
            score[a] = float(np.mean(wins))
        return max(ms, key=lambda m: score[m["fingerprint"]])

    for tag, sub in [("全体", decided),
                     ("极相似带 cos>=0.96", [d for d in decided if d["cos"] >= 0.96]),
                     ("相似带 <0.96", [d for d in decided if d["cos"] < 0.96])]:
        rows = {k: [] for k in ("sharp", "vote", "drag", "lvl", "combo", "ens")}
        for d in sub:
            ms = d["members"]
            g20 = {m["fingerprint"]: grid_plane(m["fingerprint"], 1) for m in ms}
            gdrag = {m["fingerprint"]: grid_plane(m["fingerprint"], 3) for m in ms}
            ps = sharp_pick(ms, g20)
            pv = vote_pick(ms, g20, 1)
            pd = vote_pick(ms, gdrag, 3)
            ms_a = [m for m in ms if m["fingerprint"] in accel]
            pl = (min(ms_a, key=lambda m: level_err(accel[m["fingerprint"]]["roll"]))
                  if len(ms_a) >= 2 else None)
            pe = None
            if ens:
                cand = [m for m in ms if m["fingerprint"] in ens]
                if len(cand) >= 2:
                    pe = max(cand, key=lambda m: ens[m["fingerprint"]])
            picks = {}
            if ps is not None:
                picks["sharp"] = ps
                if pl is not None:
                    picks["combo"] = ps if pl["fingerprint"] == ps["fingerprint"] else pe
            if pv is not None:
                picks["vote"] = pv
            if pd is not None:
                picks["drag"] = pd
            if pl is not None:
                picks["lvl"] = pl
            if pe is not None:
                picks["ens"] = pe
            for k, p in picks.items():
                if p is not None:
                    rows[k].append((d["winner_star"] - p["user_star"],
                                    int(p["user_star"] >= d["second_star"])))
        print(f"[{tag}]")
        for k, name in (("sharp", "锐度全局均值"), ("vote", "锐度逐格投票"),
                        ("drag", "抖动逐格投票"), ("lvl", "水平度"),
                        ("combo", "锐度+水平一致改选"), ("ens", "系综")):
            if rows[k]:
                print(f"  {name:<14}:", fmt(profile(rows[k])))


if __name__ == "__main__":
    raise SystemExit(main())
