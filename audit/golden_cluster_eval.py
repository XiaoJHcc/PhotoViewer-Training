"""
golden_cluster_eval.py — 金标准团内盲选读回 + 三级判读（台阶② clean ground truth 首考）

输入：用户填好的 golden_clusters_answers.tsv + golden_clusters_key.csv。
判读三问（失败矩阵 §19 的判决实验）：
  1. **标签噪声实测**：用户盲选 vs 锦标赛团顶的一致率——若远低于 1，坐实"团内核级标签
     是去重压低+舍入产物"，台阶② 对旧标签不可学不可考；
  2. **模型首考**：给定 scores.csv（任意模型/系综），模型预测胜者 vs 用户盲选的一致率
     （分桶 d2p/d1/same 报；tie 团单列——真无差异区不应强迫模型二选一，宪法 v1.9）；
  3. **感知渠道对照**：CV 网格锐度统计（零训练）预测胜者 vs 用户盲选——若显著高于
     超分模型，证明连拍差异是 DINO@518 看不见的像素级信息，CV 融合是台阶② 的必要通路。

用法（仓根 D:/Git/PhotoViewer 下）：
    PYTHONUTF8=1 Tools/.venv/Scripts/python.exe Training/audit/golden_cluster_eval.py \
        --scores ens=Training/train/out/m8/scores_ens.csv
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent          # Training/
import sys
sys.path.insert(0, str(ROOT / "probes"))
from feature_probe import CV_IDX_EDGE_WIDTH_P20, CV_PLANE_LEN, CV_SCALAR_COUNT  # noqa: E402

GC_DIR = Path("D:/PhotoDB/dataset/golden_clusters")
DB_DEFAULT = "D:/PhotoDB/dataset/photos_dataset.db"


def load_answers(path: Path):
    out = {}
    for ln in open(path, encoding="utf-8"):
        ln = ln.rstrip("\n")
        if not ln or ln.startswith("#") or ln.startswith("gid"):
            continue
        parts = ln.split("\t")
        if len(parts) >= 2 and parts[1].strip():
            out[parts[0]] = parts[1].strip().upper()
    return out


def cv_sharp_scores(db: str, fps: list[str]):
    """CV 网格锐度平面（edge_width_p20，越小越锐）的 nanmean 取负 = "越锐越好"分。"""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out = {}
    try:
        for fp, blob in conn.execute(
                "SELECT fingerprint, cv_grid FROM photos WHERE fingerprint IN (%s)"
                % ",".join("?" * len(fps)), fps):
            if blob is None:
                continue
            arr = np.frombuffer(blob, dtype="<f4")
            if arr.size != CV_SCALAR_COUNT * CV_PLANE_LEN:
                continue
            grid = arr.reshape(CV_SCALAR_COUNT, CV_PLANE_LEN)
            with np.errstate(all="ignore"):
                out[fp] = -float(np.nanmean(grid[CV_IDX_EDGE_WIDTH_P20]))
    finally:
        conn.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="金标准团内盲选读回 + 三级判读")
    ap.add_argument("--gc", default=str(GC_DIR))
    ap.add_argument("--scores", nargs="*", default=[], help="name=path 模型分（可多个）")
    ap.add_argument("--db", default=DB_DEFAULT)
    args = ap.parse_args()

    gc = Path(args.gc)
    answers = load_answers(gc / "golden_clusters_answers.tsv")
    key = list(csv.DictReader(open(gc / "golden_clusters_key.csv", encoding="utf-8-sig")))
    by_gid = defaultdict(list)
    for r in key:
        by_gid[r["gid"]].append(r)
    rated = [gid for gid in by_gid if gid in answers]
    print(f"已标注 {len(rated)}/{len(by_gid)} 团")
    if not rated:
        print("[退出] 答案文件为空——请先完成 golden_clusters_answers.tsv")
        return 1

    # 判读 1：用户盲选 vs 锦标赛团顶
    agree, disagree, tie_n = 0, 0, 0
    for gid in rated:
        if answers[gid] == "TIE":
            tie_n += 1
            continue
        members = by_gid[gid]
        pick = next(m for m in members if m["anon"] == answers[gid])
        top_r = max(int(m["rating"]) for m in members)
        if int(pick["rating"]) == top_r:
            agree += 1
        else:
            disagree += 1
    dec = agree + disagree
    print(f"\n## 判读1 · 用户盲选 vs 锦标赛团顶（标签噪声实测）")
    print(f"- 一致 {agree}/{dec}（{agree / dec:.3f}）· 不一致 {disagree} · tie 团 {tie_n}")
    print("  （一致率低 = 坐实团内核级标签为压低+舍入产物；tie 率高 = 同星团确无差异）")

    # 判读 2/3：各预测源 vs 用户盲选
    sources = {}
    for spec in args.scores:
        name, path = spec.split("=", 1)
        sources[name] = {r["fingerprint"]: float(r["score"])
                         for r in csv.DictReader(open(path, encoding="utf-8-sig"))}
    all_fps = [m["fingerprint"] for ms in by_gid.values() for m in ms]
    sources["cv_sharp(零训练)"] = cv_sharp_scores(args.db, all_fps)

    print("\n## 判读2/3 · 各预测源 vs 用户盲选（tie 团单列）")
    print(f"{'源':<22} {'全体':>14} {'d2p':>14} {'d1':>14} {'same':>14}")
    for name, s_of in sources.items():
        agg = defaultdict(lambda: [0, 0])
        for gid in rated:
            if answers[gid] == "TIE":
                continue
            members = [m for m in by_gid[gid] if m["fingerprint"] in s_of]
            if len(members) < 2:
                continue
            pred = max(members, key=lambda m: s_of[m["fingerprint"]])
            ok = int(pred["anon"] == answers[gid])
            agg["全体"][0] += ok
            agg["全体"][1] += 1
            b = by_gid[gid][0]["bucket"]
            agg[b][0] += ok
            agg[b][1] += 1
        cells = []
        for b in ("全体", "d2p", "d1", "same"):
            ok, n = agg[b]
            cells.append(f"{ok}/{n}={ok / n:.2f}" if n else "-")
        print(f"{name:<22} {cells[0]:>14} {cells[1]:>14} {cells[2]:>14} {cells[3]:>14}")
    cnt = Counter(by_gid[g][0]["bucket"] for g in rated)
    print(f"\n桶分布: {dict(cnt)} · chance≈1/团均成员数")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
