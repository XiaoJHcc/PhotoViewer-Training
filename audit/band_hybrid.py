"""
band_hybrid.py — 带条件混合部署分数合成（台阶② 候选形态）

机制：金标准考卷内，极相似带团（key.cos ≥ 0.96）改用专家模型分数，
带外照旧用守擂系综分数——专家的全局扭曲不计代价（"靶区学得进、全局必扭曲"
困境下的合理交付形态：LoRA 重训模型只在其能学的地带服役）。

用法（仓根 D:/Git/PhotoViewer 下）：
    PYTHONUTF8=1 Tools/.venv/Scripts/python.exe Training/audit/band_hybrid.py \
        --base ens=Training/train/out/m8_best/scores_ens.csv \
        --expert hz=Training/train/out/m5_cvfuse_hz20/scores_ep2.csv \
        --out Training/audit/out/horizon/scores_hybrid_hz.csv
然后用 golden_exam_eval.py --scores hyb=<产物> 评四层剖面。

口径注意：z-score 在考卷照片集合内各自估计（两组分数只在团内两两比较，
团间尺度差异不影响判定）；带归属 = key.csv 的 cos 列（缺省 0.9=带外，
与 golden_exam_eval.py 同口径）。
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

EXAM_DIR = Path("D:/PhotoDB/dataset/golden_exam")
COS_HI = 0.96


def _load_scores(path: str) -> dict[str, float]:
    """scores csv（fingerprint,score 或含更多列）→ fp → float。"""
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        out[r["fingerprint"]] = float(r["score"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="带条件混合：极相似带用专家分，带外用基线分")
    ap.add_argument("--base", required=True, help="name=path（带外基线，如 ens）")
    ap.add_argument("--expert", required=True, help="name=path（带内专家）")
    ap.add_argument("--exam", default=str(EXAM_DIR))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base_name, base_path = args.base.split("=", 1)
    exp_name, exp_path = args.expert.split("=", 1)
    base, exp = _load_scores(base_path), _load_scores(exp_path)

    key = list(csv.DictReader(open(Path(args.exam) / "golden_exam_key.csv",
                                encoding="utf-8-sig")))
    rows = [(r["fingerprint"], float(r["cos"]) >= COS_HI if r["cos"] else False)
            for r in key]
    fps = [fp for fp, _ in rows]

    def z(scores: dict[str, float]) -> dict[str, float]:
        v = np.array([scores[fp] for fp in fps], dtype=np.float64)
        sd = v.std()
        return {fp: (scores[fp] - v.mean()) / (sd if sd > 0 else 1.0) for fp in fps}

    zb, ze = z(base), z(exp)
    n_band = sum(1 for _, b in rows if b)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fingerprint", "score"])
        for fp, in_band in rows:
            w.writerow([fp, f"{(ze if in_band else zb)[fp]:.6f}"])
    print(f"[OK] {out_path}：{len(fps)} 张（带内 {n_band} 用 {exp_name}，"
          f"带外 {len(fps) - n_band} 用 {base_name}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
