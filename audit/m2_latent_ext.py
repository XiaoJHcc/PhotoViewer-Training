"""
m2_latent_ext.py — 潜分尺扩展：新入库事件（无盲评锚点）挂入 M3 配对域

背景（2026-08-16，2025 批 13 事件入库后）：m3_pair_gen 要求照片在潜分表内才参与配对
（window/global/derived 共用同一过滤）。新事件未做 M2 盲评锚定，b_seg 未标识。
本脚本把新事件照片追加进潜分表：g 按星照查（g 是纯星函数，已验证每星唯一值），
b_seg/score 置 NaN ——派生对的 |Δs| 比较遇 NaN 为 False 自然剔除（跨段/跨事件派生对
必须有真锚点，宪法"排序进排序出"），window（段内序）/global（事件内序）不受影响。
待用户按 M2 同法对新事件盲评后，重跑 m2_offset_fit 出正式潜分，本扩展即作废。

用法（仓根 D:/Git/PhotoViewer 下）：
    PYTHONUTF8=1 Tools/.venv/Scripts/python.exe Training/audit/m2_latent_ext.py
然后 m3_pair_gen 加 --latent-clean Training/audit/out/m2_offset_clean/latent_scores_ext.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

BASE = Path(__file__).resolve().parent / "out" / "m2_offset_clean" / "latent_scores.csv"
CLUSTERS = Path(__file__).resolve().parent / "out" / "clusters" / "clusters.csv"
OUT = Path(__file__).resolve().parent / "out" / "m2_offset_clean" / "latent_scores_ext.csv"


def main() -> int:
    base_rows = list(csv.DictReader(open(BASE, encoding="utf-8-sig")))
    g_of = {int(r["rating"]): float(r["g"]) for r in base_rows}
    assert all(isinstance(v, float) for v in g_of.values())
    have = {r["fingerprint"] for r in base_rows}

    added = 0
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fingerprint", "event", "seg_id", "cluster_id", "rating", "g", "b_seg", "score"])
        for r in base_rows:
            w.writerow([r["fingerprint"], r["event"], r["seg_id"], r["cluster_id"],
                        r["rating"], r["g"], r["b_seg"], r["score"]])
        for r in csv.DictReader(open(CLUSTERS, encoding="utf-8-sig")):
            fp = r["fingerprint"]
            if fp in have:
                continue
            rating = int(r["rating"])
            w.writerow([fp, r["event"], r["seg_id"], r["cluster_id"], rating,
                        f"{g_of[rating]:.4f}", "nan", "nan"])
            added += 1
    print(f"[OK] {OUT}：基线 {len(base_rows)} + 扩展 {added}（b_seg/score=NaN，derived 自然剔除）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
