"""事件数学习曲线：留一事件评价，用 k 个其它 train 事件拟合同一类别均衡 joint 单图头，看留出指标是否随事件数上升。

诊断“能背不能带”是否主要是独立事件太少：每个留出事件 × 每个 k 随机抽若干组训练事件（固定种子），
报留出事件 AP≥1/≥3 与同段相似跨档对准确率的均值。只读 train 事件，不打开 val/test，不选参数（正则固定 0.1）。
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "audit"))
import rating_baseline as rb
from evaluation_protocol import binary_ranking, write_json
from scene_comparator import segments_of, similar_edges

SIZES = (2, 4, 7, 13)
DRAWS = 2


def held_metrics(expected, tiers, left, right, members):
    """留出事件的 AP≥1/≥3 与同段相似跨档对准确率。"""
    cross = tiers[left] != tiers[right]
    margin = (expected[left] - expected[right])[cross] * np.sign(tiers[left] - tiers[right])[cross]
    return {"ap1": binary_ranking(tiers[members] >= 1, expected[members])["ap"],
        "ap3": binary_ranking(tiers[members] >= 3, expected[members])["ap"],
        "within_scene": float(np.mean(np.where(margin == 0, 0.5, margin > 0))) if cross.any() else None}


def main() -> int:
    """逐留出事件、逐训练事件数拟合并评价，写出均值曲线与逐次记录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    threadpool_limits(2)
    inputs = argparse.Namespace(photos=ROOT / "audit/out/m3_pairs_v2_20260924b/photos.csv",
        split_json=ROOT / "data/split-v2-20260924.json", features=ROOT / "train/out/recovery_work/corrected_features.npz",
        local=ROOT / "train/out/recovery_work/rescored_lora_v2/scores.csv",
        global_scores=ROOT / "train/out/recovery_work/corrected_rank/clip_scores.csv")
    metadata, fingerprints, features, _ = rb.load_inputs(inputs)
    tiers = np.array([min(int(metadata[fp]["rating_raw"]), 3) for fp in fingerprints])
    event_of = np.array([metadata[fp]["event"] for fp in fingerprints])
    split_of = np.array([metadata[fp]["split"] for fp in fingerprints])
    segment_key = np.array([metadata[fp]["event"] + "|" + metadata[fp]["seg_id"] for fp in fingerprints])
    dino = features["dino"] / np.sqrt(384)
    train_events = sorted(set(event_of[split_of == "train"].tolist()),
        key=lambda event: hashlib.sha256(("20260926:" + event).encode()).hexdigest())
    records = []
    for held in train_events:
        members = np.flatnonzero(event_of == held)
        left, right, _ = similar_edges(segments_of(members, segment_key), dino)
        others = [event for event in train_events if event != held]
        rng = np.random.default_rng(int(hashlib.sha256(held.encode()).hexdigest()[:8], 16))
        for size in SIZES:
            draws = 1 if size == len(others) else DRAWS
            for draw in range(draws):
                chosen = sorted(rng.choice(others, size=size, replace=False).tolist())
                fitting = np.isin(event_of, chosen)
                if set(tiers[fitting].tolist()) != set(range(4)):
                    continue
                coefficient, _ = rb.fit_classifier(features["joint"][fitting], tiers[fitting], event_of[fitting].tolist(),
                    0.1, "event_class")
                expected = np.zeros(len(fingerprints))
                expected[members] = rb.predict(features["joint"][members], coefficient) @ np.arange(4.0)
                records.append({"held_out": held, "events": size, "draw": draw, "fit_photos": int(fitting.sum()),
                    **held_metrics(expected, tiers, left, right, members)})
        print(f"{held}: done", flush=True)
    curve = dict()
    for size in SIZES:
        rows = [row for row in records if row["events"] == size]
        per_event = dict()
        for row in rows:
            per_event.setdefault(row["held_out"], []).append(row)
        curve[str(size)] = dict((key, float(np.mean([np.mean([r[key] for r in group if r[key] is not None])
            for group in per_event.values()]))) for key in ("ap1", "ap3", "within_scene"))
        curve[str(size)]["held_out_events"] = len(per_event)
        print(f"k={size}: {curve[str(size)]}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "report.json", {"scope": "train 事件留一评价；正则固定 0.1，类别均衡", "sizes": SIZES,
        "draws": DRAWS, "curve": curve, "records": records})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
