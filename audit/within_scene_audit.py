"""同场景差异定位审计：把单图四档头的剩余差距拆到“段间 / 段内相似组内 / 段内跨组”，只读 train 事件 OOF。

不训练新模型；复用 rating_baseline 的类别均衡 joint 小头三折 OOF（与正式臂逐位复现）。输出：
① 星级与模型分数方差的段间份额、段内分位化对照；② 理想段内重排上界（2s 连拍 / 相似组 cos≥0.95 / ≥0.88）
与“组内是否有正例”上界；③ 同段跨档对按 DINO 余弦分带的准确率；④ 0★ 的近重高档兄弟占比。
均为诊断，不作验收；不打开 val/test，不使用拍摄序位作打分信号。
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from pathlib import Path
import sqlite3
import sys

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "audit"))
import rating_baseline as rb
from evaluation_protocol import binary_ranking, write_json

BANDS = ((0.0, 0.73), (0.73, 0.88), (0.88, 0.95), (0.95, 0.98), (0.98, 1.01))


def baseline_oof(features, tiers, event_of, train_events):
    """复现正式类别均衡 joint 小头（正则 0.1）的训练事件三折 OOF 四档概率。"""
    threadpool_limits(2)
    folds = [train_events[fold::3] for fold in range(3)]
    result = np.full((len(tiers), 4), np.nan)
    for held in folds:
        fitting = np.isin(event_of, [event for event in train_events if event not in held])
        testing = np.flatnonzero(np.isin(event_of, held))
        coefficient, _ = rb.fit_classifier(features["joint"][fitting], tiers[fitting], event_of[fitting].tolist(),
            0.1, "event_class")
        result[testing] = rb.predict(features["joint"][testing], coefficient)
    return result


def macro_ap(labels, scores, event_of, events):
    """事件宏平均 AP 与逐事件 AP。"""
    per = dict((event, binary_ranking(labels[event_of == event], scores[event_of == event])["ap"]) for event in events)
    return float(np.mean(list(per.values()))), per


def within_rank(groups, values):
    """组内分位（0–1），组外信息完全抹去。"""
    result = np.empty_like(values)
    for group in np.unique(groups):
        members = np.flatnonzero(groups == group)
        result[members] = (rankdata(values[members]) - 0.5) / len(members)
    return result


def similarity_groups(dino, segment_key, members, threshold):
    """段内 DINO 余弦 ≥threshold 的连通分量编号（只用于诊断上界，不作选片单位）。"""
    group = np.full(len(segment_key), -1)
    offset = 0
    for segment in np.unique(segment_key[members]):
        indices = np.flatnonzero(segment_key == segment)
        components = connected_components(csr_matrix(dino[indices] @ dino[indices].T >= threshold), directed=False)[1]
        group[indices] = components + offset
        offset += components.max() + 1
    return group


def burst_groups(times, fingerprints, event_of, members, gap):
    """同事件按拍摄时间相邻间隔 ≤gap 秒串联的连拍组（只作诊断上界）。"""
    group = np.arange(len(fingerprints))
    for event in np.unique(event_of[members]):
        indices = np.flatnonzero(event_of == event)
        indices = indices[np.argsort([times[fingerprints[index]] for index in indices], kind="stable")]
        for first, second in zip(indices[:-1], indices[1:]):
            if times[fingerprints[second]] - times[fingerprints[first]] <= gap:
                group[second] = group[first]
    return group


def ideal_within(scores, ratings, group):
    """组内按真实星级重排已有分数（同星随机次序固定种子），组间分数集合不变。"""
    rng = np.random.default_rng(7)
    result = scores.copy()
    for value in np.unique(group):
        members = np.flatnonzero(group == value)
        if len(members) > 1:
            order = np.lexsort((rng.random(len(members)), -ratings[members]))
            result[members[order]] = np.sort(scores[members])[::-1]
    return result


def band_accuracy(scores, tiers, threshold, dino, segment_key, members):
    """同段跨门槛对按 DINO 余弦分带的准确率（等分记 0.5）。"""
    totals = dict((band, [0.0, 0]) for band in BANDS)
    for segment in np.unique(segment_key[members]):
        indices = np.flatnonzero(segment_key == segment)
        labels = tiers[indices] >= threshold
        if labels.all() or not labels.any():
            continue
        positive, negative = indices[labels], indices[~labels]
        cosine = dino[positive] @ dino[negative].T
        margin = scores[positive][:, None] - scores[negative][None, :]
        correct = (margin > 0) + 0.5 * (margin == 0)
        for band in BANDS:
            mask = (cosine >= band[0]) & (cosine < band[1])
            totals[band][0] += float(correct[mask].sum())
            totals[band][1] += int(mask.sum())
    return dict((f"{low:.2f}-{high:.2f}", {"accuracy": value / count if count else None, "pairs": count})
        for (low, high), (value, count) in totals.items())


def capture_seconds(database):
    """只读数据集库拍摄时间（秒，含亚秒）。"""
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    result = dict()
    for fingerprint, stamp, subsecond in connection.execute("select fingerprint, capture_time, capture_subsec from photos"):
        fraction = float("0." + str(subsecond)) if subsecond not in (None, "") and str(subsecond).isdigit() else 0.0
        result[fingerprint] = datetime.fromisoformat(stamp.replace(" ", "T")[:19]).timestamp() + fraction
    return result


def main() -> int:
    """计算四类定位诊断并写出 report.json。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("D:/PhotoDB/dataset/photos_dataset.db"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    inputs = argparse.Namespace(photos=ROOT / "audit/out/m3_pairs_v2_20260924b/photos.csv",
        split_json=ROOT / "data/split-v2-20260924.json", features=ROOT / "train/out/recovery_work/corrected_features.npz",
        local=ROOT / "train/out/recovery_work/rescored_lora_v2/scores.csv",
        global_scores=ROOT / "train/out/recovery_work/corrected_rank/clip_scores.csv")
    metadata, fingerprints, features, _ = rb.load_inputs(inputs)
    ratings = np.array([int(metadata[fp]["rating_raw"]) for fp in fingerprints])
    tiers = np.minimum(ratings, 3)
    event_of = np.array([metadata[fp]["event"] for fp in fingerprints])
    segment_key = np.array([metadata[fp]["event"] + "|" + metadata[fp]["seg_id"] for fp in fingerprints])
    members = np.flatnonzero(np.array([metadata[fp]["split"] for fp in fingerprints]) == "train")
    train_events = sorted(set(event_of[members].tolist()),
        key=lambda event: hashlib.sha256(("20260926:" + event).encode()).hexdigest())
    dino = features["dino"] / np.sqrt(384)
    probabilities = baseline_oof(features, tiers, event_of, train_events)
    times = capture_seconds(args.db)
    groupings = {"burst_2s": burst_groups(times, fingerprints, event_of, members, 2.0),
        "similar_0.95": similarity_groups(dino, segment_key, members, 0.95),
        "similar_0.88": similarity_groups(dino, segment_key, members, 0.88)}
    report = {"scope": "train 事件 OOF；诊断不作验收", "thresholds": dict()}
    for threshold in (1, 2, 3):
        labels = tiers >= threshold
        scores = probabilities[:, threshold:].sum(axis=1)
        model, per_model = macro_ap(labels, scores, event_of, train_events)
        segment, per_segment = macro_ap(labels, within_rank(segment_key, scores), event_of, train_events)
        differences = np.array([per_segment[event] - per_model[event] for event in train_events])
        samples = np.random.default_rng(1).choice(differences, size=(10000, len(differences))).mean(axis=1)
        share = []
        for event in train_events:
            indices = np.flatnonzero(event_of == event)
            rows = []
            for values in (labels[indices].astype(float), scores[indices]):
                means = dict((value, values[segment_key[indices] == value].mean()) for value in np.unique(segment_key[indices]))
                between = np.mean([(means[value] - values.mean()) ** 2 for value in segment_key[indices]])
                rows.append(between / values.var() if values.var() > 0 else None)
            share.append(rows)
        entry = {"model_ap": model, "random_ap": float(np.mean([binary_ranking(labels[event_of == event],
                scores[event_of == event])["random_ap"] for event in train_events])),
            "segment_rank_ap": segment, "segment_rank_minus_model": {"mean": float(differences.mean()),
                "event_bootstrap95": np.quantile(samples, (0.025, 0.975)).tolist()},
            "label_variance_between_segments": float(np.mean([row[0] for row in share if row[0] is not None])),
            "score_variance_between_segments": float(np.mean([row[1] for row in share if row[1] is not None])),
            "ideal_within_group_ap": dict((name, macro_ap(labels, ideal_within(scores, ratings, group), event_of,
                train_events)[0]) for name, group in groupings.items()),
            "ideal_group_has_positive_ap": dict((name, macro_ap(labels, scores + 10.0 * np.array([
                labels[group == value].any() for value in group]), event_of, train_events)[0])
                for name, group in groupings.items() if name != "burst_2s"),
            "band_accuracy": band_accuracy(scores, tiers, threshold, dino, segment_key, members)}
        report["thresholds"][str(threshold)] = entry
        print(f">={threshold}: model {model:.4f} random {entry['random_ap']:.4f} segment-rank {segment:.4f} "
            f"ideal {entry['ideal_within_group_ap']}", flush=True)
    zero = np.intersect1d(members, np.flatnonzero(tiers == 0))
    sibling = np.full(len(fingerprints), -1.0)
    for value in np.unique(segment_key[members]):
        indices = np.flatnonzero(segment_key == value)
        similarity = dino[indices] @ dino[indices].T
        np.fill_diagonal(similarity, -1)
        sibling[indices] = np.where((tiers[indices] >= 1)[None, :], similarity, -1).max(axis=1)
    report["zero_star_with_positive_sibling"] = dict((str(cut), float((sibling[zero] >= cut).mean()))
        for cut in (0.88, 0.95, 0.98))
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "report.json", report)
    print(f"0★ with ≥1★ sibling {report['zero_star_with_positive_sibling']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
