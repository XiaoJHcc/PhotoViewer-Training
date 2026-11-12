"""零样本外部模型的团内取舍判读与（闸门通过后）融合：登记见 EXECUTION-LOG 2026-11-11。

① 同段 cos≥0.88 跨档对（与 scene_comparator Stage A 同一组对）事件宏平均准确率；② v2 train 事件 golden 盲评异星对
准确率；③ 完整事件 AP≥1/≥2/≥3。闸门：某模型 ①≥0.58 且事件下界 >0.5、②≥0.60。过闸模型才做融合：分数按事件内
标准化后以 γ·z·档序倾斜单图四档概率，γ 只在拟合折事件选，报 train OOF；--open-val 仅在融合过闸后评一次 val
（γ 用全 train 事件按同规则重选，单图头取正式类别均衡臂全 train 拟合分数，不据 val 调任何量）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "audit"))
import rating_baseline as rb
from evaluation_protocol import binary_ranking, digest, write_json
from golden_label_check import pair_rows, read_groups
from scene_comparator import (GAMMAS, bootstrap, compare_metrics, context_signal, headline, measure_rows,
    segments_of, selection_key, similar_edges, tilt)
from within_scene_audit import baseline_oof

GATE = dict(pair_accuracy=0.58, golden_accuracy=0.60)
FUSION = dict(ap3_lower=0.0, balanced_tolerance=0.0)


def pair_accuracy(score: np.ndarray, left: np.ndarray, right: np.ndarray, tiers: np.ndarray, events: np.ndarray) -> dict:
    """跨档对按档序比较的事件宏平均/合并准确率（等分 0.5）。"""
    cross = tiers[left] != tiers[right]
    margin = (score[left] - score[right])[cross] * np.sign(tiers[left] - tiers[right])[cross]
    correct = np.where(margin == 0, 0.5, (margin > 0).astype(float))
    owner = events[left[cross]]
    by_event = dict((event, float(correct[owner == event].mean())) for event in sorted(set(owner.tolist())))
    return {"pairs": int(cross.sum()), "pooled": float(correct.mean()),
        "event_macro": float(np.mean(list(by_event.values()))), "by_event": by_event}


def golden_accuracy(score: np.ndarray, rows: list[dict]) -> dict:
    """golden 盲评胜者分数更高的比例（事件宏平均、合并、按星差）。"""
    result = dict()
    for name, subset in (("all", rows), ("gap1", [r for r in rows if r["gap"] == 1]),
            ("gap2", [r for r in rows if r["gap"] == 2]), ("gap3plus", [r for r in rows if r["gap"] >= 3])):
        values = dict()
        for row in subset:
            margin = score[row["winner"]] - score[row["loser"]]
            values.setdefault(row["event"], []).append(1.0 if margin > 0 else (0.5 if margin == 0 else 0.0))
        result[name] = {"pairs": len(subset), "pooled": float(np.mean([v for vs in values.values() for v in vs])),
            "event_macro": float(np.mean([np.mean(vs) for vs in values.values()])),
            "by_event": dict((event, float(np.mean(vs))) for event, vs in sorted(values.items()))}
    return result


def event_ap(score: np.ndarray, tiers: np.ndarray, event_of: np.ndarray, events: list[str]) -> dict:
    """三条边界的事件宏平均 AP 与随机 AP。"""
    result = dict()
    for threshold in (1, 2, 3):
        rows = [binary_ranking(tiers[event_of == event] >= threshold, score[event_of == event]) for event in events]
        result[f"ap{threshold}"] = float(np.mean([row["ap"] for row in rows]))
        result[f"random_ap{threshold}"] = float(np.mean([row["random_ap"] for row in rows]))
    return result


def event_zscore(values: np.ndarray, event_of: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """在给定照片内按事件标准化（只用无标签分数），其余位置为 0。"""
    result = np.zeros(len(values))
    for event in np.unique(event_of[indices]):
        members = indices[event_of[indices] == event]
        spread = values[members].std()
        result[members] = (values[members] - values[members].mean()) / (spread if spread > 0 else 1.0)
    return result


def evaluate_val(args, metadata, fingerprints, event_of, split_of, probabilities, external, names, train_indices,
        train_events) -> dict:
    """融合过闸后只评一次 val：γ 用全部 train 事件按同一规则重选，单图头取正式类别均衡臂全 train 拟合分数。"""
    formal = ROOT / "train/out/rating_baseline_20261014_balanced/joint_scores.csv"
    position = dict((fingerprint, index) for index, fingerprint in enumerate(fingerprints))
    val_indices = np.flatnonzero(split_of == "val")
    val_events = sorted(set(event_of[val_indices].tolist()))
    base = np.full((len(fingerprints), 4), np.nan)
    with formal.open(encoding="utf-8") as stream:
        header = stream.readline().strip().split(",")
        columns = [header.index(name) for name in ("p0", "p1", "p2", "p_ge3")]
        for line in stream:
            values = line.strip().split(",")
            base[position[values[0]]] = [float(values[column]) for column in columns]
    with np.load(args.val_scores, allow_pickle=False) as cached:
        val_external = dict((name, np.full(len(fingerprints), np.nan)) for name in names)
        for column, fingerprint in enumerate(cached["fingerprints"].tolist()):
            for name in names:
                val_external[name][position[fingerprint]] = cached[name][column]
    baseline_metrics = measure_rows(metadata, fingerprints, base, val_indices, val_events)
    report = {"baseline": headline(baseline_metrics), "models": dict()}
    for name in names:
        if np.isnan(val_external[name][val_indices]).any():
            raise ValueError(f"val 零样本分数不完整: {name}")
        train_signal = event_zscore(external[name], event_of, train_indices)
        grid = []
        for gamma in GAMMAS:
            metrics = measure_rows(metadata, fingerprints, tilt(probabilities, train_signal, gamma), train_indices, train_events)
            grid.append((selection_key(metrics, gamma), gamma))
        gamma = max(grid)[1]
        signal = event_zscore(val_external[name], event_of, val_indices)
        fused = base.copy()
        fused[val_indices] = tilt(base[val_indices], signal[val_indices], gamma)
        fused_metrics = measure_rows(metadata, fingerprints, fused, val_indices, val_events)
        report["models"][name] = {"gamma": gamma, "fused": headline(fused_metrics),
            "differences": compare_metrics(fused_metrics, baseline_metrics)}
        print(f"VAL {name} gamma {gamma} AP3 {headline(fused_metrics)['ap3']:.4f} vs {headline(baseline_metrics)['ap3']:.4f}", flush=True)
    return report


def main() -> int:
    """按登记顺序判读零样本分数，过闸才融合与评 val。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--val-scores", type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("D:/PhotoDB/dataset"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--open-val", action="store_true")
    args = parser.parse_args()
    inputs = argparse.Namespace(photos=ROOT / "audit/out/m3_pairs_v2_20260924b/photos.csv",
        split_json=ROOT / "data/split-v2-20260924.json", features=ROOT / "train/out/recovery_work/corrected_features.npz",
        local=ROOT / "train/out/recovery_work/rescored_lora_v2/scores.csv",
        global_scores=ROOT / "train/out/recovery_work/corrected_rank/clip_scores.csv")
    metadata, fingerprints, features, _ = rb.load_inputs(inputs)
    tiers = np.array([min(int(metadata[fp]["rating_raw"]), 3) for fp in fingerprints])
    event_of = np.array([metadata[fp]["event"] for fp in fingerprints])
    split_of = np.array([metadata[fp]["split"] for fp in fingerprints])
    segment_key = np.array([metadata[fp]["event"] + "|" + metadata[fp]["seg_id"] for fp in fingerprints])
    position = dict((fingerprint, index) for index, fingerprint in enumerate(fingerprints))
    train_indices = np.flatnonzero(split_of == "train")
    train_events = sorted(set(event_of[train_indices].tolist()),
        key=lambda event: hashlib.sha256(("20260926:" + event).encode()).hexdigest())
    folds = [train_events[fold::3] for fold in range(3)]
    dino = features["dino"] / np.sqrt(384)
    with np.load(args.scores, allow_pickle=False) as cached:
        names = [name for name in cached.files if name != "fingerprints"]
        external = dict((name, np.full(len(fingerprints), np.nan)) for name in names)
        for column, fingerprint in enumerate(cached["fingerprints"].tolist()):
            for name in names:
                external[name][position[fingerprint]] = cached[name][column]
    if any(np.isnan(external[name][train_indices]).any() for name in names):
        raise ValueError("train 照片零样本分数不完整")
    args.out.mkdir(parents=True, exist_ok=False)
    plan = {"registration": "EXECUTION-LOG 2026-11-11 外部预训练美学/画质零样本读团内取舍", "gate": GATE,
        "fusion": FUSION, "gammas": GAMMAS, "models": names,
        "input_sha256": dict((str(path), digest(path)) for path in (args.scores, inputs.photos, inputs.features, Path(__file__)))}
    write_json(args.out / "plan.json", plan)
    probabilities = baseline_oof(features, tiers, event_of, train_events)
    expected = probabilities @ np.arange(4.0)
    left, right, _ = similar_edges(segments_of(train_indices, segment_key), dino)
    golden = pair_rows(read_groups(args.dataset), position, event_of, set(train_events))
    reference = {"single_image_joint": expected}
    result = {"plan": plan, "edges": int((tiers[left] != tiers[right]).sum()), "golden_pairs": len(golden), "models": dict()}
    for name, score in list(reference.items()) + list(external.items()):
        entry = {"within_scene": pair_accuracy(score, left, right, tiers, event_of), "golden": golden_accuracy(score, golden),
            "event_ap": event_ap(score, tiers, event_of, train_events)}
        macro = np.array(list(entry["within_scene"]["by_event"].values()))
        samples = np.random.default_rng(20261111).choice(macro, size=(10000, len(macro))).mean(axis=1)
        entry["within_scene"]["event_bootstrap95"] = np.quantile(samples, (0.025, 0.975)).tolist()
        entry["passes_gate"] = bool(name != "single_image_joint"
            and entry["within_scene"]["event_macro"] >= GATE["pair_accuracy"]
            and entry["within_scene"]["event_bootstrap95"][0] > 0.5
            and entry["golden"]["all"]["event_macro"] >= GATE["golden_accuracy"])
        result["models"][name] = entry
        golden_all, ws = entry["golden"]["all"], entry["within_scene"]
        print(f"{name:20s} within-scene {ws['event_macro']:.4f} [{ws['event_bootstrap95'][0]:.3f},{ws['event_bootstrap95'][1]:.3f}] "
            f"golden {golden_all['event_macro']:.4f} (gap1 {entry['golden']['gap1']['event_macro']:.3f} "
            f"gap3+ {entry['golden']['gap3plus']['event_macro']:.3f}) AP1 {entry['event_ap']['ap1']:.4f} "
            f"AP3 {entry['event_ap']['ap3']:.4f} gate {entry['passes_gate']}", flush=True)
    passed = [name for name in names if result["models"][name]["passes_gate"]]
    result["gate_passed_models"] = passed
    write_json(args.out / "result.json", result)
    if not passed:
        print("没有模型过闸，按登记停止，不融合、不评 val", flush=True)
        return 0

    baseline_metrics = measure_rows(metadata, fingerprints, probabilities, train_indices, train_events)
    result["fusion"] = dict()
    for name in passed:
        signal = event_zscore(external[name], event_of, train_indices)
        fused = probabilities.copy()
        chosen = []
        for held in folds:
            fitting = [event for event in train_events if event not in held]
            fitting_indices = np.flatnonzero(np.isin(event_of, fitting))
            grid = []
            for gamma in GAMMAS:
                metrics = measure_rows(metadata, fingerprints, tilt(probabilities, signal, gamma), fitting_indices, fitting)
                grid.append((selection_key(metrics, gamma), gamma))
            gamma = max(grid)[1]
            chosen.append(gamma)
            held_indices = np.flatnonzero(np.isin(event_of, held))
            fused[held_indices] = tilt(probabilities[held_indices], signal[held_indices], gamma)
        fused_metrics = measure_rows(metadata, fingerprints, fused, train_indices, train_events)
        differences = compare_metrics(fused_metrics, baseline_metrics)
        ok = (differences["ap3"]["event_bootstrap95"][0] > FUSION["ap3_lower"]
            and differences["balanced_accuracy"]["mean"] >= FUSION["balanced_tolerance"])
        result["fusion"][name] = {"gammas": chosen, "baseline": headline(baseline_metrics), "fused": headline(fused_metrics),
            "differences": differences, "passed": bool(ok)}
        print(f"FUSION {name} gammas {chosen} AP3 {headline(fused_metrics)['ap3']:.4f} vs {headline(baseline_metrics)['ap3']:.4f} "
            f"diff {differences['ap3']} balanced {differences['balanced_accuracy']['mean']:+.4f} passed {ok}", flush=True)
    write_json(args.out / "result.json", result)
    fusion_passed = [name for name, entry in result["fusion"].items() if entry["passed"]]
    if not fusion_passed or not args.open_val or args.val_scores is None:
        print("融合未过闸或未授权 val，停止", flush=True)
        return 0
    val = evaluate_val(args, metadata, fingerprints, event_of, split_of, probabilities, external, fusion_passed,
        train_indices, train_events)
    result["val"] = val
    write_json(args.out / "result.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
