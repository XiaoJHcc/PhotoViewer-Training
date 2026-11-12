"""标签还是表征：在 golden 盲评团内对上，比较冻结模型对“盲评星”与“原星”两种真值的准确率。

只读 golden 批1/2/3 读回与 v2 train 事件（单图头三折 OOF、比较器留出边分数均只用原星训练、未见 golden）。
同一批对上报告：原星与盲评的方向一致率（标签可重复性）、各模型对盲评/原星的准确率与事件配对差，
按盲评星差 1／2／≥3 分层。只作诊断，不训练、不调参、不打开 val/test。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "audit"))
import rating_baseline as rb
from evaluation_protocol import write_json
from within_scene_audit import baseline_oof

BATCHES = ("golden_star", "golden_star2", "golden_star3")
BOOTSTRAP_SEED = 20261111


def read_groups(dataset: Path) -> dict:
    """读取三批盲评读回，返回 团号 → [(指纹, 盲评星, 原星)]；团号带批次前缀避免冲突。"""
    groups = defaultdict(list)
    for batch in BATCHES:
        with (dataset / batch / "golden_star_readback.tsv").open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                groups[f"{batch}:{row['gid']}"].append((row["fingerprint"], int(row["user_star"]), int(row["orig_rating"])))
    return groups


def pair_rows(groups: dict, position: dict, event_of: np.ndarray, allowed: set) -> list[dict]:
    """同团盲评星不同的有序对（胜者在前），只留指纹在库且属允许事件的对。"""
    rows = []
    for gid, members in groups.items():
        for first in range(len(members)):
            for second in range(first + 1, len(members)):
                (fa, ua, oa), (fb, ub, ob) = members[first], members[second]
                if ua == ub or fa not in position or fb not in position:
                    continue
                if event_of[position[fa]] not in allowed or event_of[position[fa]] != event_of[position[fb]]:
                    continue
                winner, loser = ((fa, oa), (fb, ob)) if ua > ub else ((fb, ob), (fa, oa))
                rows.append({"group": gid, "winner": position[winner[0]], "loser": position[loser[0]],
                    "gap": abs(ua - ub), "orig_winner": winner[1], "orig_loser": loser[1],
                    "event": event_of[position[fa]]})
    return rows


def edge_lookup(path: Path) -> dict:
    """读取比较器留出边分数（三种子平均），返回 (i,j)→f(i,j)，并补反向取负。"""
    cached = np.load(path)
    table = dict()
    for arm in ("patch", "cls", "shuffle"):
        scores = cached[arm].mean(axis=0)
        lookup = dict()
        for left, right, value in zip(cached["left"].tolist(), cached["right"].tolist(), scores.tolist()):
            lookup[(left, right)] = value
            lookup[(right, left)] = -value
        table[arm] = lookup
    return table


def correct(margin: float) -> float:
    """胜者分数更高记 1，相等记 0.5。"""
    return 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)


def summarize(rows: list[dict], key: str) -> dict:
    """对给定逐对正确列做合并与事件宏平均。"""
    usable = [row for row in rows if row.get(key) is not None]
    if not usable:
        return {"pairs": 0}
    by_event = defaultdict(list)
    for row in usable:
        by_event[row["event"]].append(row[key])
    return {"pairs": len(usable), "events": len(by_event), "pooled": float(np.mean([row[key] for row in usable])),
        "event_macro": float(np.mean([np.mean(values) for values in by_event.values()])),
        "by_event": dict((event, float(np.mean(values))) for event, values in sorted(by_event.items()))}


def paired(rows: list[dict], left: str, right: str) -> dict:
    """同一批对上两列正确率的事件配对差 bootstrap。"""
    by_event = defaultdict(list)
    for row in rows:
        if row.get(left) is not None and row.get(right) is not None:
            by_event[row["event"]].append(row[left] - row[right])
    if len(by_event) < 2:
        return {"events": len(by_event)}
    differences = np.array([np.mean(values) for values in by_event.values()])
    samples = np.random.default_rng(BOOTSTRAP_SEED).choice(differences, size=(10000, len(differences))).mean(axis=1)
    return {"mean": float(differences.mean()), "event_bootstrap95": np.quantile(samples, (0.025, 0.975)).tolist(),
        "positive_events": int((differences > 0).sum()), "events": len(differences),
        "pairs": int(sum(len(values) for values in by_event.values()))}


def main() -> int:
    """汇总标签一致率与模型双真值准确率，写出 report.json。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("D:/PhotoDB/dataset"))
    parser.add_argument("--edges", type=Path, default=ROOT / "train/out/scene_comparator_20261111/edge_scores.npz")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    inputs = argparse.Namespace(photos=ROOT / "audit/out/m3_pairs_v2_20260924b/photos.csv",
        split_json=ROOT / "data/split-v2-20260924.json", features=ROOT / "train/out/recovery_work/corrected_features.npz",
        local=ROOT / "train/out/recovery_work/rescored_lora_v2/scores.csv",
        global_scores=ROOT / "train/out/recovery_work/corrected_rank/clip_scores.csv")
    metadata, fingerprints, features, _ = rb.load_inputs(inputs)
    tiers = np.array([min(int(metadata[fp]["rating_raw"]), 3) for fp in fingerprints])
    event_of = np.array([metadata[fp]["event"] for fp in fingerprints])
    split_of = np.array([metadata[fp]["split"] for fp in fingerprints])
    train_events = sorted(set(event_of[split_of == "train"].tolist()),
        key=lambda event: hashlib.sha256(("20260926:" + event).encode()).hexdigest())
    position = dict((fingerprint, index) for index, fingerprint in enumerate(fingerprints))
    probabilities = baseline_oof(features, tiers, event_of, train_events)
    expected = probabilities @ np.arange(4.0)
    edges = edge_lookup(args.edges)
    groups = read_groups(args.dataset)
    all_pairs = pair_rows(groups, position, event_of, set(event_of.tolist()))
    rows = pair_rows(groups, position, event_of, set(train_events))
    for row in rows:
        winner, loser = row["winner"], row["loser"]
        if int(metadata[fingerprints[winner]]["rating_raw"]) != row["orig_winner"] or \
                int(metadata[fingerprints[loser]]["rating_raw"]) != row["orig_loser"]:
            row["orig_snapshot_mismatch"] = True
        orig = row["orig_winner"] - row["orig_loser"]
        row["orig_agrees"] = correct(orig)
        row["single_vs_golden"] = correct(expected[winner] - expected[loser])
        row["single_vs_orig"] = None if orig == 0 else correct((expected[winner] - expected[loser]) * np.sign(orig))
        for arm, lookup in edges.items():
            value = lookup.get((winner, loser))
            row[f"{arm}_vs_golden"] = None if value is None else correct(value)
            row[f"{arm}_vs_orig"] = None if value is None or orig == 0 else correct(value * np.sign(orig))
    report = {"scope": "v2 train 事件 golden 同团盲评对；模型只用原星训练、三折留出", "groups": len(groups),
        "pairs_all_splits": len(all_pairs), "pairs_train_events": len(rows),
        "orig_snapshot_mismatches": sum(1 for row in rows if row.get("orig_snapshot_mismatch")), "strata": dict()}
    columns = ["orig_agrees", "single_vs_golden", "single_vs_orig"] + \
        [f"{arm}_vs_{truth}" for arm in edges for truth in ("golden", "orig")]
    for name, subset in (("all", rows), ("gap1", [r for r in rows if r["gap"] == 1]),
            ("gap2", [r for r in rows if r["gap"] == 2]), ("gap3plus", [r for r in rows if r["gap"] >= 3])):
        entry = dict((column, summarize(subset, column)) for column in columns)
        entry["orig_ties"] = float(np.mean([r["orig_winner"] == r["orig_loser"] for r in subset])) if subset else None
        entry["single_golden_minus_orig"] = paired(subset, "single_vs_golden", "single_vs_orig")
        entry["patch_golden_minus_orig"] = paired(subset, "patch_vs_golden", "patch_vs_orig")
        report["strata"][name] = entry
        print(f"{name}: pairs {len(subset)} orig-agree {entry['orig_agrees'].get('pooled', float('nan')):.3f} "
            f"(orig ties {entry['orig_ties'] if entry['orig_ties'] is None else round(entry['orig_ties'], 3)}) | "
            f"single golden {entry['single_vs_golden'].get('pooled', float('nan')):.3f} orig "
            f"{entry['single_vs_orig'].get('pooled', float('nan')):.3f} | patch golden "
            f"{entry['patch_vs_golden'].get('pooled', float('nan')):.3f} (n {entry['patch_vs_golden'].get('pairs', 0)}) "
            f"cls golden {entry['cls_vs_golden'].get('pooled', float('nan')):.3f}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "report.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
