"""新事件划分评估：沿用统一协议，逐事件报告直接横评、局部盲评和固定预算代理。"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from evaluation_protocol import (abs_metrics, budget_metrics, digest, load_scores,
                                 local_metrics, mean_or_none, pair_metrics, read_rows,
                                 require_coverage, write_json)
from protocol_eval import joined_labels


def summarize_budget(rows: list[dict]) -> dict:
    """按事件等权汇总预算内旧星好片召回，并给出随机选代表的解析期望。"""
    result = {}
    for fraction in sorted({row["budget_fraction"] for row in rows}):
        subset = [row for row in rows if row["budget_fraction"] == fraction]
        result[str(fraction)] = {
            "events": len(subset),
            "actual_kept_fraction_macro": mean_or_none(row["kept"] / row["photos"] for row in subset),
            **{f"{kind}_recall_macro": mean_or_none(
                row[f"positive_{kind}_hits"] / row[f"positive_{plural}"]
                for row in subset if row[f"positive_{plural}"])
               for kind, plural in (("photo", "photos"), ("group", "groups"), ("singleton", "singletons"))},
            "random_group_recall_macro": mean_or_none(
                min(row["budget"], row["candidate_groups"]) / row["candidate_groups"]
                for row in subset if row["positive_groups"]),
            "good_representative_group_recall_macro": mean_or_none(
                row["positive_photo_hits"] / row["positive_groups"]
                for row in subset if row["positive_groups"]),
            "one_per_group_photo_recall_ceiling_macro": mean_or_none(
                min(row["budget"], row["positive_groups"]) / row["positive_photos"]
                for row in subset if row["positive_photos"]),
        }
    return result


def evaluate_split(metadata: dict, scores: dict, split: str, m3: Path,
                   abs_dir: Path, labels: list[dict]) -> dict:
    """严格检查留出覆盖，分来源评价；旧星仅作代理，不当成人类独立真值。"""
    subset = {fingerprint: row for fingerprint, row in metadata.items() if row["split"] == split}
    require_coverage(scores, subset)
    types = defaultdict(list)
    for row in read_rows(m3 / f"pairs_{split}.csv"):
        left, right = row["fp_i"], row["fp_j"]
        if left not in subset or right not in subset or row["ptype"] == "derived":
            raise ValueError("留出对越界或含教师派生标签")
        difference = int(subset[left]["rating_raw"]) - int(subset[right]["rating_raw"])
        if not difference:
            raise ValueError("旧星配对没有方向")
        winner, loser = (left, right) if difference > 0 else (right, left)
        types[row["ptype"]].append({"winner": winner, "loser": loser, "event": subset[left]["event"]})
    absolute = read_rows(abs_dir / f"pairs_{split}.csv")
    if any(row[key] not in subset for row in absolute for key in ("fp_i", "fp_j")):
        raise ValueError("横评对越过当前划分")
    events = sorted({row["event"] for row in subset.values()})
    local = [dict(row, event=metadata[row["fingerprint"]]["event"])
             for row in labels if row["fingerprint"] in subset]
    budget = budget_metrics(metadata, scores, events)
    return {"photos": len(subset), "events": events,
            "old_pairs": {kind: pair_metrics(rows, scores) for kind, rows in types.items()},
            "absolute": {source: abs_metrics([row for row in absolute if row["source"] == source], metadata, scores)
                         for source in sorted({row["source"] for row in absolute})},
            "absolute_all": abs_metrics(absolute, metadata, scores),
            "absolute_d2": abs_metrics([row for row in absolute if int(row["dstar"]) >= 2], metadata, scores),
            "local_blind": local_metrics(local, scores),
            "budget_proxy": budget, "budget_summary": summarize_budget(budget)}


def main() -> int:
    """读取显式数据路径与多个模型分数，写出带哈希的结果，拒绝覆盖历史输出。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m3", type=Path, required=True)
    parser.add_argument("--abs-pairs", type=Path, required=True)
    parser.add_argument("--labels-snapshot", type=Path, required=True)
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--splits", nargs="+", choices=("val", "test"), default=["val"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    metadata_rows = read_rows(args.m3 / "photos.csv")
    metadata = {row["fingerprint"]: row for row in metadata_rows}
    if len(metadata) != len(metadata_rows):
        raise ValueError("照片清单有重复指纹")
    inputs = [args.m3 / "photos.csv"]
    labels = []
    for batch in (2, 3):
        key = args.labels_snapshot / f"batch{batch}_key.csv"
        ratings = args.labels_snapshot / f"batch{batch}.tsv"
        labels.extend(joined_labels(key, ratings))
        inputs.extend((key, ratings))
    groups = defaultdict(list)
    for row in labels:
        if row["fingerprint"] not in metadata:
            raise ValueError("局部盲评照片缺少元数据")
        groups[row["gid"]].append(row)
    for members in groups.values():
        if len({metadata[row["fingerprint"]]["split"] for row in members}) != 1:
            raise ValueError("局部盲评组跨越划分")
    for split in args.splits:
        inputs.extend((args.m3 / f"pairs_{split}.csv", args.abs_pairs / f"pairs_{split}.csv"))
    inputs.append(Path(__file__))
    args.out.mkdir(parents=True, exist_ok=False)
    results = {}
    for index, path in enumerate(args.scores):
        scores = load_scores(path)
        result = {split: evaluate_split(metadata, scores, split, args.m3, args.abs_pairs, labels)
                  for split in args.splits}
        write_json(args.out / f"model_{index}.json", result)
        results[str(path)] = f"model_{index}.json"
        print(f"评估完成 {path}", flush=True)
    write_json(args.out / "manifest.json", {"results": results,
               "sha256": {str(path): digest(path) for path in [*inputs, *args.scores]},
               "limits": "v2 开发留出；横评跨事件宏平均单位为事件对；预算为旧星>=4代理；局部来自批2/3盲评；没有新增人工标签。"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
