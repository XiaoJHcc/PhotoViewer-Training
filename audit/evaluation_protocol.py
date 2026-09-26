"""统一评估协议：严格覆盖、等分处理、局部保留及固定预算下的旧标签代理。"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

VERSION = "culling-evaluation-v1"
SCORE_TIE_EPSILON = 1e-8


def read_rows(path: Path, delimiter: str = ",") -> list[dict]:
    """按给定分隔符读取文件，返回行字典；读取失败直接报错。"""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def digest(path: Path) -> str:
    """分块读取文件并返回 SHA256，避免大型文件一次占满内存。"""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value) -> None:
    """将可序列化结果写成 UTF-8 JSON；拒绝 NaN 避免无效指标混入报告。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_scores(path: Path) -> dict[str, float]:
    """读取完整且唯一的有限分数；重复指纹、NaN 或无穷值立即报错。"""
    result = {}
    for row in read_rows(path):
        fingerprint = row["fingerprint"]
        value = float(row["score"])
        if fingerprint in result or not math.isfinite(value):
            raise ValueError(f"重复指纹或非有限分数: {path}: {fingerprint}")
        result[fingerprint] = value
    return result


def require_coverage(scores: dict, fingerprints) -> None:
    """验证指定样本全部有分数；不允许模型通过缺失困难样本改变分母。"""
    missing = set(fingerprints) - scores.keys()
    if missing:
        raise ValueError(f"缺少 {len(missing)} 张分数，示例 {sorted(missing)[:3]}")
    if any(not math.isfinite(scores[fingerprint]) for fingerprint in fingerprints):
        raise ValueError("评估包含非有限分数")


def mean_or_none(values) -> float | None:
    """返回非空序列平均值，空分母返回 None，禁止伪造满分或零分。"""
    values = list(values)
    return float(np.mean(values)) if values else None


def pair_metrics(pairs: list[dict], scores: dict) -> dict:
    """评估 winner/loser 对；正确计一、模型等分计半、错误计零并分别报告。"""
    require_coverage(scores, {row[field] for row in pairs for field in ("winner", "loser")})
    counts = Counter()
    by_event = defaultdict(list)
    for row in pairs:
        difference = scores[row["winner"]] - scores[row["loser"]]
        outcome = "tie" if abs(difference) <= SCORE_TIE_EPSILON else "correct" if difference > 0 else "wrong"
        counts[outcome] += 1
        by_event[row["event"]].append({"correct": 1.0, "tie": 0.5, "wrong": 0.0}[outcome])
    total = len(pairs)
    return {"pairs": total, "photos": len({row[field] for row in pairs for field in ("winner", "loser")}),
            "correct": counts["correct"], "wrong": counts["wrong"], "ties": counts["tie"],
            "accuracy": (counts["correct"] + 0.5 * counts["tie"]) / total if total else None,
            "event_macro": mean_or_none(mean_or_none(values) for values in by_event.values()),
            "by_event": {event: {"pairs": len(values), "accuracy": mean_or_none(values)}
                         for event, values in sorted(by_event.items())}}


def selection_probability(scores: dict, acceptable: set[str], count: int) -> float:
    """计算前 count 张至少含一张认可照片的概率，边界等分按均匀随机取舍。"""
    count = min(count, len(scores))
    if count <= 0 or not acceptable:
        return 0.0
    boundary = sorted(scores.values(), reverse=True)[count - 1]
    certain = {fingerprint for fingerprint, value in scores.items() if value > boundary + SCORE_TIE_EPSILON}
    tied = {fingerprint for fingerprint, value in scores.items() if abs(value - boundary) <= SCORE_TIE_EPSILON}
    if certain & acceptable:
        return 1.0
    remaining = count - len(certain)
    bad = len(tied - acceptable)
    return 1.0 - (math.comb(bad, remaining) / math.comb(len(tied), remaining) if bad >= remaining else 0.0)


def local_metrics(rows: list[dict], scores: dict, contextual: bool = False) -> dict:
    """评估标注组的最佳集合保留概率；tie 单列，全零组只统计不臆造认可集合。"""
    require_coverage(scores, {(row["gid"], row["fingerprint"]) if contextual else row["fingerprint"] for row in rows})
    groups = defaultdict(list)
    for row in rows:
        groups[row["gid"]].append(row)
    details = []
    all_zero = 0
    for group_id, members in sorted(groups.items()):
        stars = {row["fingerprint"]: int(row["user_star"]) for row in members}
        if len(stars) != len(members) or len(members) < 2:
            raise ValueError(f"非法标注组: {group_id}")
        highest = max(stars.values())
        if highest == 0:
            all_zero += 1
            continue
        best = {fingerprint for fingerprint, rating in stars.items() if rating == highest}
        second = sorted(stars.values(), reverse=True)[1]
        acceptable = {fingerprint for fingerprint, rating in stars.items() if rating >= second}
        group_scores = {fingerprint: scores[(group_id, fingerprint) if contextual else fingerprint] for fingerprint in stars}
        details.append({"gid": group_id, "event": members[0]["event"], "size": len(members),
                        "unique_best": len(best) == 1,
                        "best1": selection_probability(group_scores, best, 1),
                        "best2": selection_probability(group_scores, best, 2),
                        "accepted_top2": selection_probability(group_scores, acceptable, 1),
                        "random_best1": len(best) / len(stars),
                        "random_best2": selection_probability(dict.fromkeys(stars, 0.0), best, 2),
                        "random_accepted_top2": len(acceptable) / len(stars)})

    def summarize(subset):
        """汇总指定组集合的保留率、平均张数和逐事件平均值。"""
        events = sorted({row["event"] for row in subset})
        return {"groups": len(subset),
                **{field: mean_or_none(row[field] for row in subset)
                   for field in ("best1", "best2", "accepted_top2", "random_best1", "random_best2", "random_accepted_top2")},
                "mean_kept2": mean_or_none(min(2, row["size"]) for row in subset),
                "best1_event_macro": mean_or_none(mean_or_none(row["best1"] for row in subset if row["event"] == event) for event in events)}

    unique = [row for row in details if row["unique_best"]]
    membership = {group_id: tuple(sorted(row["fingerprint"] for row in members))
                  for group_id, members in groups.items()}
    repeats = defaultdict(list)
    for row in unique:
        repeats[membership[row["gid"]]].append(row)
    set_balanced = {field: mean_or_none(mean_or_none(row[field] for row in repeated) for repeated in repeats.values())
                    for field in ("best1", "best2", "accepted_top2", "random_best1", "random_best2")}
    set_balanced["distinct_member_sets"] = len(repeats)
    return {"input_groups": len(groups), "unique_photos": len({row['fingerprint'] for row in rows}),
            "repeated_photo_slots": len(rows) - len({row['fingerprint'] for row in rows}), "all_zero_ambiguous": all_zero,
            "unique_best": summarize(unique), "tied_best": summarize([row for row in details if not row["unique_best"]]),
            "member_set_balanced": set_balanced,
            "all_nonzero": summarize(details),
            "by_event": {event: summarize([row for row in unique if row["event"] == event])
                         for event in sorted({row["event"] for row in unique})}, "details": details}


def abs_metrics(rows: list[dict], metadata: dict, scores: dict) -> dict:
    """将直接盲评对按同事件/跨事件拆报，返回对数、独立照片数与等分正确率。"""
    converted = []
    for row in rows:
        left_event = metadata[row["fp_i"]]["event"]
        right_event = metadata[row["fp_j"]]["event"]
        converted.append({"winner": row["fp_i"], "loser": row["fp_j"],
                          "event": json.dumps(sorted([left_event, right_event]), ensure_ascii=False),
                          "cross": left_event != right_event})
    return {domain: pair_metrics([row for row in converted if row["cross"] == cross], scores)
            for domain, cross in (("same_event", False), ("cross_event", True))}


def budget_metrics(metadata: dict, scores: dict, events: list[str], budgets=(0.05, 0.125, 0.2, 0.3),
                   representative_scores: dict | None = None, group_max: bool = False) -> list[dict]:
    """用指定局部分数选代表、全局分数筛选；缺省共用分数，旧星只评价，不参与选择。"""
    local_scores = scores if representative_scores is None else representative_scores
    results = []
    for event in sorted(events):
        members = {fingerprint: row for fingerprint, row in metadata.items() if row["event"] == event}
        require_coverage(scores, members)
        require_coverage(local_scores, members)
        if not members:
            raise ValueError(f"事件无照片: {event}")
        groups = defaultdict(list)
        for fingerprint, row in members.items():
            groups[row["cluster_id"]].append(fingerprint)
        best_by_group = {}
        for group_id, fingerprints in groups.items():
            best_by_group[group_id] = min(fingerprints, key=lambda fingerprint: (-local_scores[fingerprint], hashlib.sha256(fingerprint.encode()).hexdigest()))
        ranking = {best_by_group[group_id]: max(scores[fp] for fp in fingerprints)
                   if group_max else scores[best_by_group[group_id]] for group_id, fingerprints in groups.items()}
        representatives = sorted(best_by_group.values(), key=lambda fingerprint: (-ranking[fingerprint], hashlib.sha256(fingerprint.encode()).hexdigest()), reverse=False)
        positives = {fingerprint for fingerprint, row in members.items() if int(row["rating_raw"]) >= 4}
        positive_groups = {members[fingerprint]["cluster_id"] for fingerprint in positives}
        positive_singletons = {fingerprint for fingerprint in positives if len(groups[members[fingerprint]["cluster_id"]]) == 1}
        for fraction in budgets:
            budget = math.ceil(len(members) * fraction)
            selected = set(representatives[:budget])
            selected_groups = {members[fingerprint]["cluster_id"] for fingerprint in selected}
            results.append({"event": event, "budget_fraction": fraction, "photos": len(members),
                            "budget": budget, "kept": len(selected), "candidate_groups": len(groups),
                            "positive_photos": len(positives), "positive_photo_hits": len(selected & positives),
                            "positive_groups": len(positive_groups), "positive_group_hits": len(selected_groups & positive_groups),
                            "positive_singletons": len(positive_singletons), "positive_singleton_hits": len(selected & positive_singletons),
                            "score_ties": len(members) - len({scores[fingerprint] for fingerprint in members}),
                            "label_scope": "旧星>=4代理，非独立全局真值；同分以指纹哈希确定取舍"})
    return results


def verify_snapshot(path: Path) -> dict:
    """校验快照内每个输入哈希和协议版本，返回清单；任何漂移都拒绝继续。"""
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest["protocol"] != VERSION:
        raise ValueError("评估协议版本不匹配")
    for filename, item in manifest["files"].items():
        if digest(path / filename) != item["sha256"]:
            raise ValueError(f"快照内容已漂移: {filename}")
    return manifest
