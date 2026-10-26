"""统一评估：完整事件排序、四档评分、严格覆盖及历史局部考卷回归。"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

VERSION = "culling-evaluation-v1"
RATING_VERSION = "photo-rating-v2-20261014"
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


def legacy_cluster_budget_metrics(metadata: dict, scores: dict, events: list[str], budgets=(0.05, 0.125, 0.2, 0.3),
                                  representative_scores: dict | None = None, group_max: bool = False) -> list[dict]:
    """复算历史 v1 每团一张/≥4★ 口径，仅供历史回归诊断，不用于当期训练选择。"""
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


def binary_ranking(labels: np.ndarray, values: np.ndarray) -> dict:
    """返回二元排序 AP/AUC、均匀随机 AP 的解析期望与上限；等分区内均匀随机取序。"""
    labels = np.asarray(labels, dtype=bool)
    values = np.asarray(values, dtype=np.float64)
    if labels.ndim != 1 or values.shape != labels.shape or not len(labels) or not np.isfinite(values).all():
        raise ValueError("非法排序标签或分数")
    count, positives = len(labels), int(labels.sum())
    prevalence = positives / count
    if not positives:
        return {"positives": 0, "prevalence": 0.0, "ap": None, "random_ap": None,
                "ap_ceiling": None, "ap_closure": None, "auc": None, "random_auc": None}
    order = np.argsort(-values, kind="stable")
    ranked, sorted_values = labels[order], values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_values)) + 1]
    ends = np.r_[starts[1:], count]
    hits, contribution, wins = 0, 0.0, 0.0
    for start, end in zip(starts, ends):
        size = int(end - start)
        good = int(ranked[start:end].sum())
        offsets = np.arange(size, dtype=np.float64)
        expected_previous = offsets * (good - 1) / (size - 1) if size > 1 else offsets
        contribution += float(np.sum(good / size * (hits + 1 + expected_previous) / (start + offsets + 1)))
        negatives_below = count - end - (positives - hits - good)
        wins += good * (negatives_below + 0.5 * (size - good))
        hits += good
    harmonic = float(np.sum(1 / np.arange(1, count + 1, dtype=np.float64)))
    random_ap = (harmonic + (positives - 1) * (count - harmonic) / (count - 1)) / count if count > 1 else 1.0
    average_precision = contribution / positives
    return {"positives": positives, "prevalence": prevalence, "ap": average_precision,
            "random_ap": random_ap, "ap_ceiling": 1.0,
            "ap_closure": (average_precision - random_ap) / (1 - random_ap) if random_ap < 1 else None,
            "auc": wins / (positives * (count - positives)) if positives < count else None,
            "random_auc": 0.5 if positives < count else None}


def expected_top_hits(labels: np.ndarray, values: np.ndarray, count: int) -> float:
    """返回取前 count 张时的期望正例数；预算边界等分按均匀取舍，不使用指纹或真值破平局。"""
    if count <= 0 or count > len(labels):
        raise ValueError("预算越界")
    boundary = np.partition(values, len(values) - count)[len(values) - count]
    above, tied = values > boundary, values == boundary
    remaining = count - int(above.sum())
    return float(labels[above].sum() + labels[tied].sum() * remaining / tied.sum())


def event_members(metadata: dict, scores: dict, events: list[str]) -> dict:
    """验证事件、星级与分数覆盖，返回事件到全部照片的映射；不使用团顶身份筛样本。"""
    if not events or len(events) != len(set(events)):
        raise ValueError("事件为空或重复")
    requested = set(events)
    result = {event: {} for event in sorted(events)}
    for fingerprint, row in metadata.items():
        if row["event"] in requested:
            if int(row["rating_raw"]) not in range(6):
                raise ValueError(f"非法星级: {fingerprint}")
            result[row["event"]][fingerprint] = row
    for event, members in result.items():
        if not members:
            raise ValueError(f"事件无照片: {event}")
        require_coverage(scores, members)
    return result


def budget_metrics(metadata: dict, scores: dict, events: list[str], budgets=(0.125, 0.25, 0.5),
                   threshold: int = 3) -> list[dict]:
    """在完整事件中直排取预算，返回精度/召回及同分母随机和完美排序期望；默认 ≥3★。"""
    if threshold not in range(1, 6) or not budgets or any(not 0 < fraction <= 1 for fraction in budgets):
        raise ValueError("非法星级门槛或预算")
    results = []
    for event, members in event_members(metadata, scores, events).items():
        labels = np.array([int(row["rating_raw"]) >= threshold for row in members.values()])
        values = np.array([scores[fingerprint] for fingerprint in members], dtype=np.float64)
        positives, total = int(labels.sum()), len(labels)
        for fraction in budgets:
            count = math.ceil(total * fraction)
            hits = expected_top_hits(labels, values, count)
            results.append({"event": event, "budget_fraction": fraction, "threshold": threshold,
                            "photos": total, "budget": count, "kept": count,
                            "positive_photos": positives, "positive_photo_hits": hits,
                            "precision": hits / count, "recall": hits / positives if positives else None,
                            "prevalence": positives / total, "random_precision": positives / total,
                            "random_recall": count / total if positives else None,
                            "precision_ceiling": min(count, positives) / count,
                            "recall_ceiling": min(count, positives) / positives if positives else None,
                            "label_scope": "旧星代理；完整事件直排；预算取 ceil；等分均匀取舍"})
    return results


def ranking_metrics(metadata: dict, scores: dict, events: list[str], thresholds=(1, 2, 3)) -> dict:
    """逐门槛报告事件宏平均与按张数加权的 AP/AUC；随机为解析期望，逐事件明细保留。"""
    if not thresholds or any(threshold not in range(1, 6) for threshold in thresholds):
        raise ValueError("非法排序门槛")
    members_by_event = event_members(metadata, scores, events)
    result = {}
    for threshold in thresholds:
        details = {}
        for event, members in members_by_event.items():
            labels = np.array([int(row["rating_raw"]) >= threshold for row in members.values()])
            values = np.array([scores[fingerprint] for fingerprint in members])
            details[event] = {"photos": len(members), **binary_ranking(labels, values)}
        valid = [row for row in details.values() if row["ap"] is not None]
        average_precision = mean_or_none(row["ap"] for row in valid)
        random_ap = mean_or_none(row["random_ap"] for row in valid)
        result[str(threshold)] = {"events": len(details), "events_with_positives": len(valid),
            "ap_macro": average_precision, "random_ap_macro": random_ap,
            "ap_ceiling": 1.0 if valid else None,
            "ap_closure": (average_precision - random_ap) / (1 - random_ap) if valid and random_ap < 1 else None,
            "prevalence_macro": mean_or_none(row["prevalence"] for row in details.values()),
            "auc_macro": mean_or_none(row["auc"] for row in details.values() if row["auc"] is not None),
            "ap_photo_weighted": float(np.average([row["ap"] for row in valid],
                weights=[row["photos"] for row in valid])) if valid else None,
            "by_event": details}
    return result


def classification_summary(truth: np.ndarray, predictions: np.ndarray) -> dict:
    """汇总四档混淆矩阵、逐档精度/召回及准确率；缺失类别分母返回 None。"""
    confusion = np.zeros((4, 4), dtype=np.int64)
    np.add.at(confusion, (truth, predictions), 1)
    support, predicted = confusion.sum(axis=1), confusion.sum(axis=0)
    recalls = [int(confusion[category, category]) / int(support[category]) if support[category] else None
               for category in range(4)]
    return {"photos": len(truth), "confusion": confusion.tolist(),
            "accuracy": float(np.trace(confusion) / len(truth)),
            "balanced_accuracy": mean_or_none(value for value in recalls if value is not None),
            "majority_accuracy": float(support.max() / len(truth)), "accuracy_ceiling": 1.0,
            "uniform_random_balanced_accuracy": 0.25,
            "by_class": {str(category): {"support": int(support[category]),
                "predicted": int(predicted[category]), "recall": recalls[category],
                "precision": int(confusion[category, category]) / int(predicted[category]) if predicted[category] else None}
                for category in range(4)}}


def rating_metrics(metadata: dict, probabilities: dict, events: list[str]) -> dict:
    """评价 0／1／2／≥3 四类概率，返回混淆矩阵、逐事件准确率与 ≥3 档实际选出精度/召回。"""
    members_by_event = event_members(metadata, dict.fromkeys(probabilities, 0.0), events)
    all_truth, all_predictions, details = [], [], {}
    for event, members in members_by_event.items():
        values = np.array([probabilities[fingerprint] for fingerprint in members], dtype=np.float64)
        if values.shape != (len(members), 4) or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError("非法四档概率")
        if not np.allclose(values.sum(axis=1), 1.0, atol=1e-7, rtol=0):
            raise ValueError("四档概率之和不为一")
        truth = np.array([min(int(row["rating_raw"]), 3) for row in members.values()])
        predictions = values.argmax(axis=1)
        current = classification_summary(truth, predictions)
        positive, selected = truth == 3, predictions == 3
        hits = int((positive & selected).sum())
        current["selected_ge3"] = {"kept": int(selected.sum()), "positives": int(positive.sum()),
            "hits": hits, "precision": hits / int(selected.sum()) if selected.any() else None,
            "recall": hits / int(positive.sum()) if positive.any() else None,
            "prevalence": float(positive.mean())}
        details[event] = current
        all_truth.extend(truth.tolist())
        all_predictions.extend(predictions.tolist())
    pooled = classification_summary(np.array(all_truth), np.array(all_predictions))
    return {"classes": ["0", "1", "2", ">=3"], "events": len(details),
            "accuracy_macro": mean_or_none(row["accuracy"] for row in details.values()),
            "balanced_accuracy_macro": mean_or_none(row["balanced_accuracy"] for row in details.values()),
            "majority_accuracy_macro": mean_or_none(row["majority_accuracy"] for row in details.values()),
            "class_recall_macro": {str(category): mean_or_none(row["by_class"][str(category)]["recall"]
                for row in details.values() if row["by_class"][str(category)]["recall"] is not None)
                for category in range(4)}, "pooled": pooled, "by_event": details}


def verify_snapshot(path: Path) -> dict:
    """校验快照内每个输入哈希和协议版本，返回清单；任何漂移都拒绝继续。"""
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest["protocol"] != VERSION:
        raise ValueError("评估协议版本不匹配")
    for filename, item in manifest["files"].items():
        if digest(path / filename) != item["sha256"]:
            raise ValueError(f"快照内容已漂移: {filename}")
    return manifest
