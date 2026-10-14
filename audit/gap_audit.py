"""距目标差距审计：把当前候选放到"随机 / 结构上限 / 第一期目标"三条参照线之间。

只读数据集库、M3 照片清单与已有分数文件，不训练、不改标签。输出覆盖式写入 --out 目录
（JSON + Markdown）。所有召回均为事件宏平均；旧星仅作代理真值。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (0.125, 0.25, 0.5)
COS_BINS = (0.80, 0.88, 0.92, 0.95, 0.97, 0.98, 0.99, 1.01)


def load_scores(path: Path) -> dict[str, float]:
    """读取 fingerprint,score 两列分数文件，返回指纹到分数的映射。"""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {row["fingerprint"]: float(row["score"]) for row in csv.DictReader(stream)}


def rank_sum(fingerprints: list[str], *score_maps: dict) -> dict[str, float]:
    """事件内按各分数的名次相加做无参融合，避免不同分数尺度直接相加。"""
    total = dict.fromkeys(fingerprints, 0.0)
    for scores in score_maps:
        for rank, fingerprint in enumerate(sorted(fingerprints, key=lambda f: scores[f])):
            total[fingerprint] += rank
    return total


def one_per_cluster(fingerprints, meta, rep_score, rank_score, budget, group_max: bool = False) -> set[str]:
    """复刻现行候选管线：每个相似团只留代表（rep_score 最高），再按 rank_score 取前 budget。

    group_max=True 时以团内最高 rank 分给代表排序，对应官方候选 `group_quality` 变体。
    """
    clusters = defaultdict(list)
    for fingerprint in fingerprints:
        clusters[meta[fingerprint]["cluster_id"]].append(fingerprint)
    reps = {group: max(members, key=lambda f: rep_score[f]) for group, members in clusters.items()}
    if group_max:
        ranking = {f: max(rank_score[m] for m in clusters[meta[f]["cluster_id"]]) for f in reps.values()}
    else:
        ranking = {f: rank_score[f] for f in reps.values()}
    return set(sorted(reps.values(), key=lambda f: ranking[f], reverse=True)[:budget])


def top_k(fingerprints, score, budget) -> set[str]:
    """不做团去重，直接按分数取前 budget 张。"""
    return set(sorted(fingerprints, key=lambda f: score[f], reverse=True)[:budget])


def auc(positive: list[float], negative: list[float]) -> float:
    """秩和法计算 AUC，并列取平均名次。"""
    merged = sorted([(v, 1) for v in positive] + [(v, 0) for v in negative])
    rank_total, index = 0.0, 0
    while index < len(merged):
        end = index
        while end < len(merged) and merged[end][0] == merged[index][0]:
            end += 1
        rank_total += sum((index + end + 1) / 2 for k in range(index, end) if merged[k][1])
        index = end
    p, n = len(positive), len(negative)
    return (rank_total - p * (p + 1) / 2) / (p * n)


def average_precision(order: list[str], positives: set[str]) -> float | None:
    """平均精度（AP）：逐张累加命中率再除以正例总数，等价于各召回点精度的加权平均。

    优点：不需要预设预算，"多留一点"只会降低精度，因此不可被预算操纵；下限天然为正例基准率。
    """
    if not positives:
        return None
    hits, total = 0, 0.0
    for rank, fingerprint in enumerate(order, 1):
        if fingerprint in positives:
            hits += 1
            total += hits / rank
    return total / len(positives)


def precision_at(order: list[str], positives: set[str], fraction: float) -> float:
    """固定预算精度：前 fraction 张里正例所占比例（必须与基准率并列读）。"""
    count = max(1, int(round(len(order) * fraction)))
    return len(set(order[:count]) & positives) / count


def average_precision_section(meta, events, scores, rng) -> dict:
    """主判据：≥3★ 平均精度（AP）及其缺口闭合率；并列固定预算精度。

    AP 取代"达 80% 召回所需预算"——后者可被预算操纵（随机看够 78–82% 即达 80% 召回），
    且是事件构成的函数（占比低的事件"全留即达标"）。
    """
    local, glob = scores["lora"], scores["clip"]
    result = {}
    for threshold in (3, 4):
        per_selector = defaultdict(list)
        precision = defaultdict(list)
        purity_at_80 = defaultdict(list)
        budget_at_80 = defaultdict(list)
        for event, fingerprints in events.items():
            positives = {f for f in fingerprints if int(meta[f]["rating_raw"]) >= threshold}
            if not positives:
                continue
            n = len(fingerprints)
            fused = rank_sum(fingerprints, local, glob)
            oracle = {f: int(meta[f]["rating_raw"]) + rng.random() * 0.1 for f in fingerprints}
            noise = {f: rng.random() for f in fingerprints}
            # 候选按"每团代表优先"排序：代表按团内最高 CLIP 分，非代表压到底部（复刻 group_quality）
            clusters = defaultdict(list)
            for fingerprint in fingerprints:
                clusters[meta[fingerprint]["cluster_id"]].append(fingerprint)
            representatives = {group: max(members, key=lambda f: local[f]) for group, members in clusters.items()}
            candidate = {f: (max(glob[m] for m in clusters[meta[f]["cluster_id"]])
                             if f in representatives.values() else -1e9) + rng.random() * 1e-3 for f in fingerprints}
            orders = {
                "随机": sorted(fingerprints, key=lambda f: noise[f], reverse=True),
                "现行候选：团代表(LoRA)+团内最高CLIP排序(group_quality)": sorted(fingerprints, key=lambda f: candidate[f], reverse=True),
                "LoRA 直排(无团去重)": sorted(fingerprints, key=lambda f: local[f], reverse=True),
                "CLIP 直排(无团去重)": sorted(fingerprints, key=lambda f: glob[f], reverse=True),
                "LoRA+CLIP 名次和(无团去重)": sorted(fingerprints, key=lambda f: fused[f], reverse=True),
                "上限：完美打分": sorted(fingerprints, key=lambda f: oracle[f], reverse=True),
            }
            for name, order in orders.items():
                per_selector[name].append(average_precision(order, positives))
                for fraction in BUDGETS:
                    precision[f"{name}@{fraction}"].append(precision_at(order, positives, fraction))
                hits = 0
                for index, fingerprint in enumerate(order, 1):
                    hits += fingerprint in positives
                    if hits >= 0.8 * len(positives):
                        budget_at_80[name].append(index / n)
                        purity_at_80[name].append(hits / index)
                        break
        random_ap = float(np.mean(per_selector["随机"]))
        result[f">={threshold}"] = {
            "ap_macro": {name: round(float(np.mean(values)), 3) for name, values in per_selector.items()},
            "ap_closure_vs_random": {name: round((float(np.mean(values)) - random_ap) / (1 - random_ap), 3)
                                     for name, values in per_selector.items()},
            "precision_macro": {key: round(float(np.mean(values)), 3) for key, values in precision.items()},
            "budget_for_80pct_recall_macro": {name: round(float(np.mean(values)), 3) for name, values in budget_at_80.items()},
            "purity_at_80pct_recall_macro": {name: round(float(np.mean(values)), 3) for name, values in purity_at_80.items()},
        }
    return result


def budget_section(meta, events, scores, rng) -> dict:
    """各选片器在 ≥3★/≥4★ 代理真值下的预算召回、达 80% 召回所需预算，以及配对 bootstrap。"""
    local, glob = scores["lora"], scores["clip"]
    result = {}
    for threshold in (3, 4):
        rows = defaultdict(list)
        need80 = defaultdict(list)
        for fingerprints in events.values():
            positives = {f for f in fingerprints if int(meta[f]["rating_raw"]) >= threshold}
            if not positives:
                continue
            n = len(fingerprints)
            fused = rank_sum(fingerprints, local, glob)
            noisy_star = {f: int(meta[f]["rating_raw"]) + rng.random() * 0.1 for f in fingerprints}
            for fraction in BUDGETS:
                budget = math.ceil(n * fraction)
                recall = lambda selected: len(selected & positives) / len(positives)
                random_runs = []
                for _ in range(50):
                    noise = {f: rng.random() for f in fingerprints}
                    random_runs.append(recall(top_k(fingerprints, noise, budget)))
                rows[("随机", fraction)].append(sum(random_runs) / len(random_runs))
                rows[("现行候选：团代表(LoRA)+团内最高CLIP排序(group_quality)", fraction)].append(
                    recall(one_per_cluster(fingerprints, meta, local, glob, budget, group_max=True)))
                rows[("现行候选·简化变体(团代表+代表CLIP分)", fraction)].append(
                    recall(one_per_cluster(fingerprints, meta, local, glob, budget)))
                rows[("LoRA 直排(无团去重)", fraction)].append(recall(top_k(fingerprints, local, budget)))
                rows[("CLIP 直排(无团去重)", fraction)].append(recall(top_k(fingerprints, glob, budget)))
                rows[("LoRA+CLIP 名次和(无团去重)", fraction)].append(recall(top_k(fingerprints, fused, budget)))
                rows[("上限：完美打分但每团一张", fraction)].append(
                    recall(one_per_cluster(fingerprints, meta, noisy_star, noisy_star, budget)))
                rows[("上限：完美打分", fraction)].append(recall(top_k(fingerprints, noisy_star, budget)))
            for name, score in (("随机", None), ("LoRA+CLIP 名次和(无团去重)", fused)):
                order = sorted(fingerprints, key=(lambda f: rng.random()) if score is None else (lambda f: -score[f]))
                hits = 0
                for index, fingerprint in enumerate(order, 1):
                    hits += fingerprint in positives
                    if hits >= 0.8 * len(positives):
                        need80[name].append(index / n)
                        break
        # 配对 bootstrap：名次和(无团去重) − 现行候选，12.5% 预算，按事件重抽
        candidate_key = "现行候选：团代表(LoRA)+团内最高CLIP排序(group_quality)"
        fused_key = "LoRA+CLIP 名次和(无团去重)"
        delta = [b - a for a, b in zip(rows[(candidate_key, 0.125)], rows[(fused_key, 0.125)])]
        boots = sorted(sum(rng.choice(delta) for _ in delta) / len(delta) for _ in range(5000))
        # 缺口闭合率 =（实测 − 随机）/（结构上限 − 随机）：上限即靶，无需人类复测线（用户 2026-10-13 裁定）
        closure = {}
        for fraction in BUDGETS:
            random_values = rows[("随机", fraction)]
            upper_values = rows[("上限：完美打分", fraction)]
            for name in (candidate_key, "LoRA 直排(无团去重)", "CLIP 直排(无团去重)", fused_key):
                gaps = [(_upper - _random) for _upper, _random in zip(upper_values, random_values)]
                values = rows[(name, fraction)]
                ratios = [(value - _random) / gap for value, _random, gap in zip(values, random_values, gaps) if gap > 1e-9]
                closure[f"{name}@{fraction}"] = round(float(np.mean(ratios)), 3) if ratios else None
        result[f">={threshold}"] = {
            "recall_macro": {f"{name}@{fraction}": round(float(np.mean(values)), 3)
                             for (name, fraction), values in rows.items()},
            "closure_of_gap_macro": closure,
            "budget_for_80pct_recall_macro": {name: round(float(np.mean(v)), 3) for name, v in need80.items()},
            "ranksum_minus_candidate_at_0.125": {"mean": round(float(np.mean(delta)), 3),
                                                 "event_bootstrap95": [round(boots[125], 3), round(boots[4875], 3)]},
        }
    return result


def auc_section(meta, split, scores) -> dict:
    """事件内各星级门槛的 AUC（事件规模加权），量模型对 0→1 … 4→5 各边界的区分力。"""
    result = {}
    for name, score in scores.items():
        by_threshold = {}
        for threshold in (1, 2, 3, 4, 5):
            groups = defaultdict(lambda: ([], []))
            for fingerprint, row in meta.items():
                if row["split"] == split:
                    groups[row["event"]][0 if int(row["rating_raw"]) >= threshold else 1].append(score[fingerprint])
            pairs = [(auc(p, n), len(p) + len(n)) for p, n in groups.values() if p and n]
            by_threshold[f">={threshold}"] = round(sum(a * w for a, w in pairs) / sum(w for _, w in pairs), 3)
        result[name] = by_threshold
    return result


def copromotion_section(meta, split, features) -> dict:
    """同段照片对按 DINO 余弦分桶：一张晋级时另一张也晋级的条件概率——实测"雷同压低"从多像开始生效。"""
    fingerprints, matrix = features
    index = {f: i for i, f in enumerate(fingerprints)}
    segments = defaultdict(list)
    for fingerprint, row in meta.items():
        if row["split"] == split:
            segments[(row["event"], row["seg_id"])].append(fingerprint)
    result = {}
    for threshold in (1, 3):
        counts = np.zeros((len(COS_BINS) - 1, 2))
        for members in segments.values():
            if len(members) < 2:
                continue
            rows = [index[f] for f in members]
            similarity = matrix[rows] @ matrix[rows].T
            stars = np.array([int(meta[f]["rating_raw"]) for f in members])
            upper = np.triu_indices(len(members), 1)
            cos, left, right = similarity[upper], stars[upper[0]], stars[upper[1]]
            any_up = (left >= threshold) | (right >= threshold)
            both_up = (left >= threshold) & (right >= threshold)
            bins = np.digitize(cos[any_up], COS_BINS) - 1
            for b in range(len(COS_BINS) - 1):
                mask = bins == b
                counts[b] += (mask.sum(), both_up[any_up][mask].sum())
        result[f">={threshold}"] = {f"[{COS_BINS[b]:.2f},{COS_BINS[b + 1]:.2f})":
                                    {"p_both": round(counts[b, 1] / max(1, counts[b, 0]), 3), "pairs": int(counts[b, 0])}
                                    for b in range(len(COS_BINS) - 1)}
    return result


def capture_order_section(meta, split, times, local) -> dict:
    """相似团（≥3 张、唯一团顶）内团顶的拍摄序位：首张/末张/中间 vs 随机期望，检验"拍到满意为止"先验。"""
    clusters = defaultdict(list)
    for fingerprint, row in meta.items():
        if row["split"] == split and int(row["cluster_size"]) >= 3:
            clusters[(row["event"], row["cluster_id"])].append(fingerprint)
    tally, count = defaultdict(float), 0
    for members in clusters.values():
        members = sorted(members, key=lambda f: times[f])
        stars = [int(meta[f]["rating_raw"]) for f in members]
        best = max(stars)
        if best == 0 or stars.count(best) != 1:
            continue
        count += 1
        winner, length = stars.index(best), len(members)
        tally["末张"] += winner == length - 1
        tally["首张"] += winner == 0
        tally["随机期望"] += 1 / length
        tally["LoRA 团内最高"] += max(range(length), key=lambda i: local[members[i]]) == winner
    return {"clusters": count, **{k: round(v / count, 3) for k, v in tally.items()}}


def feasibility_section(meta) -> list[dict]:
    """逐事件 ≥3★ 占比与 12.5% 预算下理论最高召回：占比 >15.6% 的事件 80% 目标在数学上不可达。"""
    events = defaultdict(lambda: [0, 0, ""])
    for row in meta.values():
        entry = events[row["event"]]
        entry[0] += 1
        entry[1] += int(row["rating_raw"]) >= 3
        entry[2] = row["split"]
    return [{"split": split, "photos": n, "share_ge3": round(p / n, 3), "max_recall_at_0.125": round(min(1, 0.125 * n / p), 3)}
            for n, p, split in sorted(events.values(), key=lambda e: e[1] / e[0])]


def required_auc_section(positive_rate: float, budgets=BUDGETS) -> dict:
    """二元正态近似：给定正例率下，各预算达到 80% 召回所需的排序 AUC（不可达记 None）。

    正负例分数为 N(d,1) / N(0,1)，AUC = Φ(d/√2)。预算 cut 取**混合分布**的 (1-b) 分位，
    不能取负例分位——正例占比不小时两者差别很大。
    """
    def recall(area: float, budget: float) -> float:
        separation = math.sqrt(2) * _ndtri(area)
        low, high = -10.0, 10.0
        for _ in range(80):
            middle = (low + high) / 2
            tail = positive_rate * (1 - _ncdf(middle - separation)) + (1 - positive_rate) * (1 - _ncdf(middle))
            if tail > budget:
                low = middle
            else:
                high = middle
        return 1 - _ncdf(high - separation)

    result = {}
    for budget in budgets:
        low, high = 0.5, 0.9999
        for _ in range(60):
            middle = (low + high) / 2
            if recall(middle, budget) < 0.8:
                low = middle
            else:
                high = middle
        result[f"{budget}"] = None if high >= 0.9995 else round(high, 3)
    return result


def _ncdf(value: float) -> float:
    """标准正态累积分布函数。"""
    return 0.5 * (1 + math.erf(value / math.sqrt(2)))


def _ndtri(probability: float) -> float:
    """标准正态分位数（Acklam 近似，精度足够本用途）。"""
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if probability < plow:
        q = math.sqrt(-2 * math.log(probability))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if probability > phigh:
        q = math.sqrt(-2 * math.log(1 - probability))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q, r = probability - 0.5, (probability - 0.5) ** 2
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def main() -> int:
    """解析路径参数，逐 split 运行各审计节并写出 JSON/Markdown。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("D:/PhotoDB/dataset/photos_dataset.db"))
    parser.add_argument("--m3", type=Path, default=ROOT / "audit/out/m3_pairs_v2_20260924b")
    parser.add_argument("--local", type=Path, default=ROOT / "train/out/recovery_work/rescored_lora_v2/scores.csv")
    parser.add_argument("--global-scores", type=Path, default=ROOT / "train/out/recovery_work/corrected_rank/clip_scores.csv")
    parser.add_argument("--features", type=Path, default=ROOT / "train/out/recovery_work/corrected_features.npz")
    parser.add_argument("--out", type=Path, default=ROOT / "audit/out/gap_audit")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with (args.m3 / "photos.csv").open(encoding="utf-8-sig", newline="") as stream:
        meta = {row["fingerprint"]: row for row in csv.DictReader(stream)}
    connection = sqlite3.connect(args.db.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        times = {f: (t or "") + "." + (s or "0").zfill(3)
                 for f, t, s in connection.execute("SELECT fingerprint,capture_time,capture_subsec FROM photos")}
    finally:
        connection.close()
    scores = {"lora": load_scores(args.local), "clip": load_scores(args.global_scores)}
    archive = np.load(args.features)
    dino = archive["dino"] / np.linalg.norm(archive["dino"], axis=1, keepdims=True)
    features = (list(archive["fingerprints"]), dino)
    rng = random.Random(args.seed)

    report = {"scope": "v2 划分；旧星为代理真值；val/test 已被开发流程接触，非终考", "splits": {},
              "feasibility": feasibility_section(meta)}
    for split in ("train", "val", "test"):
        events = defaultdict(list)
        for fingerprint, row in meta.items():
            if row["split"] == split:
                events[row["event"]].append(fingerprint)
        section = {"auc": auc_section(meta, split, scores),
                   "copromotion_by_cos": copromotion_section(meta, split, features),
                   "capture_order": capture_order_section(meta, split, times, scores["lora"])}
        if split != "train":
            section["average_precision"] = average_precision_section(meta, events, scores, rng)
            section["budget"] = budget_section(meta, events, scores, rng)
        report["splits"][split] = section
        print(f"完成 {split}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "gap_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 距目标差距审计", "", report["scope"], ""]
    for split in ("val", "test"):
        for threshold, block in report["splits"][split]["average_precision"].items():
            lines += [f"## {split} {threshold}★ 平均精度 AP（主判据，事件宏平均）", "",
                      "| 选片器 | AP | AP缺口闭合 | P@12.5% | P@25% | P@50% |", "|---|---:|---:|---:|---:|---:|"]
            for name, value in block["ap_macro"].items():
                closure = block["ap_closure_vs_random"][name]
                cells = [str(block["precision_macro"][f"{name}@{fraction}"]) for fraction in BUDGETS]
                lines.append(f"| {name} | {value} | {closure} | " + " | ".join(cells) + " |")
            lines += ["", f"- R80 预算（须连同纯度读）：{block['budget_for_80pct_recall_macro']}",
                      f"- R80 点纯度：{block['purity_at_80pct_recall_macro']}", ""]
        for threshold, block in report["splits"][split]["budget"].items():
            lines += [f"## {split} {threshold}★ 召回（辅助口径，注意预算可被操纵）", "",
                      "| 选片器 | @12.5% | @25% | @50% |", "|---|---:|---:|---:|"]
            names = dict.fromkeys(key.rsplit("@", 1)[0] for key in block["recall_macro"])
            for name in names:
                cells = [str(block["recall_macro"][f"{name}@{fraction}"]) for fraction in BUDGETS]
                lines.append(f"| {name} | " + " | ".join(cells) + " |")
            lines += ["", f"- 名次和 − 现行候选 @12.5%：{block['ranksum_minus_candidate_at_0.125']}", ""]
    (args.out / "gap_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
