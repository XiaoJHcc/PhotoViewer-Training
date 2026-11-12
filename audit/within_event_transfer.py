"""团内取舍是否事件特异：同一事件内按段分两半，A 半学成对方向、B 半评价，并与跨事件模型在同一批对上对照。

只读 train 事件与修复特征；对 = 同段 DINO cos≥0.88 的跨档对（min(星,3)），胜者为档位高者。事件内差分头用 A 半的对，
正则只在 A 半内部按段两折选；跨事件差分头与单图头按 rating_baseline 三折，只用其它事件训练。不开 val/test。
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "audit"))
import rating_baseline as rb
from evaluation_protocol import write_json
from within_scene_audit import baseline_oof

SIMILAR = 0.88
PENALTIES = (0.001, 0.01, 0.1)
BOOTSTRAP_SEED = 20261111


def segment_half(key: str) -> int:
    """段到半区的确定性哈希分配，不看标签。"""
    return int(hashlib.sha256(("half:" + key).encode()).hexdigest()[:8], 16) % 2


def cross_pairs(indices, segment_key, dino, tiers):
    """给定照片内同段 cos≥SIMILAR 的跨档对，返回 (胜, 负, 段键)。"""
    winners, losers, segments = [], [], []
    for segment in np.unique(segment_key[indices]):
        members = indices[segment_key[indices] == segment]
        if len(members) < 2:
            continue
        similarity = dino[members] @ dino[members].T
        first, second = np.triu_indices(len(members), 1)
        keep = (similarity[first, second] >= SIMILAR) & (tiers[members[first]] != tiers[members[second]])
        a, b = members[first[keep]], members[second[keep]]
        up = tiers[a] > tiers[b]
        winners.append(np.where(up, a, b))
        losers.append(np.where(up, b, a))
        segments.extend([segment] * int(keep.sum()))
    if not winners:
        return np.zeros(0, int), np.zeros(0, int), np.array([])
    return np.concatenate(winners), np.concatenate(losers), np.array(segments)


def fit_difference(features, winners, losers, penalty):
    """无截距成对 logistic：P(胜者>负者)=σ(w·(x_w−x_l))，返回 w。"""
    difference = features[winners] - features[losers]
    design = np.vstack((difference, -difference))
    target = np.r_[np.ones(len(difference)), np.zeros(len(difference))]

    def objective(weights):
        """返回负对数似然与梯度。"""
        logits = design @ weights
        probability = 1 / (1 + np.exp(-logits))
        loss = np.mean(np.logaddexp(0, logits) - target * logits) + penalty * weights @ weights / 2
        gradient = design.T @ (probability - target) / len(target) + penalty * weights
        return float(loss), gradient

    result = minimize(objective, np.zeros(features.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 1000})
    return result.x


def accuracy(margin):
    """胜者分数更高的比例（相等记 0.5）。"""
    return float(np.mean(np.where(margin == 0, 0.5, margin > 0))) if len(margin) else None


def choose_penalty(features, winners, losers, segments):
    """只在 A 半内部按段哈希两折选正则，返回最优正则与网格。"""
    fold = np.array([int(hashlib.sha256(("inner:" + s).encode()).hexdigest()[:8], 16) % 2 for s in segments])
    grid = []
    for penalty in PENALTIES:
        scores = []
        for part in (0, 1):
            train, test = fold != part, fold == part
            if train.sum() < 5 or test.sum() < 5:
                continue
            weights = fit_difference(features, winners[train], losers[train], penalty)
            scores.append(accuracy((features[winners[test]] - features[losers[test]]) @ weights))
        grid.append((np.mean(scores) if scores else 0.5, -penalty))
    best = max(grid)
    return -best[1], [{"penalty": -g[1], "inner_accuracy": g[0]} for g in grid]


def bootstrap(candidate: dict, reference: dict) -> dict:
    """同事件差的 bootstrap。"""
    events = sorted(set(candidate) & set(reference))
    differences = np.array([candidate[e] - reference[e] for e in events])
    samples = np.random.default_rng(BOOTSTRAP_SEED).choice(differences, size=(10000, len(differences))).mean(axis=1)
    return {"mean": float(differences.mean()), "event_bootstrap95": np.quantile(samples, (0.025, 0.975)).tolist(),
        "positive_events": int((differences > 0).sum()), "events": len(differences)}


def main() -> int:
    """逐事件两半互评，汇总事件内/跨事件准确率与配对差。"""
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
    joint = features["joint"]
    train_events = sorted(set(event_of[split_of == "train"].tolist()),
        key=lambda event: hashlib.sha256(("20260926:" + event).encode()).hexdigest())
    folds = [train_events[fold::3] for fold in range(3)]
    single_oof = baseline_oof(features, tiers, event_of, train_events) @ np.arange(4.0)
    cross_weights = dict()
    for held in folds:
        fitting = np.flatnonzero(np.isin(event_of, [e for e in train_events if e not in held]))
        winners, losers, _ = cross_pairs(fitting, segment_key, dino, tiers)
        weights = fit_difference(joint, winners, losers, 0.01)
        for event in held:
            cross_weights[event] = weights
    results = {"within_event_difference": dict(), "cross_event_difference": dict(), "cross_event_single": dict(),
        "within_event_single": dict(), "penalties": dict(), "pairs": dict()}
    for event in train_events:
        members = np.flatnonzero(event_of == event)
        half = np.array([segment_half(key) for key in segment_key[members]])
        margins = dict((name, []) for name in ("within_event_difference", "cross_event_difference", "cross_event_single",
            "within_event_single"))
        chosen, total = [], 0
        for train_half in (0, 1):
            a, b = members[half == train_half], members[half != train_half]
            wa, la, sa = cross_pairs(a, segment_key, dino, tiers)
            wb, lb, _ = cross_pairs(b, segment_key, dino, tiers)
            if len(wa) < 20 or len(wb) < 5 or set(tiers[a].tolist()) != set(range(4)):
                continue
            penalty, grid = choose_penalty(joint, wa, la, sa)
            chosen.append({"penalty": penalty, "grid": grid, "train_pairs": int(len(wa)), "test_pairs": int(len(wb))})
            weights = fit_difference(joint, wa, la, penalty)
            margins["within_event_difference"].append((joint[wb] - joint[lb]) @ weights)
            margins["cross_event_difference"].append((joint[wb] - joint[lb]) @ cross_weights[event])
            margins["cross_event_single"].append(single_oof[wb] - single_oof[lb])
            coefficient, _ = rb.fit_classifier(joint[a], tiers[a], [event] * len(a), 0.1, "event_class")
            expected = rb.predict(joint, coefficient) @ np.arange(4.0)
            margins["within_event_single"].append(expected[wb] - expected[lb])
            total += len(wb)
        if not chosen:
            continue
        for name, values in margins.items():
            results[name][event] = accuracy(np.concatenate(values))
        results["penalties"][event] = chosen
        results["pairs"][event] = total
        print(f"{event[:14]}: pairs {total} within-diff {results['within_event_difference'][event]:.3f} "
            f"cross-diff {results['cross_event_difference'][event]:.3f} cross-single {results['cross_event_single'][event]:.3f} "
            f"within-single {results['within_event_single'][event]:.3f}", flush=True)
    summary = dict((name, float(np.mean(list(results[name].values())))) for name in margins)
    comparisons = {"within_diff_minus_cross_diff": bootstrap(results["within_event_difference"], results["cross_event_difference"]),
        "within_diff_minus_cross_single": bootstrap(results["within_event_difference"], results["cross_event_single"]),
        "within_single_minus_cross_single": bootstrap(results["within_event_single"], results["cross_event_single"])}
    print("summary", summary, flush=True)
    for name, value in comparisons.items():
        print(name, value, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "report.json", {"summary": summary, "comparisons": comparisons, **results})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
