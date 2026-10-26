"""三边界累计logistic评分探针：每边界独立视觉方向，单调概率投影后形成四档。

保持事件/类别均衡、固定正则0.1与完整train事件三折，只改变评分目标表达。
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "audit"))
from evaluation_protocol import RATING_VERSION, digest, write_json
from rating_baseline import measure, write_scores
from context_rating_probe import load_inputs, contextual_features, compare


def project_probabilities(cumulative: np.ndarray) -> np.ndarray:
    """将三累计概率作最小二乘非增投影并还原四类概率，不使用标签或可调阈值。"""
    if cumulative.ndim != 2 or cumulative.shape[1] != 3 or not np.isfinite(cumulative).all():
        raise ValueError("非法累计概率形状或值")
    if np.any(cumulative < 0) or np.any(cumulative > 1):
        raise ValueError("累计概率越界")
    total = cumulative.sum(axis=1)
    first = np.maximum.reduce((cumulative[:, 0], cumulative[:, :2].mean(axis=1), total / 3))
    last = np.minimum.reduce((cumulative[:, 2], cumulative[:, 1:].mean(axis=1), total / 3))
    middle = total - first - last
    ordered = np.column_stack((first, middle, last))
    values = np.column_stack((1 - ordered[:, 0], ordered[:, 0] - ordered[:, 1],
                              ordered[:, 1] - ordered[:, 2], ordered[:, 2]))
    if np.any(values < -1e-12):
        raise ValueError("单调投影失败")
    return np.maximum(values, 0) / np.maximum(values, 0).sum(axis=1, keepdims=True)


def fit_cumulative(features: np.ndarray, truth: np.ndarray, events: list[str]) -> tuple[np.ndarray, dict]:
    """拟合固定事件/类别等权的三累计logistic输出，返回系数与真实收敛记录。"""
    domains = Counter(zip(events, truth.tolist()))
    classes = Counter(event for event, category in domains)
    weights = np.array([1 / (domains[(event, int(category))] * classes[event]) for event, category in zip(events, truth)])
    weights /= weights.sum()
    targets = (truth[:, None] >= np.array([1, 2, 3])[None]).astype(np.float64)
    initial = np.zeros((features.shape[1] + 1, 3))
    prior = np.sum(targets * weights[:, None], axis=0)
    if np.any(prior <= 0) or np.any(prior >= 1):
        raise ValueError("训练折缺少累计边界正例或负例")
    initial[-1] = np.log(prior / (1 - prior))

    def objective(flattened):
        """返回三个等权累计边界损失和梯度，正则固定0.1、不处罚截距。"""
        coefficient = flattened.reshape(initial.shape)
        logits = features @ coefficient[:-1] + coefficient[-1]
        loss = np.sum(weights[:, None] * (np.logaddexp(0, logits) - targets * logits))
        loss += 0.05 * np.square(coefficient[:-1]).sum()
        residual = (expit(logits) - targets) * weights[:, None]
        gradient = np.vstack((features.T @ residual + 0.1 * coefficient[:-1], residual.sum(axis=0)))
        return float(loss), gradient.ravel()

    result = minimize(objective, initial.ravel(), jac=True, method="L-BFGS-B",
                      options={"maxiter": 2000, "ftol": 1e-11, "gtol": 1e-7})
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f"累计logistic未收敛: {result.message}")
    return result.x.reshape(initial.shape), {"iterations": int(result.nit), "loss": float(result.fun),
                                           "converged": True, "penalty": 0.1}


def main() -> int:
    """执行固定完整train事件三折双输入对照，保存OOF概率与对softmax的逐事件差。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photos", type=Path, default=ROOT / "audit/out/m3_pairs_v2_20260924b/photos.csv")
    parser.add_argument("--split-json", type=Path, default=ROOT / "data/split-v2-20260924.json")
    parser.add_argument("--features", type=Path, default=ROOT / "train/out/recovery_work/corrected_features.npz")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    threadpool_limits(2)
    started = time.monotonic()
    metadata, fingerprints, visual, dino = load_inputs(args)
    indexes = [index for index, fingerprint in enumerate(fingerprints) if metadata[fingerprint]["split"] == "train"]
    fingerprints = [fingerprints[index] for index in indexes]
    metadata = {fingerprint: metadata[fingerprint] for fingerprint in fingerprints}
    event_of = np.array([metadata[fingerprint]["event"] for fingerprint in fingerprints])
    truth = np.array([min(int(metadata[fingerprint]["rating_raw"]), 3) for fingerprint in fingerprints])
    events = sorted(set(event_of.tolist()), key=lambda event: hashlib.sha256(("20260926:" + event).encode()).hexdigest())
    folds = [events[index::3] for index in range(3)]
    visual = visual[indexes]
    context, identity = contextual_features(visual, dino[indexes], fingerprints, event_of, False)
    reference_path = ROOT / "train/out/context_rating_probe_20261014/result.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    inputs = [args.photos, args.split_json, args.features, args.features.with_suffix(".json"), Path(__file__),
              ROOT / "train/context_rating_probe.py", ROOT / "train/rating_baseline.py",
              ROOT / "audit/evaluation_protocol.py", reference_path]
    plan = {"protocol": RATING_VERSION, "scope": "完整train事件OOF，不打开val/test", "folds": folds,
            "penalty": 0.1, "weighting": "event_class", "context_identity": identity,
            "input_sha256": {str(path.resolve()): digest(path) for path in inputs}}
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "plan.json", plan)
    models = {}
    for name, features in (("single", visual), ("similar_context", context)):
        probabilities, traces = {}, []
        for held_out in folds:
            fitting = np.isin(event_of, sorted(set(events) - set(held_out)))
            testing = np.flatnonzero(np.isin(event_of, held_out))
            coefficient, trace = fit_cumulative(features[fitting], truth[fitting], event_of[fitting].tolist())
            cumulative = expit(features[testing] @ coefficient[:-1] + coefficient[-1])
            values = project_probabilities(cumulative)
            probabilities.update((fingerprints[index], value.tolist()) for index, value in zip(testing, values))
            traces.append({"held_out_events": held_out,
                           "crossed_probability_rows": int(np.any(np.diff(cumulative, axis=1) > 0, axis=1).sum()), **trace})
        metrics = measure(metadata, probabilities, events)
        models[name] = {"metrics": metrics, "traces": traces,
                        "difference_vs_softmax": compare(metrics, reference["models"][name]["metrics"])}
        write_scores(args.out / f"{name}_oof_scores.csv", fingerprints,
                     np.array([probabilities[fingerprint] for fingerprint in fingerprints]))
        print(f"{name} cumulative AP>=3={metrics['ranking']['3']['ap_macro']:.4f} "
              f"balanced_accuracy={metrics['rating']['balanced_accuracy_macro']:.4f}", flush=True)
    if any(digest(path) != plan["input_sha256"][str(path.resolve())] for path in inputs):
        raise ValueError("运行期间输入发生变化")
    result = {"plan": plan, "models": models, "seconds": time.monotonic() - started}
    write_json(args.out / "result.json", result)
    print(f"累计评分探针完成 {result['seconds']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
