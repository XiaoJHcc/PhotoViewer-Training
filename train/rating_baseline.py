"""修复特征上的四档基线：训练事件三折选正则，固定后评价 val，不打开 test。

0／1／2／≥3 是类别，不把星级当等距数值回归。单图小头用于量化当前特征的
信息可读性；时间窗不是强制架构。名次和标量头仅作已有排序的分档校准对照。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "audit"))
from evaluation_protocol import (RATING_VERSION, budget_metrics, digest, load_scores,
                                 ranking_metrics, rating_metrics, read_rows, write_json)
from split_eval import summarize_budget


def load_inputs(args) -> tuple[dict, list[str], dict, dict]:
    """核验 v2 事件身份、修复缓存版本/哈希及完整覆盖，返回元数据、指纹、视觉特征与固定参考分数。"""
    rows = read_rows(args.photos)
    metadata = {row["fingerprint"]: row for row in rows}
    if not rows or len(metadata) != len(rows):
        raise ValueError("照片清单为空或有重复指纹")
    split = json.loads(args.split_json.read_text(encoding="utf-8"))
    all_events = [event for name in ("train", "val", "test") for event in split[name]]
    if len(all_events) != len(set(all_events)) or set(all_events) != {row["event"] for row in rows}:
        raise ValueError("事件划分重叠或覆盖不一致")
    for row in rows:
        if row["split"] not in ("train", "val", "test") or row["event"] not in split[row["split"]]:
            raise ValueError("照片清单事件与冻结 split 不一致")
        if int(row["rating_raw"]) not in range(6):
            raise ValueError("非法星级")
    identity = json.loads(args.features.with_suffix(".json").read_text(encoding="utf-8"))
    if identity["decode_version"] != "display-plane-v3" or identity["metadata_sha256"] != digest(args.photos):
        raise ValueError("修复缓存版本或照片身份不一致")
    with np.load(args.features, allow_pickle=False) as cached:
        fingerprints = cached["fingerprints"].tolist()
        dino, clip = cached["dino"].astype(np.float64), cached["features"].astype(np.float64)
    if fingerprints != sorted(metadata) or set(identity["images"]) != set(fingerprints):
        raise ValueError("特征指纹顺序、覆盖或来源图像身份不一致")
    for array, dimension in ((dino, 384), (clip, 768)):
        if array.shape != (len(fingerprints), dimension) or not np.isfinite(array).all():
            raise ValueError("特征形状或有限性错误")
        if not np.allclose(np.linalg.norm(array, axis=1), 1.0, atol=1e-4):
            raise ValueError("视觉特征不是逐张 L2 归一化")
    references = {"lora": load_scores(args.local), "clip_rank": load_scores(args.global_scores)}
    if any(set(scores) != set(metadata) for scores in references.values()):
        raise ValueError("参考分数覆盖不一致")
    features = {"dino": dino * np.sqrt(384), "clip": clip * np.sqrt(768)}
    features["joint"] = np.concatenate((features["dino"], features["clip"]), axis=1) / np.sqrt(2)
    total_rank = np.zeros(len(fingerprints), dtype=np.float64)
    for event in all_events:
        indexes = [index for index, fingerprint in enumerate(fingerprints) if metadata[fingerprint]["event"] == event]
        for scores in references.values():
            values = [scores[fingerprints[index]] for index in indexes]
            total_rank[indexes] += (rankdata(values) - 0.5) / len(indexes)
    references["rank_sum"] = dict(zip(fingerprints, total_rank.tolist()))
    features["rank_sum"] = total_rank[:, None]
    return metadata, fingerprints, features, references


def fit_classifier(features: np.ndarray, truth: np.ndarray, events: list[str], penalty: float,
                   weighting: str = "event") -> tuple[np.ndarray, dict]:
    """按指定事件/类别权重拟合四类 softmax 小头，返回系数与收敛记录；不作星级数值回归。"""
    counts = Counter(events)
    weights = np.array([1 / counts[event] for event in events], dtype=np.float64)
    if weighting == "event_class":
        domains = Counter(zip(events, truth.tolist()))
        classes_per_event = Counter(event for event, category in domains)
        weights = np.array([1 / (domains[(event, int(category))] * classes_per_event[event])
                            for event, category in zip(events, truth)], dtype=np.float64)
    elif weighting != "event":
        raise ValueError("未知分类损失权重")
    weights /= weights.sum()
    if set(truth.tolist()) != set(range(4)):
        raise ValueError("拟合照片未覆盖四档标签")
    prior = np.bincount(truth, weights=weights, minlength=4)
    initial = np.zeros((features.shape[1] + 1, 4), dtype=np.float64)
    initial[-1] = np.log(prior)

    def objective(flattened):
        """返回事件等权分类负对数似然与梯度，正则只作用于视觉系数。"""
        coefficient = flattened.reshape(initial.shape)
        logits = features @ coefficient[:-1] + coefficient[-1]
        normalizer = logsumexp(logits, axis=1)
        loss = weights @ (normalizer - logits[np.arange(len(truth)), truth])
        loss += penalty * np.square(coefficient[:-1]).sum() / 2
        residual = softmax(logits, axis=1)
        residual[np.arange(len(truth)), truth] -= 1
        residual *= weights[:, None]
        gradient = np.vstack((features.T @ residual + penalty * coefficient[:-1], residual.sum(axis=0)))
        return float(loss), gradient.ravel()

    result = minimize(objective, initial.ravel(), jac=True, method="L-BFGS-B",
                      options={"maxiter": 2000, "ftol": 1e-11, "gtol": 1e-7})
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f"分类优化未收敛: {result.message}")
    return result.x.reshape(initial.shape), {"photos": len(truth), "events": len(counts),
        "class_counts": np.bincount(truth, minlength=4).tolist(), "event_weighted_prior": prior.tolist(),
        "loss": float(result.fun), "iterations": int(result.nit), "converged": True, "weighting": weighting}


def predict(features: np.ndarray, coefficient: np.ndarray) -> np.ndarray:
    """将视觉特征与已拟合系数转为四类概率，返回每行和为一的数组。"""
    return softmax(features @ coefficient[:-1] + coefficient[-1], axis=1)


def measure(metadata: dict, probabilities: dict, events: list[str]) -> dict:
    """只用显式事件的标签评价四档概率、三个边界排序及 ≥3★ 预算曲线。"""
    ranking = {}
    for threshold in (1, 2, 3):
        scores = {fingerprint: float(np.sum(values[threshold:])) for fingerprint, values in probabilities.items()}
        ranking.update(ranking_metrics(metadata, scores, events, thresholds=(threshold,)))
    score_ge3 = {fingerprint: float(values[3]) for fingerprint, values in probabilities.items()}
    return {"ranking": ranking, "rating": rating_metrics(metadata, probabilities, events),
            "budget": summarize_budget(budget_metrics(metadata, score_ge3, events))}


def candidate_key(record: dict) -> tuple:
    """按训练事件 OOF ≥3★ AP 选择正则；精确同分才比较四档平均召回与更强正则。"""
    return (record["metrics"]["ranking"]["3"]["ap_macro"],
            record["metrics"]["rating"]["balanced_accuracy_macro"], record["penalty"])


def paired_difference(candidate: dict, reference: dict, section: str, metric: str) -> dict:
    """对相同事件的指标差做固定种子 bootstrap，返回均差、区间及逐事件差；不据此调参。"""
    left = candidate[section]["3"]["by_event"] if section == "ranking" else candidate[section]["by_event"]
    right = reference[section]["3"]["by_event"] if section == "ranking" else reference[section]["by_event"]
    if left.keys() != right.keys():
        raise ValueError("对照事件不一致")
    differences = {event: left[event][metric] - right[event][metric] for event in left}
    values = np.array(list(differences.values()))
    generator = np.random.default_rng(20261014)
    samples = generator.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return {"mean": float(values.mean()), "event_bootstrap95": np.quantile(samples, (0.025, 0.975)).tolist(),
            "by_event": differences}


def write_scores(path: Path, fingerprints: list[str], probabilities: np.ndarray) -> None:
    """保存全覆盖 ≥3★ 分数、四类概率与预测档位；文件存在时拒绝覆盖。"""
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["fingerprint", "score", "p0", "p1", "p2", "p_ge3", "predicted_class"])
        for fingerprint, values in zip(fingerprints, probabilities):
            writer.writerow([fingerprint, float(values[3]), *values.tolist(), int(values.argmax())])


def render_report(result: dict) -> str:
    """将固定参数的 val 结果渲染为可读表，含随机/上限及低分档召回。"""
    lines = ["# 四档视觉基线", "", "v2 开发留出；只评 val，不打开 test；旧星为代理真值。",
             f"类别 = 0／1／2／≥3；损失权重 = {result['plan']['weighting']}；参数仅由 train 三折选择。", "",
             "| val 模型 | AP≥3 | 随机 AP | 上限 | AP 缺口闭合 | 四档平均召回 | 0 / 1 / 2 / ≥3 召回 |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for name, entry in result["models"].items():
        metrics = entry["val"]
        ranking, rating = metrics["ranking"]["3"], metrics["rating"]
        recalls = " / ".join(f"{rating['class_recall_macro'][str(category)]:.3f}" for category in range(4))
        lines.append(f"| {name} | {ranking['ap_macro']:.4f} | {ranking['random_ap_macro']:.4f} | 1.000 | "
                     f"{ranking['ap_closure']:.4f} | {rating['balanced_accuracy_macro']:.4f} | {recalls} |")
    lines.extend(["", "四档平均召回的均匀随机参照 = 0.25，上限 = 1.0；准确率必须连同多数类基线读。",
                  "完整逐事件 AP、各边界 AUC、混淆矩阵、预算精度/召回见 result.json。", "",
                  "## 已有分数的冻结直排对照", "",
                  "| val 排序 | AP≥1 | AP≥2 | AP≥3 |", "|---|---:|---:|---:|"])
    for name, entry in result["references"].items():
        lines.append(f"| {name} | " + " | ".join(f"{entry['ranking'][str(threshold)]['ap_macro']:.4f}"
                                                   for threshold in (1, 2, 3)) + " |")
    lines.extend(["", "参考模型历史上已经使用 train 标签；名次和校准的 train OOF 仅隔离校准头，",
                  "不能把它当作整个参考模型从未见过该事件的 OOF；val 对照不受此校准泄漏影响。",
                  "参考排序器不参与视觉模型的 OOF 选择。",
                  "类别均衡头输出为加权目标下的决策概率，未经真实星级比例校准。",
                  "本实验不含 CV、patch 或上下文，不将结论外推为这些输入无效。"])
    return "\n".join(lines) + "\n"


def main() -> int:
    """核验输入、执行事件三折小头训练并冻结参数，只评 val，保存系数、分数与证据哈希。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photos", type=Path, default=ROOT / "audit/out/m3_pairs_v2_20260924b/photos.csv")
    parser.add_argument("--split-json", type=Path, default=ROOT / "data/split-v2-20260924.json")
    parser.add_argument("--features", type=Path, default=ROOT / "train/out/recovery_work/corrected_features.npz")
    parser.add_argument("--local", type=Path, default=ROOT / "train/out/recovery_work/rescored_lora_v2/scores.csv")
    parser.add_argument("--global-scores", type=Path, default=ROOT / "train/out/recovery_work/corrected_rank/clip_scores.csv")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--weighting", choices=("event", "event_class"), default="event")
    args = parser.parse_args()
    started = time.monotonic()
    threadpool_limits(2)
    metadata, fingerprints, features, references = load_inputs(args)
    train_events = sorted({row["event"] for row in metadata.values() if row["split"] == "train"},
                          key=lambda event: hashlib.sha256(("20260926:" + event).encode()).hexdigest())
    val_events = sorted({row["event"] for row in metadata.values() if row["split"] == "val"})
    folds = [train_events[fold::3] for fold in range(3)]
    if any(not fold for fold in folds):
        raise ValueError("训练事件不足以三折留出")
    inputs = [args.photos, args.split_json, args.features, args.features.with_suffix(".json"), args.local,
              args.global_scores, Path(__file__), ROOT / "audit/evaluation_protocol.py", ROOT / "audit/split_eval.py"]
    input_hashes = {str(path.resolve()): digest(path) for path in inputs}
    args.out.mkdir(parents=True, exist_ok=False)
    plan = {"protocol": RATING_VERSION, "folds": folds, "penalties": [0.01, 0.1, 1.0, 10.0],
            "weighting": args.weighting,
            "classes": ["0", "1", "2", ">=3"], "decode_version": "display-plane-v3",
            "selection": "train 事件 OOF AP>=3；精确同分才比较四档平均召回；不打开 test",
            "reference_limit": "rank_sum 的 OOF 只留出校准头；底层历史排序器已经见过 train 标签",
            "prediction": "单图视觉头比名次和校准的四档平均召回提高>=0.03，AP>=3下降不超过0.01",
            "rejection": "事件差 bootstrap 跨零、AP>=3下降>0.01或只单事件有收益；不进长训练",
            "input_sha256": input_hashes}
    write_json(args.out / "plan.json", plan)
    truth = np.array([min(int(metadata[fingerprint]["rating_raw"]), 3) for fingerprint in fingerprints])
    event_of = np.array([metadata[fingerprint]["event"] for fingerprint in fingerprints])
    models = {}
    for name, feature in features.items():
        candidates = []
        for penalty in plan["penalties"]:
            out_of_fold, traces = {}, []
            for held_out in folds:
                fitting = np.isin(event_of, [event for event in train_events if event not in held_out])
                testing = np.flatnonzero(np.isin(event_of, held_out))
                coefficient, trace = fit_classifier(feature[fitting], truth[fitting], event_of[fitting].tolist(),
                                                     penalty, args.weighting)
                probabilities = predict(feature[testing], coefficient)
                out_of_fold.update((fingerprints[index], values.tolist()) for index, values in zip(testing, probabilities))
                traces.append({"held_out_events": held_out, **trace})
            metrics = measure(metadata, out_of_fold, train_events)
            record = {"features": name, "penalty": penalty, "metrics": metrics, "traces": traces}
            write_json(args.out / f"{name}_{penalty}_oof.json", record)
            candidates.append(record)
            print(f"{name} penalty={penalty} OOF AP>=3={metrics['ranking']['3']['ap_macro']:.4f} "
                  f"balanced_accuracy={metrics['rating']['balanced_accuracy_macro']:.4f}", flush=True)
        chosen = max(candidates, key=candidate_key)
        fitting = np.isin(event_of, train_events)
        coefficient, trace = fit_classifier(feature[fitting], truth[fitting], event_of[fitting].tolist(),
                                             chosen["penalty"], args.weighting)
        probabilities = predict(feature, coefficient)
        np.save(args.out / f"{name}_coefficient.npy", coefficient)
        write_scores(args.out / f"{name}_scores.csv", fingerprints, probabilities)
        scores = dict(zip(fingerprints, probabilities.tolist()))
        models[name] = {"penalty": chosen["penalty"], "oof": chosen["metrics"], "trace": trace,
                        "val": measure(metadata, scores, val_events)}
        print(f"{name} fixed penalty={chosen['penalty']} val AP>=3="
              f"{models[name]['val']['ranking']['3']['ap_macro']:.4f}", flush=True)
    chosen_name = max(("dino", "clip", "joint"), key=lambda name: candidate_key({"metrics": models[name]["oof"],
                                                                               "penalty": models[name]["penalty"]}))
    prior = np.mean([np.bincount(truth[event_of == event], minlength=4) / (event_of == event).sum()
                     for event in train_events], axis=0)
    models["train_prior"] = {"val": measure(metadata, dict.fromkeys(fingerprints, prior.tolist()), val_events)}
    comparisons = {name: {"val_balanced_accuracy_vs_rank_calibration": paired_difference(
        entry["val"], models["rank_sum"]["val"], "rating", "balanced_accuracy"),
        "val_ap_ge3_vs_rank_calibration": paired_difference(entry["val"], models["rank_sum"]["val"], "ranking", "ap")}
        for name, entry in models.items() if name in ("dino", "clip", "joint")}
    frozen_references = {name: {"ranking": ranking_metrics(metadata, scores, val_events),
                              "budget": summarize_budget(budget_metrics(metadata, scores, val_events))}
                         for name, scores in references.items()}
    result = {"plan": plan, "selected_by_train_oof": chosen_name, "models": models,
              "references": frozen_references, "comparisons": comparisons,
              "seconds": time.monotonic() - started, "scope": "v2 val 开发留出，非终考；本次未评 test"}
    if any(digest(path) != input_hashes[str(path.resolve())] for path in inputs):
        raise ValueError("运行期间输入发生变化")
    write_json(args.out / "result.json", result)
    (args.out / "report.md").write_text(render_report(result), encoding="utf-8")
    print(f"完成四档基线: {args.out}; {result['seconds']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
