"""无标签相似邻域四档探针：完整训练事件OOF，容量匹配的随机邻域对照。

同一评分头输入本片视觉与相对邻域残差；不把人工时间线或每团一张当目标。
邻域不读标签、团顶、时间或模型分数，推理需要当前完整事件的无标签视觉输入。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "audit"))
from evaluation_protocol import RATING_VERSION, digest, read_rows, write_json
from rating_baseline import fit_classifier, predict, measure, paired_difference, write_scores


def load_inputs(args) -> tuple[dict, list[str], np.ndarray, np.ndarray]:
    """核验修复特征/元数据哈希和完整覆盖，返回视觉特征与只用于检索的归一化DINO。"""
    rows = read_rows(args.photos)
    metadata = {row["fingerprint"]: row for row in rows}
    if len(metadata) != len(rows):
        raise ValueError("元数据指纹重复")
    split = json.loads(args.split_json.read_text(encoding="utf-8"))
    events = [event for name in ("train", "val", "test") for event in split[name]]
    if len(events) != len(set(events)) or set(events) != {row["event"] for row in rows}:
        raise ValueError("事件划分重叠或缺失")
    if any(row["event"] not in split[row["split"]] or int(row["rating_raw"]) not in range(6) for row in rows):
        raise ValueError("元数据split或星级错误")
    identity = json.loads(args.features.with_suffix(".json").read_text(encoding="utf-8"))
    if identity["decode_version"] != "display-plane-v3" or identity["metadata_sha256"] != digest(args.photos):
        raise ValueError("修复视觉缓存身份不符")
    with np.load(args.features, allow_pickle=False) as cached:
        fingerprints = cached["fingerprints"].tolist()
        dino, clip = cached["dino"].astype(np.float64), cached["features"].astype(np.float64)
    if fingerprints != sorted(metadata) or set(identity["images"]) != set(fingerprints):
        raise ValueError("特征覆盖或顺序不一致")
    for feature, dimension in ((dino, 384), (clip, 768)):
        if feature.shape != (len(fingerprints), dimension) or not np.isfinite(feature).all():
            raise ValueError("特征维度或有限性错误")
        if not np.allclose(np.linalg.norm(feature, axis=1), 1.0, atol=1e-4):
            raise ValueError("检索特征未L2归一化")
    visual = np.concatenate((dino * np.sqrt(384), clip * np.sqrt(768)), axis=1) / np.sqrt(2)
    return metadata, fingerprints, visual, dino


def contextual_features(visual: np.ndarray, dino: np.ndarray, fingerprints: list[str],
                        event_of: np.ndarray, random_neighbors: bool) -> tuple[np.ndarray, dict]:
    """从同事件无标签视觉取8个邻居，返回本片+残差+关系统计；随机臂与相似臂容量一致。"""
    output = np.empty((len(visual), visual.shape[1] * 2 + 6), dtype=np.float64)
    neighbor_identities = {}
    for event in sorted(set(event_of.tolist())):
        indexes = np.flatnonzero(event_of == event)
        if len(indexes) < 9:
            raise ValueError("事件不足9张，无法执行预登记8邻居")
        similarity = dino[indexes] @ dino[indexes].T
        np.fill_diagonal(similarity, -np.inf)
        if random_neighbors:
            generator = np.random.default_rng(int(hashlib.sha256(("context-20261014:" + event).encode()).hexdigest()[:8], 16))
            neighbors = np.empty((len(indexes), 8), dtype=np.int64)
            for row in range(len(indexes)):
                candidates = np.delete(np.arange(len(indexes)), row)
                neighbors[row] = generator.choice(candidates, 8, replace=False)
        else:
            neighbors = np.argsort(-similarity, axis=1, kind="stable")[:, :8]
        selected_similarity = np.take_along_axis(similarity, neighbors, axis=1)
        weights = softmax(selected_similarity * 10, axis=1)
        global_neighbors = indexes[neighbors]
        if np.any(global_neighbors == indexes[:, None]) or any(event_of[index] != event for index in global_neighbors.ravel()):
            raise ValueError("邻居含本片或越过事件")
        center = np.sum(visual[global_neighbors] * weights[:, :, None], axis=1)
        residual = visual[indexes] - center
        dino_residual = np.linalg.norm(residual[:, :384], axis=1) / np.sqrt(384 / 2)
        clip_residual = np.linalg.norm(residual[:, 384:], axis=1) / np.sqrt(768 / 2)
        relationship = np.column_stack((selected_similarity.max(axis=1), selected_similarity.mean(axis=1),
            selected_similarity.min(axis=1), selected_similarity.std(axis=1), dino_residual, clip_residual))
        output[indexes] = np.concatenate((visual[indexes], residual, relationship), axis=1)
        neighbor_identities[event] = hashlib.sha256("\n".join(
            fingerprints[index] + ":" + ",".join(fingerprints[neighbor] for neighbor in member_neighbors)
            for index, member_neighbors in zip(indexes, global_neighbors)).encode()).hexdigest()
    if not np.isfinite(output).all():
        raise ValueError("上下文特征非有限")
    return output, {"neighbors": 8, "temperature": 10, "random_neighbors": random_neighbors,
                    "features": output.shape[1], "neighbor_sha256_by_event": neighbor_identities,
                    "gate": "同事件、排除本片、无标签/时间/团顶/模型分数"}


def fit_oof(features: np.ndarray, metadata: dict, fingerprints: list[str], folds: list[list[str]]) -> tuple[dict, list[dict]]:
    """固定0.1/event_class执行训练事件三折，返回每张留出概率与收敛记录。"""
    event_of = np.array([metadata[fingerprint]["event"] for fingerprint in fingerprints])
    truth = np.array([min(int(metadata[fingerprint]["rating_raw"]), 3) for fingerprint in fingerprints])
    training_events = {event for fold in folds for event in fold}
    probabilities, traces = {}, []
    for held_out in folds:
        fitting = np.isin(event_of, sorted(training_events - set(held_out)))
        testing = np.flatnonzero(np.isin(event_of, held_out))
        coefficient, trace = fit_classifier(features[fitting], truth[fitting], event_of[fitting].tolist(), 0.1, "event_class")
        values = predict(features[testing], coefficient)
        probabilities.update((fingerprints[index], value.tolist()) for index, value in zip(testing, values))
        traces.append({"held_out_events": held_out, **trace})
    return probabilities, traces


def compare(candidate: dict, reference: dict) -> dict:
    """对相同完整事件比较三个边界AP与四档平均召回，固定事件bootstrap，不参与参数搜索。"""
    result = {"balanced_accuracy": paired_difference(candidate, reference, "rating", "balanced_accuracy")}
    for threshold in (1, 2, 3):
        left = {"ranking": {"3": candidate["ranking"][str(threshold)]}}
        right = {"ranking": {"3": reference["ranking"][str(threshold)]}}
        result[f"ap_ge{threshold}"] = paired_difference(left, right, "ranking", "ap")
    return result


def main() -> int:
    """先跑train完整事件容量对照，满足预登记门才允许显式--evaluate-val固定训练；始终不打开test。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photos", type=Path, default=ROOT / "audit/out/m3_pairs_v2_20260924b/photos.csv")
    parser.add_argument("--split-json", type=Path, default=ROOT / "data/split-v2-20260924.json")
    parser.add_argument("--features", type=Path, default=ROOT / "train/out/recovery_work/corrected_features.npz")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evaluate-val", action="store_true")
    args = parser.parse_args()
    threadpool_limits(2)
    started = time.monotonic()
    metadata, fingerprints, visual, dino = load_inputs(args)
    event_of = np.array([metadata[fingerprint]["event"] for fingerprint in fingerprints])
    train_events = sorted({row["event"] for row in metadata.values() if row["split"] == "train"},
                          key=lambda event: hashlib.sha256(("20260926:" + event).encode()).hexdigest())
    folds = [train_events[index::3] for index in range(3)]
    inputs = [args.photos, args.split_json, args.features, args.features.with_suffix(".json"), Path(__file__),
              ROOT / "train/rating_baseline.py", ROOT / "audit/evaluation_protocol.py", ROOT / "audit/split_eval.py"]
    plan = {"protocol": RATING_VERSION, "folds": folds, "penalty": 0.1, "weighting": "event_class",
            "neighbors": 8, "temperature": 10, "scope": "完整train事件OOF；上下文只用同事件无标签输入；不评test",
            "prediction": "相似邻域相对单图AP>=3和四档平均召回各+0.02；优于容量匹配随机邻域",
            "input_sha256": {str(path.resolve()): digest(path) for path in inputs}}
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "plan.json", plan)
    train_indexes = np.flatnonzero(np.isin(event_of, train_events))
    train_fingerprints = [fingerprints[index] for index in train_indexes]
    train_metadata = {fingerprint: metadata[fingerprint] for fingerprint in train_fingerprints}
    arrays = {"single": visual[train_indexes]}
    identities = {}
    for name, random_neighbors in (("similar_context", False), ("random_context", True)):
        arrays[name], identities[name] = contextual_features(visual[train_indexes], dino[train_indexes],
            train_fingerprints, event_of[train_indexes], random_neighbors)
    models = {}
    for name, features in arrays.items():
        probabilities, traces = fit_oof(features, train_metadata, train_fingerprints, folds)
        metrics = measure(train_metadata, probabilities, train_events)
        models[name] = {"metrics": metrics, "traces": traces}
        write_scores(args.out / f"{name}_oof_scores.csv", train_fingerprints,
                     np.array([probabilities[fingerprint] for fingerprint in train_fingerprints]))
        print(f"{name} train OOF AP>=3={metrics['ranking']['3']['ap_macro']:.4f} "
              f"balanced_accuracy={metrics['rating']['balanced_accuracy_macro']:.4f}", flush=True)
    difference = {"similar_vs_single": compare(models["similar_context"]["metrics"], models["single"]["metrics"]),
                  "similar_vs_random": compare(models["similar_context"]["metrics"], models["random_context"]["metrics"])}
    gain = difference["similar_vs_single"]
    gate = (gain["ap_ge3"]["mean"] >= 0.02 and gain["ap_ge3"]["event_bootstrap95"][0] > 0
            and gain["balanced_accuracy"]["mean"] >= 0.02
            and difference["similar_vs_random"]["ap_ge3"]["mean"] > 0)
    result = {"plan": plan, "models": models, "context_identity": identities,
              "difference": difference, "train_gate": bool(gate)}
    if args.evaluate_val:
        if not gate:
            raise ValueError("预登记train闸未通过，不得据val寻找有效参数")
        truth = np.array([min(int(metadata[fingerprint]["rating_raw"]), 3) for fingerprint in fingerprints])
        fitting = np.isin(event_of, train_events)
        val_events = sorted({row["event"] for row in metadata.values() if row["split"] == "val"})
        requested = np.flatnonzero(np.isin(event_of, train_events + val_events))
        selected_fingerprints = [fingerprints[index] for index in requested]
        features, context_identity = contextual_features(visual[requested], dino[requested],
            selected_fingerprints, event_of[requested], False)
        coefficient, trace = fit_classifier(features[np.isin(event_of[requested], train_events)],
            truth[fitting], event_of[fitting].tolist(), 0.1, "event_class")
        probabilities = predict(features, coefficient)
        np.save(args.out / "similar_context_coefficient.npy", coefficient)
        write_scores(args.out / "similar_context_scores.csv", selected_fingerprints, probabilities)
        result["val"] = {"metrics": measure(metadata, dict(zip(selected_fingerprints, probabilities.tolist())), val_events),
                         "trace": trace, "context_identity": context_identity}
    if any(digest(path) != plan["input_sha256"][str(path.resolve())] for path in inputs):
        raise ValueError("执行期间输入漂移")
    result["seconds"] = time.monotonic() - started
    write_json(args.out / "result.json", result)
    lines = ["# 无标签视觉邻域上下文", "", plan["scope"], "完整事件分母；单头四档；参数固定；无邻居标签。", "",
             "| train事件OOF | AP≥3 | 随机AP | 上限 | 四档平均召回 |", "|---|---:|---:|---:|---:|"]
    for name, model in models.items():
        metrics = model["metrics"]
        ranking = metrics["ranking"]["3"]
        lines.append(f"| {name} | {ranking['ap_macro']:.4f} | {ranking['random_ap_macro']:.4f} | 1.000 | "
                     f"{metrics['rating']['balanced_accuracy_macro']:.4f} |")
    lines.extend(["", f"预登记train闸：{gate}；四档随机参照0.25、上限1.0；逐事件与各档明细见result.json。",
                  "训练事件OOF说明方向，不能替代独立新事件验收。"])
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"完成上下文探针: {args.out}; train_gate={gate}; {result['seconds']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
