"""训练事件小集的 CV 增量可行性探针，固定视觉基线与损失，只改变有效技术输入。

同照片事件级三折 OOF，既不打开 val/test，也不把小集指标当完整事件成绩。
CV 聚合只是信息可读性探针，不实现“选最锐”规则，不声称替代语义分块融合。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "audit"))
from evaluation_protocol import RATING_VERSION, digest, read_rows, write_json
from rating_baseline import fit_classifier, predict, measure, paired_difference, write_scores


def robust_statistics(values: np.ndarray, include_quantiles: bool) -> list[float]:
    """聚合有效格读数并显式携带缺失率；全缺失只给零占位与缺失率一，不假装技术质量为零。"""
    finite = values[np.isfinite(values)]
    levels = [0.1, 0.25, 0.5, 0.75, 0.9] if include_quantiles else [0.5]
    quantiles = np.quantile(finite, levels).tolist() if finite.size else [0.0] * len(levels)
    return [*quantiles, 1 - finite.size / values.size]


def grid_features(grid: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """将修复网格变为固定全局分位数和4×4粗格读数；角度使用双角正余弦，缺失率单独保留。"""
    if grid.shape != (7, 32, 32) or np.isinf(grid).any():
        raise ValueError("非法 CV 网格")
    for index in (0, 1, 2, 3, 5, 6):
        if np.any(grid[index][np.isfinite(grid[index])] < 0):
            raise ValueError("CV 读数范围错误")
    planes = {"edge_count_log": np.log1p(grid[0]), "sharp_width_log": np.log1p(grid[1]),
              "median_width_log": np.log1p(grid[2]), "drag_width_log": np.log1p(grid[3]),
              "drag_cos2": np.cos(2 * grid[4]), "drag_sin2": np.sin(2 * grid[4]),
              "anisotropy": grid[5], "contrast": grid[6] / 255}
    values, names = [], []
    for name, plane in planes.items():
        statistics = robust_statistics(plane, True)
        values.extend(statistics)
        names.extend(f"{name}_global_{statistic}" for statistic in ("q10", "q25", "q50", "q75", "q90", "missing"))
        for row in range(4):
            for column in range(4):
                block = plane[row * 8:(row + 1) * 8, column * 8:(column + 1) * 8]
                values.extend(robust_statistics(block, False))
                names.extend(f"{name}_tile{row}_{column}_{statistic}" for statistic in ("median", "missing"))
    return np.array(values, dtype=np.float64), names


def load_inputs(args) -> tuple[dict, list[str], np.ndarray, np.ndarray, dict]:
    """核验确定性 train 抽样、技术缓存完整性及身份/输出哈希，返回同照片视觉/CV 输入。"""
    rows = read_rows(args.photos)
    metadata = {row["fingerprint"]: row for row in rows}
    sources = json.loads(args.sources.read_text(encoding="utf-8"))
    selected = {row["Fingerprint"]: row for row in sources}
    if len(selected) != len(sources):
        raise ValueError("来源指纹重复")
    identity = json.loads((args.cache / "identity.json").read_text(encoding="utf-8"))
    complete = json.loads((args.cache / "complete.json").read_text(encoding="utf-8"))
    if identity["decode_version"] != "display-plane-v3" or identity["cv_spec"] != "cv_grid_v5_contrast+decode-display-plane-v3":
        raise ValueError("技术输入版本错误")
    if identity["grid_size"] != 32 or len(identity["scalar_names"]) != 7 or identity["save_patches"]:
        raise ValueError("技术输入契约错误")
    if identity["sources_sha256"] != digest(args.sources) or identity["requested"] != len(sources):
        raise ValueError("来源清单或请求量漂移")
    if complete["completed"] != len(sources) or complete["identity_sha256"] != digest(args.cache / "identity.json"):
        raise ValueError("技术提取未完整通过")
    if len(metadata) != len(rows):
        raise ValueError("照片元数据有重复指纹")
    by_event = defaultdict(list)
    for fingerprint, row in metadata.items():
        if row["split"] == "train":
            by_event[row["event"]].append(fingerprint)
    expected = set()
    for fingerprints in by_event.values():
        expected.update(sorted(fingerprints, key=lambda fingerprint:
            hashlib.sha256(("cv-feasibility-20261014:" + fingerprint).encode()).hexdigest())[:64])
    if expected != set(selected):
        raise ValueError("可行性抽样偏离预登记规则")
    subset = {fingerprint: metadata[fingerprint] for fingerprint in sorted(selected)}
    cache_identity = json.loads(args.features.with_suffix(".json").read_text(encoding="utf-8"))
    if cache_identity["decode_version"] != "display-plane-v3" or cache_identity["metadata_sha256"] != digest(args.photos):
        raise ValueError("视觉缓存身份错误")
    with np.load(args.features, allow_pickle=False) as archive:
        all_fingerprints = archive["fingerprints"].tolist()
        index_of = {fingerprint: index for index, fingerprint in enumerate(all_fingerprints)}
        if len(index_of) != len(all_fingerprints) or set(index_of) != set(metadata):
            raise ValueError("视觉缓存覆盖错误")
        indexes = [index_of[fingerprint] for fingerprint in subset]
        dino = archive["dino"][indexes].astype(np.float64)
        clip = archive["features"][indexes].astype(np.float64)
    for feature, dimension in ((dino, 384), (clip, 768)):
        if feature.shape != (len(subset), dimension) or not np.isfinite(feature).all():
            raise ValueError("视觉形状或有限性错误")
        if not np.allclose(np.linalg.norm(feature, axis=1), 1, atol=1e-4):
            raise ValueError("视觉特征未逐张归一化")
    visual = np.concatenate((dino * np.sqrt(384), clip * np.sqrt(768)), axis=1) / np.sqrt(2)
    grids, hashes, feature_names = [], {}, []
    for fingerprint in subset:
        record_path = args.cache / f"{fingerprint}.json"
        grid_path = args.cache / f"{fingerprint}.cv.bin"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record["fingerprint"] != fingerprint or record["source_path"] != str(Path(selected[fingerprint]["Path"]).resolve()):
            raise ValueError("技术缓存来源错误")
        if digest(Path(record["source_path"])) != record["source_sha256"] or digest(grid_path) != record["cv_sha256"]:
            raise ValueError("技术输入哈希漂移")
        values = np.fromfile(grid_path, dtype="<f4").reshape(7, 32, 32)
        features, feature_names = grid_features(values)
        grids.append(features)
        hashes[fingerprint] = {"record_sha256": digest(record_path), "cv_sha256": record["cv_sha256"],
                               "source_sha256": record["source_sha256"]}
    return subset, list(subset), visual, np.stack(grids), {"identity": identity,
        "cv_feature_names": feature_names, "samples": hashes}


def transform_cv(training: np.ndarray, testing: np.ndarray) -> tuple[np.ndarray, dict]:
    """仅用拟合折估计 CV 中心/尺度并应用到测试折；常量维度尺度为一，返回真实变换记录。"""
    center = training.mean(axis=0)
    scale = training.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    normalized = (testing - center) / scale
    if not np.isfinite(normalized).all():
        raise ValueError("CV 标准化后含非有限值")
    return normalized, {"center": center.tolist(), "scale": scale.tolist()}


def run_arm(visual: np.ndarray, technical: np.ndarray, truth: np.ndarray, event_of: np.ndarray,
            fingerprints: list[str], folds: list[list[str]], use_cv: bool) -> tuple[dict, list[dict]]:
    """固定正则0.1执行事件三折，返回每张真实 OOF 概率；只改变 CV 输入。"""
    probabilities, traces = {}, []
    all_events = sorted(set(event_of.tolist()))
    for held_out in folds:
        fitting = np.isin(event_of, [event for event in all_events if event not in held_out])
        testing = np.flatnonzero(np.isin(event_of, held_out))
        train_feature, test_feature = visual[fitting], visual[testing]
        transformation = None
        if use_cv:
            train_cv = technical[fitting].copy()
            test_cv = technical[testing].copy()
            test_cv, transformation = transform_cv(train_cv, test_cv)
            train_cv = (train_cv - np.array(transformation["center"])) / np.array(transformation["scale"])
            train_feature = np.concatenate((train_feature, train_cv), axis=1)
            test_feature = np.concatenate((test_feature, test_cv), axis=1)
        coefficient, trace = fit_classifier(train_feature, truth[fitting], event_of[fitting].tolist(), 0.1, "event_class")
        prediction = predict(test_feature, coefficient)
        probabilities.update((fingerprints[index], values.tolist()) for index, values in zip(testing, prediction))
        traces.append({"held_out_events": held_out, "transform": transformation, **trace})
    return probabilities, traces


def main() -> int:
    """运行预登记固定小集双臂，仅报告训练事件 OOF，输出概率、哈希与逐事件差异。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--photos", type=Path, default=ROOT / "audit/out/m3_pairs_v2_20260924b/photos.csv")
    parser.add_argument("--features", type=Path, default=ROOT / "train/out/recovery_work/corrected_features.npz")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    threadpool_limits(2)
    started = time.monotonic()
    metadata, fingerprints, visual, technical, identity = load_inputs(args)
    events = sorted({row["event"] for row in metadata.values()},
                    key=lambda event: hashlib.sha256(("20260926:" + event).encode()).hexdigest())
    folds = [events[fold::3] for fold in range(3)]
    truth = np.array([min(int(metadata[fingerprint]["rating_raw"]), 3) for fingerprint in fingerprints])
    event_of = np.array([metadata[fingerprint]["event"] for fingerprint in fingerprints])
    inputs = [args.sources, args.photos, args.features, args.features.with_suffix(".json"), args.cache / "identity.json",
              args.cache / "complete.json", Path(__file__), ROOT / "train/rating_baseline.py",
              ROOT / "audit/evaluation_protocol.py", ROOT / "audit/split_eval.py"]
    plan = {"protocol": RATING_VERSION, "scope": "train事件确定性小集可行性；不是完整事件成绩；不打开val/test",
            "folds": folds, "photos": len(fingerprints), "per_event": 64,
            "penalty": 0.1, "weighting": "event_class", "technical_identity": identity,
            "prediction": "CV让AP>=1和四档平均召回各提高>=0.02，AP>=3降幅<=0.01",
            "input_sha256": {str(path.resolve()): digest(path) for path in inputs}}
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "plan.json", plan)
    models = {}
    for name, use_cv in (("visual", False), ("visual_cv", True)):
        probabilities, traces = run_arm(visual, technical, truth, event_of, fingerprints, folds, use_cv)
        metrics = measure(metadata, probabilities, events)
        models[name] = {"metrics": metrics, "traces": traces}
        write_scores(args.out / f"{name}_oof_scores.csv", fingerprints,
                     np.array([probabilities[fingerprint] for fingerprint in fingerprints]))
        print(f"{name} OOF AP>=1={metrics['ranking']['1']['ap_macro']:.4f}, "
              f"AP>=3={metrics['ranking']['3']['ap_macro']:.4f}, "
              f"balanced_accuracy={metrics['rating']['balanced_accuracy_macro']:.4f}", flush=True)
    before, after = models["visual"]["metrics"], models["visual_cv"]["metrics"]
    difference = {"ap_ge3": paired_difference(after, before, "ranking", "ap"),
                  "balanced_accuracy": paired_difference(after, before, "rating", "balanced_accuracy")}
    for threshold in (1, 2):
        comparison_after = {"ranking": {"3": after["ranking"][str(threshold)]}}
        comparison_before = {"ranking": {"3": before["ranking"][str(threshold)]}}
        difference[f"ap_ge{threshold}"] = paired_difference(comparison_after, comparison_before, "ranking", "ap")
    result = {"plan": plan, "models": models, "difference": difference, "seconds": time.monotonic() - started}
    if any(digest(path) != plan["input_sha256"][str(path.resolve())] for path in inputs):
        raise ValueError("执行期间输入发生变化")
    write_json(args.out / "result.json", result)
    lines = ["# 修复 CV 增量可行性探针", "", plan["scope"],
             "每 train 事件确定性抽64张；事件三折；同照片、固定正则0.1、类别均衡分类头。",
             "CV只用修复全分辨率提取，不把低星当抖动原因，不使用旧patch或EXIF参数。", "",
             "| OOF小集 | AP≥1 | AP≥2 | AP≥3 | 随机AP≥3 | 上限 | 四档平均召回 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, model in models.items():
        metrics = model["metrics"]
        ranking = metrics["ranking"]
        lines.append(f"| {name} | {ranking['1']['ap_macro']:.4f} | {ranking['2']['ap_macro']:.4f} | "
                     f"{ranking['3']['ap_macro']:.4f} | {ranking['3']['random_ap_macro']:.4f} | 1.000 | "
                     f"{metrics['rating']['balanced_accuracy_macro']:.4f} |")
    lines.extend(["", "四档平均召回随机参照0.25、上限1.0；缺口闭合与各档召回/精度/混淆矩阵见result.json。", "",
                  "| CV增量指标 | 平均差 | 事件bootstrap95% |", "|---|---:|---|"])
    for name, comparison in difference.items():
        lines.append(f"| {name} | {comparison['mean']:.4f} | {comparison['event_bootstrap95']} |")
    lines.extend(["", "该小集只能判断是否值得下一步，不能作为新模型验收或与全事件仪表盘直接比较。",
                  "负结果只约束此CV聚合与线性头，不外推为语义空间融合无效。"])
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"完成可行性探针: {args.out}; {result['seconds']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
