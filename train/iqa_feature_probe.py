"""专用画质预训练的受控探针：冻结TOPIQ，个人四档监督只训练小头，train事件留出。

默认复用既有896张确定性小集，输入只用已核验修复PNG，不改原片或产品像素路径。
TOPIQ分数不是最终美学分；同时保留模型末层前的256维质量表征供个人监督学习。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "audit"))
from evaluation_protocol import RATING_VERSION, digest, read_rows, write_json
from rating_baseline import measure, paired_difference, write_scores
from technical_probe import transform_cv
from context_rating_probe import fit_oof, compare


def extract_quality(fingerprints: list[str], renders: Path, identity: dict, output: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """冻结TOPIQ在修复PNG上提取分数与256维末层前质量向量，记录图像/权重/实现版本哈希。"""
    import pyiqa
    import torch

    if not torch.cuda.is_available():
        raise ValueError("预登记GPU环境不可用")
    torch.manual_seed(20261014)
    torch.set_num_threads(2)
    model = pyiqa.create_metric("topiq_nr", device="cuda").eval()
    captured = []

    def capture_features(module, arguments):
        """只读捕获最终线性预测层输入，不改变TOPIQ模型输出或梯度。"""
        captured.append(arguments[0].detach().cpu().numpy())

    hook = model.net.score_linear[-1].register_forward_pre_hook(capture_features)
    weights = hashlib.sha256()
    for name, tensor in sorted(model.net.state_dict().items()):
        weights.update(name.encode())
        weights.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    score_parts, feature_parts, images = [], [], {}
    started = time.monotonic()
    try:
        with torch.inference_mode():
            for start in range(0, len(fingerprints), 8):
                members = fingerprints[start:start + 8]
                pixels = []
                for fingerprint in members:
                    path = renders / f"{fingerprint}.png"
                    image_hash = digest(path)
                    if image_hash != identity["images"][fingerprint]:
                        raise ValueError("修复PNG与已核验特征来源不一致")
                    with Image.open(path) as image:
                        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
                    if array.shape != (518, 518, 3):
                        raise ValueError("修复图像尺寸不是518")
                    pixels.append(array.transpose(2, 0, 1) / 255)
                    images[fingerprint] = image_hash
                captured.clear()
                scores = model(torch.from_numpy(np.stack(pixels)).to("cuda")).cpu().numpy().reshape(-1)
                if len(captured) != 1 or captured[0].shape != (len(members), 256):
                    raise ValueError("质量表征hook形状或次数错误")
                score_parts.append(scores)
                feature_parts.append(captured[0].copy())
                if start % 128 == 0 or start + len(members) == len(fingerprints):
                    print(f"TOPIQ [{start + len(members)}/{len(fingerprints)}] {time.monotonic() - started:.1f}s", flush=True)
    finally:
        hook.remove()
    scores, features = np.concatenate(score_parts).astype(np.float64), np.concatenate(feature_parts).astype(np.float64)
    if not np.isfinite(scores).all() or not np.isfinite(features).all():
        raise ValueError("TOPIQ输出包含非有限值")
    np.savez_compressed(output / "quality_features.npz", fingerprints=np.array(fingerprints), scores=scores, features=features)
    manifest = {"model": "topiq_nr", "decode_version": "display-plane-v3", "images": images,
        "state_sha256": weights.hexdigest(), "implementation_sha256": digest(Path(inspect.getfile(type(model.net)))),
        "torch": torch.__version__, "pyiqa": importlib.metadata.version("pyiqa"),
        "feature_hook": "CFANet.score_linear final Linear input,256 dimensions",
        "source_pixel_policy": "既有修复518PNG直接转0..1，不额外重采样；使用官方模型内部预处理",
        "cache_sha256": digest(output / "quality_features.npz")}
    write_json(output / "quality_features.json", manifest)
    return scores, features, manifest


def main() -> int:
    """固定样本/折/正则，比较视觉与新增IQA观测，仅在train事件评价，不打开val/test。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photos", type=Path, default=ROOT / "audit/out/m3_pairs_v2_20260924b/photos.csv")
    parser.add_argument("--features", type=Path, default=ROOT / "train/out/recovery_work/corrected_features.npz")
    parser.add_argument("--sources", type=Path, default=ROOT / "audit/out/technical_input_20261014/probe_sources.json")
    parser.add_argument("--renders", type=Path, default=Path("D:/PhotoDB/dataset/render518-display-plane-v3"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    threadpool_limits(2)
    started = time.monotonic()
    all_metadata = {row["fingerprint"]: row for row in read_rows(args.photos)}
    sources = json.loads(args.sources.read_text(encoding="utf-8"))
    fingerprints = sorted(row["Fingerprint"] for row in sources)
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("输入指纹重复")
    metadata = {fingerprint: all_metadata[fingerprint] for fingerprint in fingerprints}
    if any(row["split"] != "train" for row in metadata.values()):
        raise ValueError("IQA小集探针不允许val/test")
    events = sorted({row["event"] for row in metadata.values()},
                    key=lambda event: hashlib.sha256(("20260926:" + event).encode()).hexdigest())
    expected = set()
    for event in events:
        members = [fingerprint for fingerprint, row in all_metadata.items() if row["event"] == event]
        expected.update(sorted(members, key=lambda fingerprint:
            hashlib.sha256(("cv-feasibility-20261014:" + fingerprint).encode()).hexdigest())[:64])
    if expected != set(fingerprints) or len(events) != 14:
        raise ValueError("IQA样本偏离固定14事件抽样")
    identity = json.loads(args.features.with_suffix(".json").read_text(encoding="utf-8"))
    if identity["decode_version"] != "display-plane-v3" or identity["metadata_sha256"] != digest(args.photos):
        raise ValueError("视觉输入身份错误")
    with np.load(args.features, allow_pickle=False) as cached:
        index_of = {fingerprint: index for index, fingerprint in enumerate(cached["fingerprints"].tolist())}
        indexes = [index_of[fingerprint] for fingerprint in fingerprints]
        dino, clip = cached["dino"][indexes].astype(np.float64), cached["features"][indexes].astype(np.float64)
    visual = np.concatenate((dino * np.sqrt(384), clip * np.sqrt(768)), axis=1) / np.sqrt(2)
    folds = [events[index::3] for index in range(3)]
    paths = [args.photos, args.features, args.features.with_suffix(".json"), args.sources, Path(__file__),
             ROOT / "train/rating_baseline.py", ROOT / "train/context_rating_probe.py", ROOT / "train/technical_probe.py",
             ROOT / "audit/evaluation_protocol.py", ROOT / "audit/split_eval.py"]
    plan = {"protocol": RATING_VERSION, "folds": folds, "penalty": 0.1, "weighting": "event_class",
            "scope": "固定train小集896张，IQA预训练新增观测；不打开val/test；非完整事件成绩",
            "model": "topiq_nr", "sample_policy": "每train事件64张，固定SHA，与技术探针相同",
            "input_sha256": {str(path.resolve()): digest(path) for path in paths}}
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "plan.json", plan)
    scores, quality_features, quality_identity = extract_quality(fingerprints, args.renders, identity, args.out)
    normalized = quality_features / np.linalg.norm(quality_features, axis=1, keepdims=True) * np.sqrt(256)
    arrays = {"visual": visual, "visual_iqa_score": np.column_stack((visual, scores)),
              "visual_iqa_embedding": np.concatenate((visual, normalized), axis=1)}
    event_of = np.array([metadata[fingerprint]["event"] for fingerprint in fingerprints])
    models = {}
    for name, array in arrays.items():
        if name == "visual_iqa_score":
            from rating_baseline import fit_classifier, predict
            probabilities, traces = {}, []
            truth = np.array([min(int(metadata[fingerprint]["rating_raw"]), 3) for fingerprint in fingerprints])
            for held_out in folds:
                fitting = np.isin(event_of, sorted(set(events) - set(held_out)))
                testing = np.flatnonzero(np.isin(event_of, held_out))
                transformed, transformation = transform_cv(scores[fitting, None], scores[testing, None])
                train_quality = (scores[fitting, None] - np.array(transformation["center"])) / np.array(transformation["scale"])
                coefficient, trace = fit_classifier(np.column_stack((visual[fitting], train_quality)),
                    truth[fitting], event_of[fitting].tolist(), 0.1, "event_class")
                values = predict(np.column_stack((visual[testing], transformed)), coefficient)
                probabilities.update((fingerprints[index], value.tolist()) for index, value in zip(testing, values))
                traces.append({"held_out_events": held_out, "quality_transform": transformation, **trace})
        else:
            probabilities, traces = fit_oof(array, metadata, fingerprints, folds)
        metrics = measure(metadata, probabilities, events)
        models[name] = {"metrics": metrics, "traces": traces}
        write_scores(args.out / f"{name}_oof_scores.csv", fingerprints,
                     np.array([probabilities[fingerprint] for fingerprint in fingerprints]))
        print(f"{name} AP>=1={metrics['ranking']['1']['ap_macro']:.4f} "
              f"AP>=3={metrics['ranking']['3']['ap_macro']:.4f} "
              f"balanced_accuracy={metrics['rating']['balanced_accuracy_macro']:.4f}", flush=True)
    difference = {name: compare(model["metrics"], models["visual"]["metrics"])
                  for name, model in models.items() if name != "visual"}
    raw_quality = dict(zip(fingerprints, scores.tolist()))
    from evaluation_protocol import ranking_metrics
    result = {"plan": plan, "quality_identity": quality_identity, "models": models, "difference": difference,
              "zero_shot_quality_ranking": ranking_metrics(metadata, raw_quality, events),
              "seconds": time.monotonic() - started}
    if any(digest(path) != plan["input_sha256"][str(path.resolve())] for path in paths):
        raise ValueError("执行期间输入发生变化")
    write_json(args.out / "result.json", result)
    lines = ["# 冻结专用画质观测", "", plan["scope"], "同样本/折/正则/损失，只改变质量输入；预训练分数不当最终美学分。", "",
             "| 小集train OOF | AP≥1 | AP≥2 | AP≥3 | 随机AP≥3 | 上限 | 四档平均召回 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, model in models.items():
        metrics, ranking = model["metrics"], model["metrics"]["ranking"]
        lines.append(f"| {name} | {ranking['1']['ap_macro']:.4f} | {ranking['2']['ap_macro']:.4f} | "
                     f"{ranking['3']['ap_macro']:.4f} | {ranking['3']['random_ap_macro']:.4f} | 1.000 | "
                     f"{metrics['rating']['balanced_accuracy_macro']:.4f} |")
    lines.extend(["", "四档随机参照0.25，上限1.0；逐事件、各档指标和配对bootstrap见result.json。",
                  "这是518输入的可行性探针，不替代全分辨率技术判别，也不能与完整事件仪表盘直接比较。"])
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"完成IQA观测探针: {result['seconds']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
