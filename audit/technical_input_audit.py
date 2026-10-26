"""对修复技术侧缓存作只读核验：渲染/CLS 口径、旧 CV/patch 差异和网格几何。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image

from evaluation_protocol import digest, read_rows, write_json

ROOT = Path(__file__).resolve().parents[1]


def compare_arrays(current: np.ndarray, previous: np.ndarray) -> dict:
    """逐标量统计有限值差异与 NaN 掩码差异；缺读数不伪造成零差异。"""
    if current.shape != previous.shape or np.isinf(current).any() or np.isinf(previous).any():
        raise ValueError("网格形状或有限性错误")
    shared = np.isfinite(current) & np.isfinite(previous)
    difference = np.abs(current[shared] - previous[shared])
    return {"shared_finite": int(shared.sum()),
            "nan_mask_difference": int((np.isnan(current) != np.isnan(previous)).sum()),
            "mean_abs_difference": float(difference.mean()) if difference.size else None,
            "max_abs_difference": float(difference.max()) if difference.size else None,
            "exact_equal": bool(np.array_equal(current, previous, equal_nan=True))}


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    """计算有限非零向量的余弦，形状不匹配或退化时直接拒绝。"""
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("余弦向量形状或有限性错误")
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if not denominator:
        raise ValueError("余弦向量全零")
    return float(np.dot(left, right) / denominator)


def main() -> int:
    """核验完整提取样本的输出哈希与来源，写出带版本和硬闸的真实审计报告。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--features", type=Path, default=ROOT / "train/out/recovery_work/corrected_features.npz")
    parser.add_argument("--renders", type=Path, default=Path("D:/PhotoDB/dataset/render518-display-plane-v3"))
    parser.add_argument("--photos", type=Path, default=ROOT / "audit/out/m3_pairs_v2_20260924b/photos.csv")
    parser.add_argument("--db", type=Path, default=Path("D:/PhotoDB/dataset/photos_dataset.db"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    identity = json.loads((args.cache / "identity.json").read_text(encoding="utf-8"))
    complete = json.loads((args.cache / "complete.json").read_text(encoding="utf-8"))
    sources = json.loads(args.sources.read_text(encoding="utf-8"))
    if identity["decode_version"] != "display-plane-v3" or not identity["save_patches"] or not identity["save_renders"]:
        raise ValueError("审计缓存版本或输出选项不一致")
    if complete["completed"] != len(sources) or complete["identity_sha256"] != digest(args.cache / "identity.json"):
        raise ValueError("缓存提取未完整完成")
    if identity["sources_sha256"] != digest(args.sources):
        raise ValueError("来源清单身份漂移")
    metadata = {row["fingerprint"]: row for row in read_rows(args.photos)}
    feature_identity = json.loads(args.features.with_suffix(".json").read_text(encoding="utf-8"))
    if feature_identity["decode_version"] != identity["decode_version"] or feature_identity["metadata_sha256"] != digest(args.photos):
        raise ValueError("视觉缓存与元数据身份不一致")
    with np.load(args.features, allow_pickle=False) as archive:
        index_of = {fingerprint: index for index, fingerprint in enumerate(archive["fingerprints"].tolist())}
        features = archive["dino"].astype(np.float64)
    connection = sqlite3.connect(args.db.resolve().as_uri() + "?mode=ro", uri=True)
    details = []
    try:
        for source in sources:
            fingerprint = source["Fingerprint"]
            if metadata[fingerprint]["split"] != "train":
                raise ValueError("核验样本不属于预登记的 train")
            record_path = args.cache / f"{fingerprint}.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if digest(Path(source["Path"])) != record["source_sha256"]:
                raise ValueError("原片来源发生变化")
            for suffix, key in (("cv.bin", "cv_sha256"), ("cls.bin", "cls_sha256"),
                                ("patch.bin", "patches_sha256"), ("png", "render_sha256")):
                if digest(args.cache / f"{fingerprint}.{suffix}") != record[key]:
                    raise ValueError("缓存输出哈希漂移")
            row = connection.execute("SELECT cv_grid,cv_image_width,cv_image_height,cv_grid_spec FROM photos WHERE fingerprint=?",
                                     (fingerprint,)).fetchone()
            if row is None or row[0] is None:
                raise ValueError("核验样本缺少旧网格")
            current = np.fromfile(args.cache / f"{fingerprint}.cv.bin", dtype="<f4").reshape(7, 32, 32)
            previous = np.frombuffer(row[0], dtype="<f4").reshape(7, 32, 32)
            cls = np.fromfile(args.cache / f"{fingerprint}.cls.bin", dtype="<f4").astype(np.float64)
            patch = np.fromfile(args.cache / f"{fingerprint}.patch.bin", dtype="<f4").reshape(1024, 384)
            old_patch_row = connection.execute("SELECT patch_tokens FROM photo_patches WHERE fingerprint=? AND model_id=?",
                (fingerprint, "dinov3_vits16_f32_518_v1")).fetchone()
            if old_patch_row is None:
                raise ValueError("核验样本缺少旧 patch")
            old_patch = np.frombuffer(old_patch_row[0], dtype="<f4").reshape(1024, 384)
            with Image.open(args.cache / f"{fingerprint}.png") as image:
                current_pixels = np.asarray(image.convert("RGB"), dtype=np.int16)
            with Image.open(args.renders / f"{fingerprint}.png") as image:
                previous_pixels = np.asarray(image.convert("RGB"), dtype=np.int16)
            if current_pixels.shape != previous_pixels.shape:
                raise ValueError("渲染像素尺寸不一致")
            pixel_difference = np.abs(current_pixels - previous_pixels)
            patch_cosines = np.sum(patch * old_patch, axis=1) / (np.linalg.norm(patch, axis=1) * np.linalg.norm(old_patch, axis=1))
            details.append({"fingerprint": fingerprint, "event": metadata[fingerprint]["event"],
                "format": Path(source["Path"]).suffix.lower(), "current_dimensions": [record["width"], record["height"]],
                "old_dimensions": [row[1], row[2]], "old_cv_spec": row[3],
                "cv": compare_arrays(current, previous),
                "cv_by_scalar": {name: compare_arrays(current[index], previous[index])
                                 for index, name in enumerate(identity["scalar_names"])},
                "cls_cosine_vs_corrected": cosine(cls, features[index_of[fingerprint]]),
                "patch_cosine_vs_legacy_mean": float(patch_cosines.mean()),
                "render_mean_abs_difference": float(pixel_difference.mean()),
                "render_max_abs_difference": int(pixel_difference.max()),
                "current_patch_finite": bool(np.isfinite(patch).all()),
                "record_sha256": digest(record_path)})
    finally:
        connection.close()
    cv_positions = (np.arange(32) + 0.5) / 32
    patch_positions = (np.arange(32) + 0.5) * identity["patch_size"] / identity["input_size"]
    report = {"identity": identity, "sample_count": len(details),
        "render_gate": all(row["cls_cosine_vs_corrected"] >= 0.999 for row in details),
        "cv_changed_samples": sum(not row["cv"]["exact_equal"] for row in details),
        "dimensions_changed_samples": sum(row["current_dimensions"] != row["old_dimensions"] for row in details),
        "minimum_cls_cosine": min(row["cls_cosine_vs_corrected"] for row in details),
        "geometry": {"cv_grid_centers_normalized": cv_positions.tolist(),
            "patch_centers_normalized": patch_positions.tolist(),
            "maximum_center_offset_input_pixels": float(np.max(np.abs(cv_positions - patch_positions)) * identity["input_size"]),
            "patch_covered_input_pixels": 32 * identity["patch_size"],
            "input_pixels": identity["input_size"],
            "note": "CV 在全画面均匀采样；518 输入的 patch 覆盖前512像素，中心并非严格重合；不能声称像素级一一对齐。"},
        "details": details,
        "inputs": {str(path.resolve()): digest(path) for path in (args.sources, args.photos, args.features,
                    args.cache / "identity.json", args.cache / "complete.json", Path(__file__))}}
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "report.json", report)
    lines = ["# 修复技术输入核验", "", f"样本 {len(details)}，只用训练事件、按事件/格式确定性抽样，不据标签筛样本。",
             f"当前共享算法 CLS 对修复缓存最小 cosine = {report['minimum_cls_cosine']:.6f}；输入闸 = {report['render_gate']}。",
             f"旧 CV 与重算不同 {report['cv_changed_samples']}/{len(details)}；尺寸不同 {report['dimensions_changed_samples']}/{len(details)}。",
             "", report["geometry"]["note"], "",
             "| 事件 | 格式 | 尺寸变更 | CV平均绝对差 | CLS cosine | 渲染像素平均差 |",
             "|---|---|---|---:|---:|---:|"]
    for row in details:
        lines.append(f"| {row['event'] or '旧批'} | {row['format']} | "
            f"{row['old_dimensions']} → {row['current_dimensions']} | {row['cv']['mean_abs_difference']:.4f} | "
            f"{row['cls_cosine_vs_corrected']:.6f} | {row['render_mean_abs_difference']:.4f} |")
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("sample_count", "render_gate", "cv_changed_samples",
                                                   "dimensions_changed_samples", "minimum_cls_cosine")}, ensure_ascii=False))
    return 0 if report["render_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
