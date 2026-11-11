"""从修复 518 渲染提取冻结 DINOv3 patch 网格缓存，供同场景比较实验使用，不读评分标签。

只消费 display-plane-v3 修复 PNG，按照片清单的 split 选取（默认 train+val，test 不提取）。
patch 为 last_hidden_state 末 1024 token（跳过 CLS 与 register），与产品 ONNX 导出一致；
fp32 前向（bf16 autocast 实测个别 token 与产品 ONNX cos 低至 0.98，不过闸）；启动前用技术侧缓存中
产品 ONNX 的 patch/CLS 做逐 token 一致性闸（严格真源）。来源身份由逐图渲染 SHA256 对修复特征清单核验；
CLS 对修复特征缓存只防粗错（该缓存为 bf16 提取，个别图与 fp32 低至 cos 0.985，阈值 0.95）。
输出为 float16 memmap（指纹为 corrected_features 顺序的子集）+ 身份 JSON；支持断点续跑，
全部覆盖并复核哈希后才写 complete.json。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "audit"))
from evaluation_protocol import digest

TOKENS, DIMENSION, SIZE = 1024, 384, 518
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def encode(model, images: list[np.ndarray], device: str, dtype) -> tuple[np.ndarray, np.ndarray]:
    """批量前向，返回 L2 归一化 CLS 与末 1024 个 patch token（float32）。"""
    pixels = torch.from_numpy(np.stack(images)).to(device)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    pixels = (pixels.permute(0, 3, 1, 2).float() / 255 - mean) / std
    with torch.inference_mode(), torch.autocast(device, dtype=dtype, enabled=dtype != torch.float32):
        hidden = model(pixel_values=pixels).last_hidden_state.float()
    cls = torch.nn.functional.normalize(hidden[:, 0], dim=-1)
    return cls.cpu().numpy(), hidden[:, -TOKENS:].cpu().numpy()


def read_render(path: Path) -> np.ndarray:
    """读取修复渲染 PNG 并校验 518 方形尺寸，返回 RGB uint8 数组。"""
    with Image.open(path) as image:
        if image.size != (SIZE, SIZE):
            raise ValueError(f"渲染尺寸错误 {path.name}: {image.size}")
        return np.array(image.convert("RGB"))


def parity_gate(model, technical: Path, device: str, dtype) -> dict:
    """对技术侧缓存中产品 ONNX 产出的 patch/CLS 逐 token 比对，返回统计；不过闸即抛错。"""
    entries = sorted(path.name[:-len(".patch.bin")] for path in technical.glob("*.patch.bin"))
    if not entries:
        raise ValueError("技术侧缓存没有 patch 样本")
    patch_cos, cls_cos = [], []
    for fingerprint in entries:
        record = json.loads((technical / f"{fingerprint}.json").read_text(encoding="utf-8"))
        render = technical / f"{fingerprint}.png"
        if hashlib.sha256(render.read_bytes()).hexdigest() != record["render_sha256"]:
            raise ValueError(f"技术样本渲染哈希不符 {fingerprint}")
        cls, patches = encode(model, [read_render(render)], device, dtype)
        reference = np.fromfile(technical / f"{fingerprint}.patch.bin", dtype="<f4").reshape(TOKENS, DIMENSION)
        reference_cls = np.fromfile(technical / f"{fingerprint}.cls.bin", dtype="<f4")
        numerator = (patches[0] * reference).sum(axis=1)
        patch_cos.append(numerator / np.linalg.norm(patches[0], axis=1) / np.linalg.norm(reference, axis=1))
        cls_cos.append(float(cls[0] @ reference_cls / np.linalg.norm(reference_cls)))
    patch_cos = np.concatenate(patch_cos)
    result = {"samples": len(entries), "patch_cos_mean": float(patch_cos.mean()),
        "patch_cos_min": float(patch_cos.min()), "cls_cos_min": float(min(cls_cos))}
    if result["patch_cos_min"] < 0.99 or result["patch_cos_mean"] < 0.999 or result["cls_cos_min"] < 0.999:
        raise ValueError(f"patch 一致性闸失败: {result}")
    return result


def comparable_identity(plan: dict) -> dict:
    """返回续跑时必须一致的身份字段（不含一致性闸的浮点统计）。"""
    return {key: value for key, value in plan.items() if key != "parity_gate"}


def main() -> int:
    """核验输入身份与产品一致性后按批提取 patch 网格，记录全部哈希，支持续跑。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photos", type=Path, default=ROOT / "audit/out/m3_pairs_v2_20260924b/photos.csv")
    parser.add_argument("--features", type=Path, default=ROOT / "train/out/recovery_work/corrected_features.npz")
    parser.add_argument("--render-dir", type=Path, default=Path("D:/PhotoDB/dataset/render518-display-plane-v3"))
    parser.add_argument("--technical", type=Path,
        default=Path("D:/PhotoDB/dataset/technical-display-plane-v3-sample-20261026-final"))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if "test" in args.splits:
        raise ValueError("本实验不提取 test")
    dtype = torch.float32
    if (args.render_dir / "decode-version.txt").read_text().strip() != "display-plane-v3":
        raise ValueError("渲染解码版本错误")
    with args.photos.open(encoding="utf-8-sig") as stream:
        split_of = {row["fingerprint"]: row["split"] for row in csv.DictReader(stream)}
    identity = json.loads(args.features.with_suffix(".json").read_text(encoding="utf-8"))
    if identity["decode_version"] != "display-plane-v3" or identity["metadata_sha256"] != digest(args.photos):
        raise ValueError("修复特征缓存身份不符")
    with np.load(args.features, allow_pickle=False) as cached:
        all_fingerprints = cached["fingerprints"].tolist()
        corrected_dino = cached["dino"].astype(np.float64)
    position = {fingerprint: index for index, fingerprint in enumerate(all_fingerprints)}
    fingerprints = [fp for fp in all_fingerprints if split_of[fp] in args.splits]
    model = AutoModel.from_pretrained(args.model_dir, local_files_only=True).to(args.device).eval()
    gate = parity_gate(model, args.technical, args.device, dtype)
    print(f"parity gate {gate}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    plan = {"decode_version": "display-plane-v3", "splits": args.splits, "photos": len(fingerprints),
        "tokens": TOKENS, "dimension": DIMENSION, "dtype": "float16",
        "layout": "[photo, token, dim]；token 为 32x32 行优先网格", "model_dir": str(args.model_dir),
        "weights_sha256": digest(args.model_dir / "model.safetensors"), "autocast": str(dtype),
        "photos_sha256": digest(args.photos), "features_sha256": digest(args.features),
        "code_sha256": digest(Path(__file__)), "parity_gate": gate}
    plan_path = args.output / "identity.json"
    if plan_path.exists():
        previous = json.loads(plan_path.read_text(encoding="utf-8"))
        if comparable_identity(previous) != comparable_identity(plan):
            raise ValueError("续跑身份与已有缓存不一致，换输出目录")
    else:
        plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output / "fingerprints.json").write_text(json.dumps(fingerprints), encoding="utf-8")
    shape = (len(fingerprints), TOKENS, DIMENSION)
    patch_path = args.output / "patches.f16.npy"
    if patch_path.exists():
        patches = np.load(patch_path, mmap_mode="r+")
    else:
        patches = np.lib.format.open_memmap(patch_path, mode="w+", dtype=np.float16, shape=shape)
    if patches.shape != shape or patches.dtype != np.float16:
        raise ValueError("续跑 patch 缓存形状不符")
    progress_path = args.output / "progress.json"
    hashes = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    remaining = [index for index, fp in enumerate(fingerprints) if fp not in hashes]
    cls_cos_min, started = 1.0, time.monotonic()
    for batch_number, start in enumerate(range(0, len(remaining), args.batch)):
        batch = remaining[start:start + args.batch]
        images, digests = [], {}
        for index in batch:
            fingerprint = fingerprints[index]
            path = args.render_dir / f"{fingerprint}.png"
            digests[fingerprint] = hashlib.sha256(path.read_bytes()).hexdigest()
            if digests[fingerprint] != identity["images"][fingerprint]:
                raise ValueError(f"修复渲染与特征缓存来源不一致 {fingerprint}")
            images.append(read_render(path))
        cls, tokens = encode(model, images, args.device, dtype)
        reference = corrected_dino[[position[fingerprints[index]] for index in batch]]
        cls_cos_min = min(cls_cos_min, float((cls * reference).sum(axis=1).min()))
        if cls_cos_min < 0.95:
            raise ValueError(f"CLS 与修复特征缓存不一致: {cls_cos_min}")
        if not np.isfinite(tokens).all() or np.abs(tokens).max() > 6e4:
            raise ValueError("patch 非有限或超出 float16 范围")
        patches[batch] = tokens.astype(np.float16)
        patches.flush()
        hashes.update(digests)
        temporary = progress_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(hashes), encoding="utf-8")
        temporary.replace(progress_path)
        if batch_number % 40 == 0:
            print(f"patches {len(hashes)}/{len(fingerprints)} {time.monotonic() - started:.0f}s "
                f"cls_min={cls_cos_min:.6f}", flush=True)
    if set(hashes) != set(fingerprints):
        raise ValueError("patch 缓存未覆盖全部照片")
    del patches
    complete = {"photos": len(fingerprints), "patch_sha256": digest(patch_path),
        "cls_cos_min_vs_corrected": cls_cos_min, "parity_gate": gate}
    (args.output / "complete.json").write_text(json.dumps(complete, indent=2), encoding="utf-8")
    print(f"patch cache complete {complete}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
