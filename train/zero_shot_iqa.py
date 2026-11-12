"""外部预训练美学/画质模型零样本打分：只读产品路径保持宽高比预览，不拟合本库任何标签。

预览来自 `DatasetBuilder --dump-technical --preview-long-edge`（PhotoDecode → sRGB8 → 长边缩放，方向与比例保持），
逐图核验预览哈希与来源记录；各模型用 pyiqa 官方权重与自身预处理逐张前向（宽高比不同，不拼批），
分数方向统一为“高 = 好”。默认只打 train 事件；val 只有在登记闸门通过后才允许 --splits val。
输出 scores.npz（指纹 × 模型）与身份 JSON，可续跑；不打开 test。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import pyiqa
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "audit"))
from evaluation_protocol import digest

MODELS = ("musiq-ava", "topiq_iaa", "nima", "laion_aes", "topiq_nr", "musiq", "clipiqa+", "liqe")


def state_digest(metric) -> str:
    """对模型全部参数/缓冲取 SHA256，固定权重身份。"""
    hasher = hashlib.sha256()
    for name, tensor in sorted(metric.net.state_dict().items()):
        hasher.update(name.encode())
        hasher.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def load_preview(cache: Path, fingerprint: str) -> torch.Tensor:
    """读取并核验单张预览，返回 [1,3,H,W] 0–1 张量。"""
    record = json.loads((cache / f"{fingerprint}.json").read_text(encoding="utf-8"))
    path = cache / f"{fingerprint}.preview.png"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["preview_sha256"]:
        raise ValueError(f"预览哈希不符 {fingerprint}")
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def main() -> int:
    """核验预览缓存完成闸后逐模型逐张打分，记录权重/实现/输入哈希，支持续跑。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photos", type=Path, default=ROOT / "audit/out/m3_pairs_v2_20260924b/photos.csv")
    parser.add_argument("--previews", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    args = parser.parse_args()
    if "test" in args.splits:
        raise ValueError("不打开 test")
    identity = json.loads((args.previews / "identity.json").read_text(encoding="utf-8"))
    complete = json.loads((args.previews / "complete.json").read_text(encoding="utf-8"))
    if identity["decode_version"] != "display-plane-v3" or not identity.get("preview_long_edge"):
        raise ValueError("预览缓存身份不符")
    if complete["completed"] != complete["requested"] or complete["identity_sha256"] != digest(args.previews / "identity.json"):
        raise ValueError("预览缓存未完成或身份漂移")
    with args.photos.open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    fingerprints = sorted(row["fingerprint"] for row in rows if row["split"] in args.splits)
    args.out.mkdir(parents=True, exist_ok=True)
    scores_path = args.out / "scores.npz"
    scores = dict()
    if scores_path.exists():
        with np.load(scores_path, allow_pickle=False) as cached:
            if cached["fingerprints"].tolist() != fingerprints:
                raise ValueError("续跑照片集合不一致")
            scores = dict((name, cached[name].copy()) for name in cached.files if name != "fingerprints")
    manifest_path = args.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
        "previews": str(args.previews), "preview_identity_sha256": digest(args.previews / "identity.json"),
        "photos_sha256": digest(args.photos), "splits": args.splits, "code_sha256": digest(Path(__file__)),
        "pyiqa_version": pyiqa.__version__, "torch_version": torch.__version__, "models": dict()}
    for name in args.models:
        if name in scores and np.isfinite(scores[name]).all():
            print(f"{name} 已完成，复用", flush=True)
            continue
        metric = pyiqa.create_metric(name, device="cuda", as_loss=False)
        metric.eval()
        values = np.full(len(fingerprints), np.nan)
        started = time.monotonic()
        with torch.inference_mode():
            for index, fingerprint in enumerate(fingerprints):
                output = metric(load_preview(args.previews, fingerprint).cuda())
                values[index] = float(output.reshape(-1)[0])
                if index % 2000 == 0:
                    print(f"{name} {index}/{len(fingerprints)} {time.monotonic() - started:.0f}s", flush=True)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} 输出非有限")
        scores[name] = -values if metric.lower_better else values
        manifest["models"][name] = {"lower_better_flipped": bool(metric.lower_better), "state_sha256": state_digest(metric),
            "implementation_sha256": digest(Path(inspect.getfile(type(metric.net)))),
            "seconds": round(time.monotonic() - started, 1)}
        np.savez(scores_path, fingerprints=np.array(fingerprints), **scores)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{name} done {manifest['models'][name]['seconds']}s", flush=True)
        del metric
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
