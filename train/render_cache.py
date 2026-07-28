"""
render_cache.py — 梯4 前置：fingerprint → 518×518 渲染缓存的闸门与路径解析

**2026-07-27 重要更正**：本脚本原有的 Python 渲染路径（PIL/pillow-heif 解码 +
BILINEAR 缩放）经 CLS 一致性闸门证伪——与 C# 入库管线产出的 cls_vector 余弦均值
仅 0.90（重采样/解码实现差异，BILINEAR/BICUBIC/LANCZOS/BOX 均无法弥合）。
**渲染缓存的正路是 DatasetBuilder `--dump-render`**（与 DINO 预处理同一
LoadBitmap + RenderInputBitmap 路径，PNG 无损落盘 <fingerprint>.png）：
    dotnet run --project Training/DatasetBuilder -- --manifest D:/PhotoDB/dataset/manifest.2026-07-19.json --dump-render D:/PhotoDB/dataset/render518
    dotnet run --project Training/DatasetBuilder -- D:/PhotoDB/20240212 --dump-render D:/PhotoDB/dataset/render518

本脚本保留两个职能：
1. 正确性闸门（--gate N）：随机抽 N 张缓存 PNG，用冻结 ViT-S(torch, ModelScope 缓存)
   前向算 CLS，与 DB 内 cls_vector（C# 管线产物）比 cosine——均值 ≥0.99 证明
   训练像素与入库特征同分布。
2. 路径解析验证（--resolve-only）：fingerprint → 代表件绝对路径的命中审计。

用法（仓根 D:/Git/PhotoViewer 下）：
    PYTHONUTF8=1 Tools/.venv-gpu/Scripts/python.exe Training/train/render_cache.py --gate 50
    PYTHONUTF8=1 Tools/.venv-gpu/Scripts/python.exe Training/train/render_cache.py --resolve-only
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys

DB_DEFAULT = "D:/PhotoDB/dataset/photos_dataset.db"
MANIFEST_DEFAULT = "D:/PhotoDB/dataset/manifest.2026-07-19.json"
LEGACY_ROOT = "D:/PhotoDB/20240212"          # event_label IS NULL 的旧批（单文件夹入库）
OUT_DEFAULT = "D:/PhotoDB/dataset/render518"
MODEL_VITS = "dinov3_vits16_f32_518_v1"      # 闸门对照用的 DB 特征（原片路）

# ---------------------------------------------------------------------------


def load_roots(manifest_path: str) -> dict[str | None, list[str]]:
    """event_label → 入库根目录列表（同事件多文件夹/多相机子目录）。"""
    import json
    m = json.load(open(manifest_path, encoding="utf-8"))
    roots: dict[str | None, list[str]] = {}
    for f in m["folders"]:
        roots.setdefault(f.get("eventLabel"), []).append(f["path"])
    return roots


def resolve_photos(db: str, roots: dict[str | None, list[str]]):
    """fingerprint → 代表件绝对路径（存在性校验）。返回 (ok: dict, missing: list)。"""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT fingerprint, source_rel_path, event_label FROM photos").fetchall()
    finally:
        conn.close()
    ok, missing = {}, []
    for fp, rel, ev in rows:
        cands = [os.path.join(r, rel) for r in roots.get(ev, [])]
        if ev is None:
            cands.append(os.path.join(LEGACY_ROOT, rel))
        hit = next((c for c in cands if os.path.isfile(c)), None)
        if hit:
            ok[fp] = hit
        else:
            missing.append((fp, rel, ev))
    return ok, missing


def _worker_init():  # 已废弃：Python 渲染路径被 CLS 闸门证伪（见模块 docstring），保留函数体防引用残留
    raise RuntimeError("Python 渲染路径已废弃：请用 DatasetBuilder --dump-render 产出渲染缓存")


# ---------------------------------------------------------------------------

def gate(db, out_dir, n, seed=0):
    """抽 n 张缓存 PNG（DatasetBuilder --dump-render 产物），torch ViT-S 前向 CLS vs DB cls_vector。"""
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModel

    mid = os.path.expanduser(
        "~/.cache/modelscope/hub/models/facebook/dinov3-vits16-pretrain-lvd1689m")
    model = AutoModel.from_pretrained(mid).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        ref = {fp: np.frombuffer(b, dtype="<f4") for fp, b in conn.execute(
            "SELECT fingerprint, cls_vector FROM photo_features WHERE model_id=?", (MODEL_VITS,))}
    finally:
        conn.close()
    fps = [fp for fp in ref if os.path.isfile(os.path.join(out_dir, f"{fp}.png"))]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(fps), size=min(n, len(fps)), replace=False)

    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    cos = []
    with torch.no_grad():
        for ix in pick:
            fp = fps[int(ix)]
            img = Image.open(os.path.join(out_dir, f"{fp}.png")).convert("RGB")
            x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
            x = x.permute(2, 0, 1).unsqueeze(0).to(dev)
            x = (x - mean) / std
            cls = model(pixel_values=x).last_hidden_state[:, 0, :].float().cpu().numpy()[0]
            a = cls / (np.linalg.norm(cls) + 1e-12)
            b = ref[fp] / (np.linalg.norm(ref[fp]) + 1e-12)
            cos.append(float(a @ b))
    cos = np.array(cos)
    print(f"GATE: n={len(cos)} cosine mean={cos.mean():.4f} min={cos.min():.4f} "
          f"p10={np.percentile(cos, 10):.4f}")
    ok = cos.mean() >= 0.99
    print("GATE " + ("PASS" if ok else "FAIL（渲染路径与入库特征不同分布，须排查重采样/解码差异）"))
    return 0 if ok else 1


def resolve_only(db, manifest):
    roots = load_roots(manifest)
    ok_map, missing = resolve_photos(db, roots)
    print(f"路径解析: {len(ok_map)} 命中 / {len(missing)} 缺失")
    for fp, rel, ev in missing[:20]:
        print(f"  [缺失] {fp[:8]} {ev} {rel}")
    return 0 if not missing else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="梯4 前置：渲染缓存闸门 + 路径解析（渲染本身走 DatasetBuilder --dump-render）")
    ap.add_argument("--db", default=DB_DEFAULT)
    ap.add_argument("--manifest", default=MANIFEST_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--gate", type=int, default=0, help="抽 N 张缓存 PNG 做 CLS 一致性闸门")
    ap.add_argument("--resolve-only", action="store_true", help="只验证 fingerprint → 源文件路径解析")
    args = ap.parse_args()
    if args.gate > 0:
        return gate(args.db, args.out, args.gate)
    return resolve_only(args.db, args.manifest)


if __name__ == "__main__":
    sys.exit(main())
