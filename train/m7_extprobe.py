"""
m7_extprobe.py — E4：外部先验探针（CLIP-L/14 + LAION improved-aesthetic-predictor，零训练）

回答的问题："瓶颈是 DINO 先验，还是监督结构/任务本身？"
LAION 美学头是为美学打分而生的外部先验（AVA + SAC + LOGOS 上训练的 CLIP-L/14 线性头），
**零训练**直接给全库打分，与我们的模型同口径对照（对级/Δ分层/top-1/rho/recall + cos 分层）：
  - 若它也落在 0.55-0.62 → 瓶颈不在 backbone 先验，而在监督结构/任务难度；
  - 若显著更高 → 先验重要，特征更换或蒸馏值得重议。

产物：train/out/m7_extprobe/{scores.csv, m7_report.md}

用法（仓根 D:/Git/PhotoViewer 下）：
    PYTHONUTF8=1 Tools/.venv-gpu/Scripts/python.exe Training/train/m7_extprobe.py
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m4_baseline import ABS_KEY_DEFAULT, ABS_TSV_DEFAULT, M3_DIR_DEFAULT, evaluate, load_pairs, load_photos  # noqa: E402
from m5_lora import CACHE_DEFAULT, cos_strata  # noqa: E402

CLIP_MS = "AI-ModelScope/clip-vit-large-patch14"
LAION_HEAD = "D:/PhotoDB/dataset/models/sac_logos_ava1-l14-linearMSE.pth"
OUT_DEFAULT = "Training/train/out/m7_extprobe"


def load_clip(device):
    """CLIP-L/14：优先本机 ModelScope 缓存，缺则 snapshot_download。"""
    from transformers import CLIPModel, CLIPImageProcessor
    try:
        from modelscope import snapshot_download
        path = snapshot_download(CLIP_MS)
    except Exception:
        # 已在本机缓存时直接定位
        base = os.path.expanduser("~/.cache/modelscope/hub/models")
        cand = [p for p in Path(base).glob("AI-ModelScope/clip-vit-large-patch14*")]
        path = str(cand[0]) if cand else CLIP_MS
    model = CLIPModel.from_pretrained(path).eval().to(device)
    proc = CLIPImageProcessor.from_pretrained(path)
    return model, proc


def load_laion_head(device):
    """improved-aesthetic-predictor：MLP(768→1024→128→64→16→1)，吃归一化 CLIP-L/14 图像嵌入。"""
    import torch
    from torch import nn
    sd = torch.load(LAION_HEAD, map_location=device)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    # 原实现：Linear 堆叠，ReLU 被注释、Dropout 在 eval 恒等（索引 0/2/4/6/7 对应）
    mlp = nn.Sequential(
        nn.Linear(768, 1024), nn.Dropout(0.2),
        nn.Linear(1024, 128), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.Dropout(0.1),
        nn.Linear(64, 16), nn.Linear(16, 1))
    sd = {k.replace("layers.", "", 1): v for k, v in sd.items()}   # 原版外层包 self.layers
    mlp.load_state_dict(sd)
    return mlp.eval().to(device)


def main() -> int:
    import torch
    from PIL import Image
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)
    model, proc = load_clip(device)
    head = load_laion_head(device)

    fps, _X, metas = load_photos(M3_DIR_DEFAULT, "D:/PhotoDB/dataset/photos_dataset.db",
                                 "dinov3_vits16_f32_518_v1+clhe2.0ycc1.0")
    meta_by_fp = dict(zip(fps, metas))
    scores = np.zeros(len(fps), dtype=np.float32)
    B = 64
    with torch.no_grad():
        for s in range(0, len(fps), B):
            imgs = [Image.open(os.path.join(CACHE_DEFAULT, f"{fp}.png")).convert("RGB")
                    for fp in fps[s:s + B]]
            x = proc(images=imgs, return_tensors="pt").pixel_values.to(device)
            out = model.get_image_features(pixel_values=x)
            feat = out if torch.is_tensor(out) else out.pooler_output
            feat = feat / feat.norm(dim=-1, keepdim=True)
            sc = head(feat).squeeze(-1)
            scores[s:s + B] = sc.float().cpu().numpy()
            if s % 1024 == 0:
                print(f"  打分 {s}/{len(fps)}", flush=True)

    pairs_tr = load_pairs(M3_DIR_DEFAULT, "train", meta_by_fp)
    pairs_va = load_pairs(M3_DIR_DEFAULT, "val", meta_by_fp)
    pairs_te = load_pairs(M3_DIR_DEFAULT, "test", meta_by_fp)
    abs_probe = defaultdict(list)
    abs_key = {r["new_name"]: r for r in csv.DictReader(open(ABS_KEY_DEFAULT, encoding="utf-8-sig"))}
    abs_rated = {n: int(r) for n, r in
                 (line.rstrip("\n").split("\t") for line in open(ABS_TSV_DEFAULT, encoding="utf-8"))}
    for name, k in abs_key.items():
        if k["fingerprint"] in meta_by_fp and meta_by_fp[k["fingerprint"]]["split"] == "test":
            abs_probe[k["event_label"]].append((k["fingerprint"], abs_rated[name]))

    class Stub:
        def __init__(self, s):
            import torch as _t
            self.s = _t.tensor(s).unsqueeze(-1)

        def eval(self):
            return self

        def __call__(self, x):
            return self.s

    R, emerg, s_of = evaluate(Stub(scores), np.zeros((len(fps), 1)), fps, metas,
                              {"train": pairs_tr, "val": pairs_va, "test": pairs_te}, abs_probe)
    strata = cos_strata(M3_DIR_DEFAULT, meta_by_fp, s_of)

    out = Path(OUT_DEFAULT)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "scores.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["fingerprint", "event", "seg_id", "split", "rating", "score"])
        for fp, m in zip(fps, metas):
            wr.writerow([fp, m["event"], m["seg"], m["split"], m["rating"], f"{s_of[fp]:.4f}"])

    from m5_lora import report_block
    L = ["# E4 外部先验探针：CLIP-L/14 + LAION 美学头（零训练）\n"]
    L.extend(report_block(R, emerg))
    L.append("- test window 对按 cos 分层: " + " · ".join(
        f"{k} {a:.3f}(n={n})" for k, (a, n) in strata.items()))
    (out / "m7_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    t = R["test"]
    print(f"\ntest: window {t['pairs']['window'][0]:.3f} global {t['pairs']['global'][0]:.3f} "
          f"derived {t['pairs']['derived'][0]:.3f} top1 {t['top1'][0]:.3f} "
          f"rho {t['seg_rho'][0]:.3f} recall {t['recall_event'][0]:.3f}")
    print("cos 分层: " + " · ".join(f"{k} {a:.3f}" for k, (a, n) in strata.items()))
    print(f"[OK] {out}/m7_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
