"""
m5_lora.py — 梯4（决策 8 最终逃生梯）：ViT-S/16@518 LoRA 微调 + 同口径评估

背景：梯1-3 全证死（冻结特征跨场零迁移/弱脉搏/分辨率不放大）。梯4 直接微调
backbone 本身（LoRA 注入每层 attention q/k/v/o_proj），用 M3 训练对做加权软偏好
学习，评估与 m4_baseline 完全同口径（import 复用 evaluate），与梯2/梯3 数字可比。

模型：DINOv3ViTModel（ModelScope 缓存，HF 门控 401 绕行）全冻结；
    LoRA(r=16, α=32, dropout 0.05) × 12 层 × q/k/v/o_proj；头 = Linear(384→1) 吃 CLS。
损失：与 M4 同式的加权 softplus 对级 logistic（window/global/derived 三类全用）。
训练：bf16 autocast；batch 16 对（32 图）；AdamW（LoRA 1e-4 / 头 1e-3）；
    固定 epoch 不早停（交接教训：val 近 chance 时早停≈随机）；每 epoch 全库打分 +
    全指标评估，报告轨迹。无增广（翻转改构图语义、影调增广违宪）；EXIF/CV 不进模型。
评估：桩 net 喂 m4_baseline.evaluate → 同一份报告（对级按类/按Δ、段内 top-1、
    段内 Spearman、recall@12.5% ×2、0-5 容差三层、abs 涌现代理）。

用法（仓根 D:/Git/PhotoViewer 下）：
    PYTHONUTF8=1 Tools/.venv-gpu/Scripts/python.exe Training/train/m5_lora.py --smoke
    PYTHONUTF8=1 Tools/.venv-gpu/Scripts/python.exe Training/train/m5_lora.py --epochs 3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m4_baseline import ABS_KEY_DEFAULT, ABS_TSV_DEFAULT, M3_DIR_DEFAULT, evaluate, load_pairs, load_photos  # noqa: E402

CACHE_DEFAULT = "D:/PhotoDB/dataset/render518"
MID_DEFAULT = os.path.expanduser(
    "~/.cache/modelscope/hub/models/facebook/dinov3-vits16-pretrain-lvd1689m")

LORA_R = 16
LORA_ALPHA = 32
LORA_DROP = 0.05
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
BATCH_PAIRS = 16
EPOCHS = 3
LR_LORA = 1e-4
LR_HEAD = 1e-3
SEED = 0

# ---------------------------------------------------------------------------


class ClsHead:
    """头 v1：Linear 吃 CLS。"""

    @staticmethod
    def build(dim, _n_reg):
        from torch import nn

        class _H(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(dim, 1)

            def forward(self, h):
                return self.lin(h[:, 0, :]).squeeze(-1)
        return _H()


class PatchHead:
    """头 v2（E1，用户 patch 假设）：CLS + patch token 注意力池化拼接 → Linear。
    patch token = "什么在哪里"的构成抽象；可学 query 池化保留空间语义的软选择。"""

    @staticmethod
    def build(dim, n_reg):
        import torch
        from torch import nn

        class _H(nn.Module):
            def __init__(self):
                super().__init__()
                self.q = nn.Parameter(torch.randn(dim) * 0.02)
                self.lin = nn.Linear(dim * 2, 1)
                self.n_reg = n_reg

            def forward(self, h):
                cls = h[:, 0, :]
                patch = h[:, 1 + self.n_reg:, :]
                w = torch.softmax(torch.einsum("btd,d->bt", patch, self.q) / dim ** 0.5, dim=1)
                pooled = torch.einsum("bt,btd->bd", w, patch)
                return self.lin(torch.cat([cls, pooled], -1)).squeeze(-1)
        return _H()


HEADS = {"cls": ClsHead, "patch": PatchHead}


def build_model(mid: str, device: str, head_kind: str = "cls"):
    """冻结 backbone + 注入 LoRA + 评分头。返回 (model, head, lora_params)。"""
    import torch
    from torch import nn
    from transformers import AutoModel

    class LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear):
            super().__init__()
            self.base = base
            self.drop = nn.Dropout(LORA_DROP)
            self.A = nn.Linear(base.in_features, LORA_R, bias=False)
            self.B = nn.Linear(LORA_R, base.out_features, bias=False)
            nn.init.zeros_(self.B.weight)
            self.scaling = LORA_ALPHA / LORA_R

        def forward(self, x):
            return self.base(x) + self.B(self.A(self.drop(x))) * self.scaling

    model = AutoModel.from_pretrained(mid)
    for p in model.parameters():
        p.requires_grad_(False)

    def set_submodule(root, path, new):
        mod = root
        parts = path.split(".")
        for p in parts[:-1]:
            mod = mod[int(p)] if p.isdigit() else getattr(mod, p)
        last = parts[-1]
        if last.isdigit():
            mod[int(last)] = new
        else:
            setattr(mod, last, new)

    n_inj = 0
    for name, mod in list(model.named_modules()):
        if name.endswith(LORA_TARGETS) and isinstance(mod, nn.Linear):
            set_submodule(model, name, LoRALinear(mod))
            n_inj += 1
    assert n_inj == 48, f"LoRA 注入点 {n_inj} != 48（12 层 × 4 proj）"
    n_reg = int(getattr(model.config, "num_register_tokens", 4))
    head = HEADS[head_kind].build(model.config.hidden_size, n_reg)
    model.to(device)
    head.to(device)
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad]
    print(f"LoRA 注入 {n_inj} 处；可训参数 LoRA {sum(p.numel() for p in lora_params)} + "
          f"头({head_kind}) {sum(p.numel() for p in head.parameters())}", flush=True)
    return model, head, lora_params


def forward_scores(model, head, x):
    """x: [B,3,518,518] 已归一化 → [B] 分数。"""
    h = model(pixel_values=x).last_hidden_state
    return head(h.float())


# ---------------------------------------------------------------------------

class PairDataset:
    """pairs → (fp_i, fp_j, y, w)；图像在 collate 时从渲染缓存读。"""

    def __init__(self, pairs, cache):
        self.pairs = pairs
        self.cache = cache

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, ix):
        fi, fj, y, w, _pt, _d = self.pairs[ix]
        return fi, fj, y, w


def _load_img(cache, fp):
    from PIL import Image
    with Image.open(os.path.join(cache, f"{fp}.png")) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


class PairCollate:
    """顶层可调用类（Windows spawn worker 须可 pickle）。返回 [2B,H,W,3] uint8 + y + w。"""

    def __init__(self, cache):
        self.cache = cache

    def __call__(self, items):
        import torch
        fi, fj, y, w = zip(*items)
        imgs = [_load_img(self.cache, f) for f in (*fi, *fj)]
        x = torch.from_numpy(np.stack(imgs))            # 前 B = i，后 B = j
        return x, torch.tensor(y, dtype=torch.float32), torch.tensor(w, dtype=torch.float32)


class FpDataset:
    def __init__(self, fps):
        self.fps = fps

    def __len__(self):
        return len(self.fps)

    def __getitem__(self, ix):
        return self.fps[ix]


class FpCollate:
    def __init__(self, cache):
        self.cache = cache

    def __call__(self, items):
        import torch
        return torch.from_numpy(np.stack([_load_img(self.cache, f) for f in items]))


def score_all(model, head, fps, cache, device, batch=64):
    """全库打分（inference，bf16）→ np.array，与 fps 同序。"""
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(FpDataset(fps), batch_size=batch, shuffle=False, num_workers=6,
                        collate_fn=FpCollate(cache), pin_memory=True, persistent_workers=True)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = np.zeros(len(fps), dtype=np.float32)
    ix = 0
    model.eval()
    head.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        for x in loader:
            x = x.to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
            x = (x - mean) / std
            s = forward_scores(model, head, x)
            out[ix:ix + len(s)] = s.float().cpu().numpy()
            ix += len(s)
    return out


class _StubNet:
    """把预算分数包装成 m4_baseline.evaluate 期待的 net 接口。"""

    def __init__(self, scores):
        import torch
        self._s = torch.from_numpy(scores).unsqueeze(-1)

    def eval(self):
        return self

    def __call__(self, _x):
        return self._s


def cos_strata(m3_dir, meta_by_fp, s_of):
    """test window 对按 cos 分层的一致率——高 cos 带内容受控，是纯构图判别的主指标。"""
    rows = [r for r in csv.DictReader(open(f"{m3_dir}/pairs_test.csv", encoding="utf-8-sig"))
            if r["ptype"] == "window"]
    out = {}
    for lo, hi in ((0.0, 0.88), (0.88, 0.95), (0.95, 0.98), (0.98, 1.01)):
        ok = []
        for r in rows:
            c = float(r["cos"])
            if not (lo <= c < hi):
                continue
            mi, mj = meta_by_fp.get(r["fp_i"]), meta_by_fp.get(r["fp_j"])
            if mi is None or mj is None or r["fp_i"] not in s_of or r["fp_j"] not in s_of:
                continue
            y = np.sign(mi["rating"] - mj["rating"])
            if y == 0:
                continue
            ok.append(int(np.sign(s_of[r["fp_i"]] - s_of[r["fp_j"]]) == y))
        if ok:
            out[f"[{lo},{hi})"] = (float(np.mean(ok)), len(ok))
    return out


# ---------------------------------------------------------------------------

def report_block(R, emerg):
    """与 m4_baseline 报告同格式（逐 split）。"""
    L = []
    for split, sub in R.items():
        L.append(f"## {split}\n")
        L.append("- 对级一致率（按类型）: " + " · ".join(
            f"{k} {a:.3f}(n={n})" for k, (a, n) in sub["pairs"].items()))
        L.append("- 对级一致率（按 |Δ星级|）: " + " · ".join(
            f"Δ{'≥3' if k == 3 else k} {a:.3f}(n={n})" for k, (a, n) in sub["delta"].items()))
        t1, ch, nt = sub["top1"]
        L.append(f"- 段内 top-1 命中: **{t1:.3f}**（chance {ch:.3f}，n={nt} 段）")
        mr, sr, nr = sub["seg_rho"]
        L.append(f"- 段内 Spearman: **{mr:.3f}±{sr:.3f}**（n={nr} 段）")
        for mode in ("event", "seg_quota"):
            rc, g, nd = sub[f"recall_{mode}"]
            L.append(f"- ≥3★ 召回@Top12.5%（{mode}）: **{rc:.3f}**（{g}/{nd}）")
        st = sub["star"]
        L.append(f"- 0-5 容差（exact/±1/±2 累计）: 团顶清洁 {st['clean'][0]:.3f}/{st['clean'][1]:.3f}/"
                 f"{st['clean'][2]:.3f}(n={st['clean'][3]}) · 非团顶 {st['dirty'][0]:.3f}/"
                 f"{st['dirty'][1]:.3f}/{st['dirty'][2]:.3f}(n={st['dirty'][3]})")
        L.append("")
    L.append("## 绝对涌现代理（test 事件 abs_set 盲评照片，金标准未到）\n")
    for ev, (rho, n) in sorted(emerg.items()):
        L.append(f"- {ev}: Spearman {rho:.3f}（n={n}）")
    return L


def main() -> int:
    ap = argparse.ArgumentParser(description="梯4：ViT-S LoRA 微调 + 同口径评估")
    ap.add_argument("--m3", default=M3_DIR_DEFAULT)
    ap.add_argument("--cache", default=CACHE_DEFAULT)
    ap.add_argument("--mid", default=MID_DEFAULT, help="ViT-S 权重路径（ModelScope 缓存）")
    ap.add_argument("--abs-key", default=ABS_KEY_DEFAULT)
    ap.add_argument("--abs-tsv", default=ABS_TSV_DEFAULT)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "out" / "m5_lora"))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-pairs", type=int, default=BATCH_PAIRS)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--head", default="cls", choices=list(HEADS), help="评分头：cls / patch（E1 空间头）")
    ap.add_argument("--w-window", type=float, default=1.0, help="window 对权重乘子")
    ap.add_argument("--w-global", type=float, default=1.0, help="global 对权重乘子")
    ap.add_argument("--w-derived", type=float, default=1.0, help="derived 对权重乘子")
    ap.add_argument("--min-cos", type=float, default=0.0, help="E2：window 对只留 cos ≥ 阈值")
    ap.add_argument("--tie-loss", type=float, default=0.0, help="E3：团等价辅助损失权重 λ（0=关）")
    ap.add_argument("--clean-top", action="store_true", help="E5：window 对只留两端皆团顶/孤立照（干净监督）")
    ap.add_argument("--any-top", action="store_true", help="E6：window 对只留至少一端团顶（剔双落败者/双非顶）")
    ap.add_argument("--smoke", action="store_true", help="512 对 × 30 步冒烟，不评估")
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__}", flush=True)

    # 元数据与对（与 m4 同口径；X 特征不用，只为 fps/metas 顺序与过滤语义）
    fps, _X, metas = load_photos(args.m3, "D:/PhotoDB/dataset/photos_dataset.db",
                                 "dinov3_vits16_f32_518_v1+clhe2.0ycc1.0")
    meta_by_fp = dict(zip(fps, metas))
    missing = [fp for fp in fps if not os.path.isfile(os.path.join(args.cache, f"{fp}.png"))]
    if missing:
        print(f"[ERROR] 渲染缓存缺 {len(missing)} 张（先跑 DatasetBuilder --dump-render）", flush=True)
        return 1
    pairs_tr = load_pairs(args.m3, "train", meta_by_fp)
    pairs_va = load_pairs(args.m3, "val", meta_by_fp)
    pairs_te = load_pairs(args.m3, "test", meta_by_fp)
    # 监督重构（可选）：按 ptype 加权乘子——derived 主监督实验等
    wmul = {"window": args.w_window, "global": args.w_global, "derived": args.w_derived}
    if any(v != 1.0 for v in wmul.values()):
        pairs_tr = [(fi, fj, y, w * wmul[pt], pt, d) for fi, fj, y, w, pt, d in pairs_tr]
        print(f"监督加权: {wmul}", flush=True)
    # E2 相似带专注：window 对只留 cos ≥ --min-cos（跨内容噪声对剔除）
    if args.min_cos > 0:
        cos_of = {(r["fp_i"], r["fp_j"]): float(r["cos"])
                  for r in csv.DictReader(open(f"{args.m3}/pairs_train.csv", encoding="utf-8-sig"))}
        before = len(pairs_tr)
        pairs_tr = [p for p in pairs_tr if p[4] != "window"
                    or cos_of.get((p[0], p[1]), 0.0) >= args.min_cos]
        print(f"E2 min-cos={args.min_cos}: window 对 {before - len(pairs_tr)} 剔除，剩 {len(pairs_tr)}", flush=True)
    # E5 干净监督：window 对只留两端皆团顶/孤立照（排除"重复压低"污染数据，宪法 v1.8 作训练选择）
    if args.clean_top:
        before = len(pairs_tr)
        pairs_tr = [p for p in pairs_tr if p[4] != "window"
                    or (meta_by_fp[p[0]]["is_top"] and meta_by_fp[p[1]]["is_top"])]
        from collections import Counter
        print(f"E5 干净监督: window 对剔除非团顶 {before - len(pairs_tr)}，"
              f"剩 {len(pairs_tr)}（{Counter(p[4] for p in pairs_tr)}）", flush=True)
    # E6 忠实版：window 对只留"至少一端团顶"（保留胜者vs落败者团内监督，剔双落败者/双非顶）
    if args.any_top:
        before = len(pairs_tr)
        pairs_tr = [p for p in pairs_tr if p[4] != "window"
                    or (meta_by_fp[p[0]]["is_top"] or meta_by_fp[p[1]]["is_top"])]
        from collections import Counter
        print(f"E6 至少一端团顶: window 对剔除 {before - len(pairs_tr)}，"
              f"剩 {len(pairs_tr)}（{Counter(p[4] for p in pairs_tr)}）", flush=True)
    # E3 团等价辅助：同团同星对（训练侧 tie）→ |si-sj| 项，教"内容保持的变化不该改分"
    tie_pairs = []
    if args.tie_loss > 0:
        by_cluster = defaultdict(list)
        for fp, m in meta_by_fp.items():
            if m["split"] == "train" and m["cluster"] >= 0:
                by_cluster[(m["event"], m["seg"], m["cluster"])].append(fp)
        rng = np.random.default_rng(SEED)
        for _, members in by_cluster.items():
            if len(members) < 2:
                continue
            same_star = [(a, b) for i, a in enumerate(members) for b in members[i + 1:]
                         if meta_by_fp[a]["rating"] == meta_by_fp[b]["rating"]]
            for a, b in same_star[:6]:          # 每团至多 6 对，防大团淹没
                tie_pairs.append((a, b, 0, args.tie_loss, "tie", 0))
        rng.shuffle(tie_pairs)
        tie_pairs = tie_pairs[:20000]
        pairs_tr = pairs_tr + tie_pairs
        print(f"E3 团等价辅助: {len(tie_pairs)} 对 tie（λ={args.tie_loss}）", flush=True)
    print(f"对: train {len(pairs_tr)} · val {len(pairs_va)} · test {len(pairs_te)}", flush=True)

    model, head, lora_params = build_model(args.mid, device, args.head)
    opt = torch.optim.AdamW([
        {"params": lora_params, "lr": LR_LORA},
        {"params": head.parameters(), "lr": LR_HEAD},
    ], weight_decay=0.0)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # abs 涌现代理集（同 m4 main）
    abs_probe = defaultdict(list)
    abs_key = {r["new_name"]: r for r in csv.DictReader(open(args.abs_key, encoding="utf-8-sig"))}
    abs_rated = {n: int(r) for n, r in
                 (line.rstrip("\n").split("\t") for line in open(args.abs_tsv, encoding="utf-8"))}
    for name, k in abs_key.items():
        if k["fingerprint"] in meta_by_fp and meta_by_fp[k["fingerprint"]]["split"] == "test":
            abs_probe[k["event_label"]].append((k["fingerprint"], abs_rated[name]))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        sub = pairs_tr[:512]
        loader = DataLoader(PairDataset(sub, args.cache), batch_size=args.batch_pairs,
                            shuffle=True, num_workers=args.workers,
                            collate_fn=PairCollate(args.cache), pin_memory=True)
        model.train()
        head.train()
        for step, (x, y, w) in enumerate(loader):
            x = x.to(device).permute(0, 3, 1, 2).float().div_(255.0)
            x = (x - mean) / std
            y, w = y.to(device), w.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                s = forward_scores(model, head, x)
            si, sj = s[: len(y)], s[len(y):]              # 前 B = i，后 B = j
            diff = si - sj
            per = torch.where(y == 0, diff.abs(), torch.nn.functional.softplus(-diff * y))
            loss = (per * w).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 10 == 0:
                print(f"step {step} loss {loss.item():.4f} "
                      f"VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)
            if step >= 29:
                break
        print(f"SMOKE OK peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)
        return 0

    g = torch.Generator().manual_seed(SEED)
    loader = DataLoader(PairDataset(pairs_tr, args.cache), batch_size=args.batch_pairs,
                        shuffle=True, num_workers=args.workers, generator=g,
                        collate_fn=PairCollate(args.cache), pin_memory=True,
                        persistent_workers=True)
    pairs_tr_eval = [p for p in pairs_tr if p[4] != "tie"]   # tie 对只进损失，不进指标

    history = []
    report = ["# 梯4 LoRA 微调报告（决策 8 最终逃生梯）\n",
              f"LoRA r={LORA_R} α={LORA_ALPHA} drop={LORA_DROP} 目标={LORA_TARGETS}；"
              f"batch {args.batch_pairs} 对；lr {LR_LORA}/{LR_HEAD}；固定 {args.epochs} epoch 不早停\n"]
    for ep in range(1, args.epochs + 1):
        model.train()
        head.train()
        tot = cnt = 0.0
        for x, y, w in loader:
            x = x.to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
            x = (x - mean) / std
            y, w = y.to(device, non_blocking=True), w.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                s = forward_scores(model, head, x)
            si, sj = s[: len(y)], s[len(y):]              # 前 B = i，后 B = j
            diff = si - sj
            per = torch.where(y == 0, diff.abs(), torch.nn.functional.softplus(-diff * y))
            loss = (per * w).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            cnt += 1
            if cnt % 500 == 0:
                print(f"ep{ep} step {cnt}/{len(loader)} loss {tot / cnt:.4f}", flush=True)

        scores = score_all(model, head, fps, args.cache, device)
        X_dummy = np.zeros((len(fps), 1), dtype=np.float32)
        R, emerg, s_of = evaluate(_StubNet(scores), X_dummy, fps, metas,
                                  {"train": pairs_tr_eval, "val": pairs_va, "test": pairs_te},
                                  abs_probe)
        row = {"epoch": ep, "loss": tot / max(cnt, 1)}
        for split in ("train", "val", "test"):
            row[f"{split}_top1"] = R[split]["top1"][0]
            row[f"{split}_rho"] = R[split]["seg_rho"][0]
            row[f"{split}_recall_ev"] = R[split]["recall_event"][0]
        d = R["test"]["delta"]
        num = sum(a * n for k, (a, n) in d.items() if k >= 2)
        den = sum(n for k, (a, n) in d.items() if k >= 2)
        row["test_d2"] = num / den if den else None
        history.append(row)
        print(f"ep{ep} loss {row['loss']:.4f} | test top1 {row['test_top1']:.3f} "
              f"rho {row['test_rho']:.3f} recall {row['test_recall_ev']:.3f}", flush=True)

        torch.save({"lora": {n: p for n, p in model.state_dict().items()
                             if "lora" in n.lower() or (".A." in n or ".B." in n)},
                    "head": head.state_dict()}, out / f"ckpt_ep{ep}.pt")
        with open(out / f"scores_ep{ep}.csv", "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["fingerprint", "event", "seg_id", "split", "rating", "score"])
            for fp, m in zip(fps, metas):
                wr.writerow([fp, m["event"], m["seg"], m["split"], m["rating"], f"{s_of[fp]:.4f}"])
        report.append(f"\n# Epoch {ep}（loss {row['loss']:.4f}）\n")
        report.extend(report_block(R, emerg))
        strata = cos_strata(args.m3, meta_by_fp, s_of)
        row["strata"] = {k: round(v[0], 4) for k, v in strata.items()}
        report.append("- test window 对按 cos 分层: " + " · ".join(
            f"{k} {a:.3f}(n={n})" for k, (a, n) in strata.items()))
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (out / "m5_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"\n[OK] {out}/m5_report.md + history.json + ckpt_ep*.pt + scores_ep*.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
