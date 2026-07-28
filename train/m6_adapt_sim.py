"""
m6_adapt_sim.py — 事件内自适应验证 + 事件内干净可学性测试（用户两质疑的合流实验）

背景：跨场迁移全灭（梯1-4）；唯一真实弱信号住相似带（LoRA CLS 0.586-0.588）；
"事件内可学性 79-94%"是泄漏+内容混杂上界，不能作为自适应路线的依据。
本实验在 test 事件上模拟产品新工作流，用**诚实切分**回答：
    抽 k 张"用户已标"（团顶主动采样 vs 随机对照）→ 只微调评分头（backbone+LoRA 冻结）
    → 测**剩余未标照片**的对级一致率（cos 分层）/ 段内 top-1 / seg-rho / recall@12.5%。
对照 = 同一模型零样本（不微调）在同一批未标照片上的同样指标。

起点模型 = 梯4 CLS LoRA ep3（m5_lora 产物）；评估口径与 m4/m5 同族。

用法（仓根 D:/Git/PhotoViewer 下）：
    PYTHONUTF8=1 Tools/.venv-gpu/Scripts/python.exe Training/train/m6_adapt_sim.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m4_baseline import M3_DIR_DEFAULT  # noqa: E402
from m5_lora import CACHE_DEFAULT, MID_DEFAULT, build_model, forward_scores, score_all  # noqa: E402

CKPT_DEFAULT = "Training/train/out/m5_lora/ckpt_ep3.pt"
OUT_DEFAULT = "Training/train/out/m6_adapt_sim"
KS = (25, 50, 100)
SEEDS = (0, 1, 2)
ADAPT_STEPS = 100
ADAPT_LR = 1e-3

# ---------------------------------------------------------------------------


def load_meta(m3_dir):
    photos = {r["fingerprint"]: dict(
        event=r["event"], seg=int(r["seg_id"]), cluster=int(r["cluster_id"]),
        is_top=int(r["is_cluster_top"]), rating=int(r["rating_raw"]),
        split=r["split"])
        for r in csv.DictReader(open(f"{m3_dir}/photos.csv", encoding="utf-8-sig"))}
    pairs = [(r["fp_i"], r["fp_j"], r["ptype"], float(r["cos"]), float(r["weight"]))
             for r in csv.DictReader(open(f"{m3_dir}/pairs_test.csv", encoding="utf-8-sig"))]
    return photos, pairs


def sample_labeled(fps, photos, k, seed, mode):
    """k 张"用户已标"。active = 段×团轮询团顶优先（模拟主动采样）；random = 均匀随机。"""
    rng = np.random.default_rng(seed)
    if mode == "random":
        idx = rng.choice(len(fps), size=min(k, len(fps)), replace=False)
        return {fps[i] for i in idx}
    # active: 按段分组，段内团顶排前，团内其余随后；段间轮转取，直到 k
    by_seg = defaultdict(list)
    for fp in fps:
        by_seg[photos[fp]["seg"]].append(fp)
    queues = []
    for seg, members in by_seg.items():
        rng.shuffle(members)
        members.sort(key=lambda f: -photos[f]["is_top"])
        queues.append(members)
    picked, i = [], 0
    while len(picked) < k and any(queues):
        q = queues[i % len(queues)]
        if q:
            picked.append(q.pop(0))
        i += 1
    return set(picked)


def build_feat_cache(model, fps, cache, device):
    """预算每张图 head 的输入特征（CLS 头的输入 = CLS token）→ {fp: np[384]}。"""
    import torch
    feats = {}
    from torch.utils.data import DataLoader
    from m5_lora import FpCollate, FpDataset
    loader = DataLoader(FpDataset(fps), batch_size=64, shuffle=False, num_workers=6,
                        collate_fn=FpCollate(cache), pin_memory=True, persistent_workers=True)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    ix = 0
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        for x in loader:
            x = x.to(device).permute(0, 3, 1, 2).float().div_(255.0)
            x = (x - mean) / std
            h = model(pixel_values=x).last_hidden_state
            cls = h[:, 0, :].float().cpu().numpy()
            for j in range(len(cls)):
                feats[fps[ix]] = cls[j]
                ix += 1
    return feats


def adapt_with_feats(head, feats, adapt_pairs, device, steps=ADAPT_STEPS, lr=ADAPT_LR):
    import torch
    xi = torch.tensor(np.stack([feats[p[0]] for p in adapt_pairs]), device=device)
    xj = torch.tensor(np.stack([feats[p[1]] for p in adapt_pairs]), device=device)
    y = torch.tensor([p[2] for p in adapt_pairs], dtype=torch.float32, device=device)
    w = torch.tensor([p[3] for p in adapt_pairs], dtype=torch.float32, device=device)
    head.train()
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(steps):
        si = head.lin(xi).squeeze(-1)
        sj = head.lin(xj).squeeze(-1)
        loss = (torch.nn.functional.softplus(-(si - sj) * y) * w).sum() / w.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    head.eval()
    return steps


# ---------------------------------------------------------------------------

def seg_metrics(fps, photos, score_of, eval_pairs):
    """对级（全体 + cos 分层）+ 段内 top-1 + seg-rho + recall@12.5%（事件级）。"""
    res = {}
    buckets = ((0.0, 0.88), (0.88, 0.95), (0.95, 0.98), (0.98, 1.01))
    for name, lo, hi in [("all", 0.0, 1.01)] + [(f"[{lo},{hi})", lo, hi) for lo, hi in buckets]:
        ok = [p for p in eval_pairs if lo <= p[3] < hi]
        ok = [int(np.sign(score_of[p[0]] - score_of[p[1]]) == p[2]) for p in ok]
        res[f"pair_{name}"] = (float(np.mean(ok)), len(ok)) if ok else (float("nan"), 0)
    by_seg = defaultdict(list)
    for fp in fps:
        by_seg[photos[fp]["seg"]].append(fp)
    hits = ch = n_top = 0
    rhos = []
    from scipy.stats import spearmanr
    for _, group in by_seg.items():
        rs = np.array([photos[fp]["rating"] for fp in group])
        if len(group) < 2 or rs.max() == rs.min():
            continue
        sc = np.array([score_of[fp] for fp in group])
        if group[int(sc.argmax())] is not None and photos[group[int(sc.argmax())]]["rating"] == rs.max():
            hits += 1
        ch += (rs == rs.max()).mean()
        n_top += 1
        if len(group) >= 5:
            rho = spearmanr(rs, sc).statistic
            if not np.isnan(rho):
                rhos.append(float(rho))
    res["top1"] = (hits / n_top if n_top else float("nan"), ch / n_top if n_top else float("nan"), n_top)
    res["rho"] = (float(np.mean(rhos)), len(rhos))
    k = max(1, round(len(fps) * 0.125))
    top = sorted(fps, key=lambda f: -score_of[f])[:k]
    got = sum(1 for fp in top if photos[fp]["rating"] >= 3)
    need = sum(1 for fp in fps if photos[fp]["rating"] >= 3)
    res["recall"] = (got / need if need else float("nan"), got, need)
    return res


def main() -> int:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    photos, pairs = load_meta(M3_DIR_DEFAULT)
    events = sorted({p["event"] for p in photos.values() if p["split"] == "test"})
    print(f"test 事件: {events}", flush=True)

    model, head, _ = build_model(MID_DEFAULT, device, "cls")
    ckpt = torch.load(CKPT_DEFAULT, map_location=device)
    model.load_state_dict(ckpt["lora"], strict=False)
    head_sd = ckpt["head"]
    if "weight" in head_sd:                     # 梯4 ckpt 是裸 Linear（weight/bias）→ 映射到 _H.lin.*
        head_sd = {f"lin.{k}": v for k, v in head_sd.items()}
    head.load_state_dict(head_sd)
    model.eval()
    head.eval()

    out = Path(OUT_DEFAULT)
    out.mkdir(parents=True, exist_ok=True)
    report = ["# 事件内自适应验证（k 张已标 → 微调头 → 测剩余）\n"]
    all_rows = []
    for ev in events:
        ev_fps = [fp for fp, p in photos.items() if p["event"] == ev]
        ev_pairs = [(fi, fj, int(np.sign(photos[fi]["rating"] - photos[fj]["rating"])), cos, w)
                    for fi, fj, pt, cos, w in pairs
                    if pt in ("window", "global") and fi in photos and fj in photos
                    and photos[fi]["event"] == ev and photos[fj]["event"] == ev]
        ev_pairs = [p for p in ev_pairs if p[2] != 0]
        feats = build_feat_cache(model, ev_fps, CACHE_DEFAULT, device)
        zero_scores = {fp: float(head.lin(torch.tensor(feats[fp], device=device)).item())
                       for fp in ev_fps}
        head0 = {k: v.clone() for k, v in head.state_dict().items()}
        eval_fps_all = ev_fps  # 段内指标在全体上算（含已标张，两口径可比）

        report.append(f"\n## {ev}（{len(ev_fps)} 张，对 n={len(ev_pairs)}）\n")
        for k in KS:
            if k > len(ev_fps) * 0.4:          # 小事件跳过：已标不得超过 40%，保评估集效力
                continue
            for mode in ("active", "random"):
                for seed in SEEDS:
                    labeled = sample_labeled(ev_fps, photos, k, seed, mode)
                    unlabeled = set(ev_fps) - labeled
                    # 监督 = 已标×已标两两全对（用户 50 次评级 = 本场自有尺度的 k 个序点，序即监督）
                    ad_pairs = [(fi, fj, int(np.sign(photos[fi]["rating"] - photos[fj]["rating"])), 1.0, 1.0)
                                for fi in labeled for fj in labeled
                                if photos[fi]["rating"] != photos[fj]["rating"]]
                    ev_pairs_un = [p for p in ev_pairs if p[0] in unlabeled and p[1] in unlabeled]
                    head.load_state_dict(head0)
                    adapt_with_feats(head, feats, ad_pairs, device)
                    ad_scores = {fp: float(head.lin(torch.tensor(feats[fp], device=device)).item())
                                 for fp in ev_fps}
                    z = seg_metrics(eval_fps_all, photos, zero_scores, ev_pairs_un)
                    a = seg_metrics(eval_fps_all, photos, ad_scores, ev_pairs_un)
                    row = dict(event=ev, k=k, mode=mode, seed=seed, n_adapt=len(ad_pairs),
                               n_eval=len(ev_pairs_un),
                               z_pair=z["pair_all"][0], a_pair=a["pair_all"][0],
                               z_hi=z["pair_[0.88,0.95)"][0], a_hi=a["pair_[0.88,0.95)"][0],
                               z_hi2=z["pair_[0.95,0.98)"][0], a_hi2=a["pair_[0.95,0.98)"][0],
                               z_top1=z["top1"][0], a_top1=a["top1"][0],
                               z_rho=z["rho"][0], a_rho=a["rho"][0],
                               z_recall=z["recall"][0], a_recall=a["recall"][0])
                    all_rows.append(row)
                    print(f"{ev} k={k} {mode} s{seed} | pair {row['z_pair']:.3f}→{row['a_pair']:.3f} "
                          f"sim095 {row['z_hi']:.3f}→{row['a_hi']:.3f} top1 {row['z_top1']:.3f}→{row['a_top1']:.3f} "
                          f"recall {row['z_recall']:.3f}→{row['a_recall']:.3f} (adapt n={len(ad_pairs)})", flush=True)
        # 事件小结
        rows_ev = [r for r in all_rows if r["event"] == ev]
        report.append("| k | 采样 | adapt n | 对级 零样本→适配 | 相似带 零样本→适配 | top-1 零样本→适配 | recall 零样本→适配 |")
        report.append("|---|---|---|---|---|---|---|")
        for k in KS:
            for mode in ("active", "random"):
                rs = [r for r in rows_ev if r["k"] == k and r["mode"] == mode]
                if not rs:
                    continue
                m = lambda key: (float(np.nanmean([r[f"z_{key}"] for r in rs])),
                                 float(np.nanmean([r[f"a_{key}"] for r in rs])))
                report.append(
                    f"| {k} | {mode} | {int(np.mean([r['n_adapt'] for r in rs]))} | "
                    f"{m('pair')[0]:.3f}→{m('pair')[1]:.3f} | {m('hi')[0]:.3f}→{m('hi')[1]:.3f} | "
                    f"{m('top1')[0]:.3f}→{m('top1')[1]:.3f} | {m('recall')[0]:.3f}→{m('recall')[1]:.3f} |")
        report.append("")
        head.load_state_dict(head0)

    (out / "rows.json").write_text(json.dumps(all_rows, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "m6_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\n[OK] {out}/m6_report.md + rows.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
