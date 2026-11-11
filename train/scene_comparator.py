"""同场景空间比较器：冻结 DINOv3 patch 网格 + 交叉注意力，判断同段相似两张照片哪张星级档更高。

实验登记见 EXECUTION-LOG 2026-11-11。Stage A：train 事件三折留出的跨档相似对准确率，与单图 joint 头、
同结构 CLS 比较器和 patch 逐图置换比较器对照；Stage B：对同段相似邻居的平均比较优势与单图四档概率
线性融合，报完整事件 AP/四档；--open-val 只在两道 train 闸均通过后执行一次，参数全部固定，不打开 test。
星级只按 min(星,3) 档序比较，不作数值回归；邻居只按无标签 DINO 余弦选取。
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
import torch
from threadpoolctl import threadpool_limits
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "audit"))
import rating_baseline as rb
from evaluation_protocol import digest, load_scores, write_json

SIMILAR = 0.88
SEEDS = (0, 1, 2)
ARMS = ("patch", "cls", "shuffle")
STEPS, BATCH, WIDTH, LEARNING_RATE, DECAY, WARMUP = 1500, 128, 96, 3e-4, 0.05, 100
GAMMAS = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5)
BOOTSTRAP_SEED = 20261111
GATE_A = dict(accuracy=0.63, kill_below=0.62)
GATE_B = dict(ap3=0.02, ap1=0.03, balanced_tolerance=0.0)


class Comparator(nn.Module):
    """反对称比较器 f(A,B)=h(A|B)-h(B|A)；h 让 A 的每个 token 对 B 做交叉注意力，读出差异后注意力池化。"""

    def __init__(self, tokens: int, width: int, shuffle: bool):
        super().__init__()
        self.norm = nn.LayerNorm(384)
        self.project = nn.Linear(384, width)
        self.position = nn.Parameter(torch.randn(tokens, width) * 0.02)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.readout = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(), nn.Dropout(0.1), nn.Linear(width, 1))
        self.pool = nn.Linear(3 * width, 1)
        self.shuffle = shuffle

    def embed(self, tokens: torch.Tensor, generator) -> torch.Tensor:
        """投影 token 并加可学习位置；置换对照在加位置前逐图随机打乱 token 顺序，抹去空间排布。"""
        x = self.project(self.norm(tokens.float()))
        if self.shuffle:
            order = torch.argsort(torch.rand(x.shape[:2], device=x.device, generator=generator), dim=1)
            x = torch.gather(x, 1, order[..., None].expand_as(x))
        return x + self.position

    def half(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """返回 x 相对 y 的标量优势：逐 token 在 y 中软匹配，读出 [自身, 差, 积] 后按学习权重池化。"""
        query = self.query(x)
        attended = F.scaled_dot_product_attention(query.unsqueeze(1), self.key(y).unsqueeze(1),
            y.to(query.dtype).unsqueeze(1)).squeeze(1).to(x.dtype)
        joined = torch.cat((x, x - attended, x * attended), dim=-1)
        weights = torch.softmax(self.pool(joined).squeeze(-1).float(), dim=1)
        return (weights * self.readout(joined).squeeze(-1).float()).sum(dim=1)

    def forward(self, left: torch.Tensor, right: torch.Tensor, generator=None) -> torch.Tensor:
        """返回 left 档位高于 right 的 logit；交换输入严格取负。"""
        x, y = self.embed(left, generator), self.embed(right, generator)
        return self.half(x, y) - self.half(y, x)


class TokenStore:
    """常驻 GPU 的输入张量（patch 臂 fp16 [n,1024,384]，CLS 臂 [n,1,384]），按全局照片下标取批。"""

    def __init__(self, data: torch.Tensor, rows: np.ndarray, total: int):
        lookup = np.full(total, -1, dtype=np.int64)
        lookup[rows] = np.arange(len(rows))
        self.data, self.lookup = data, torch.from_numpy(lookup).cuda()

    def __call__(self, indices: np.ndarray) -> torch.Tensor:
        """返回给定全局下标的 token；缺失照片直接报错，不静默补零。"""
        rows = self.lookup[torch.as_tensor(indices, device="cuda")]
        if bool((rows < 0).any()):
            raise ValueError("输入缺少所需照片")
        return self.data[rows]


class PairSampler:
    """按事件等权、事件内照片等权抽取跨档相似对，返回 (高档, 低档) 全局下标；同档对不参与训练。"""

    def __init__(self, left: np.ndarray, right: np.ndarray, tiers: np.ndarray, events: np.ndarray):
        cross = tiers[left] != tiers[right]
        upper = tiers[left] > tiers[right]
        high, low = np.where(upper, left, right)[cross], np.where(upper, right, left)[cross]
        partners = defaultdict(list)
        for winner, loser in zip(high.tolist(), low.tolist()):
            partners[winner].append((loser, True))
            partners[loser].append((winner, False))
        by_event = defaultdict(list)
        for photo in partners:
            by_event[events[photo]].append(photo)
        self.events = sorted(by_event)
        self.photos = dict((event, np.array(sorted(photos))) for event, photos in by_event.items())
        self.partners = dict((photo, (np.array([other for other, _ in pairs]), np.array([win for _, win in pairs])))
            for photo, pairs in partners.items())
        self.pairs = int(len(high))

    def draw(self, rng: np.random.Generator, size: int) -> tuple[np.ndarray, np.ndarray]:
        """抽取 size 个有向跨档对。"""
        high, low = np.empty(size, dtype=np.int64), np.empty(size, dtype=np.int64)
        for slot in range(size):
            event = self.events[rng.integers(len(self.events))]
            photo = self.photos[event][rng.integers(len(self.photos[event]))]
            others, wins = self.partners[photo]
            pick = rng.integers(len(others))
            high[slot], low[slot] = (photo, others[pick]) if wins[pick] else (others[pick], photo)
        return high, low


def similar_edges(segment_members: list[np.ndarray], dino: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回同段 DINO 余弦 ≥SIMILAR 的无向边（全局下标 i<j 顺序）与余弦；不读标签。"""
    left, right, cosine = [], [], []
    for members in segment_members:
        if len(members) < 2:
            continue
        similarity = dino[members] @ dino[members].T
        first, second = np.triu_indices(len(members), 1)
        keep = similarity[first, second] >= SIMILAR
        left.append(members[first[keep]])
        right.append(members[second[keep]])
        cosine.append(similarity[first, second][keep])
    return np.concatenate(left), np.concatenate(right), np.concatenate(cosine)


def segments_of(indices: np.ndarray, segment_key: np.ndarray) -> list[np.ndarray]:
    """把给定照片按事件内拍摄段分组。"""
    groups = defaultdict(list)
    for index in indices.tolist():
        groups[segment_key[index]].append(index)
    return [np.array(sorted(members)) for _, members in sorted(groups.items())]


def load_patch_store(cache: Path, fingerprints: list[str], needed: np.ndarray) -> TokenStore:
    """从已完成的 patch 缓存把所需照片按块搬到 GPU，返回 TokenStore。"""
    cached = json.loads((cache / "fingerprints.json").read_text(encoding="utf-8"))
    row_of = dict((fingerprint, row) for row, fingerprint in enumerate(cached))
    array = np.load(cache / "patches.f16.npy", mmap_mode="r")
    needed = np.sort(needed)
    data = torch.empty((len(needed), 1024, 384), dtype=torch.float16, device="cuda")
    for start in range(0, len(needed), 256):
        chunk = needed[start:start + 256]
        rows = np.array([row_of[fingerprints[index]] for index in chunk])
        order = np.argsort(rows)
        block = np.empty((len(rows), 1024, 384), dtype=np.float16)
        block[order] = array[rows[order]]
        data[start:start + len(chunk)] = torch.from_numpy(block).cuda()
    return TokenStore(data, needed, len(fingerprints))


def train_model(arm: str, store: TokenStore, sampler: PairSampler, seed: int,
        steps: int = STEPS) -> tuple[Comparator, dict]:
    """固定步数训练一个比较器（无早停），返回模型与损失/拟合对准确率记录。"""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    tokens = 1 if arm == "cls" else 1024
    model = Comparator(tokens, WIDTH, arm == "shuffle").cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=DECAY)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / WARMUP))
    losses = torch.zeros(steps, device="cuda")
    started = time.monotonic()
    model.train()
    for step in range(steps):
        high, low = sampler.draw(rng, BATCH)
        swap = rng.random(BATCH) < 0.5
        left, right = np.where(swap, low, high), np.where(swap, high, low)
        target = torch.as_tensor((~swap).astype(np.float32), device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(store(left), store(right), generator)
        loss = F.binary_cross_entropy_with_logits(logit.float(), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
        losses[step] = loss.detach()
    losses = losses.cpu().numpy()
    if not np.isfinite(losses).all():
        raise ValueError(f"{arm} 训练损失非有限")
    high, low = sampler.draw(np.random.default_rng(seed + 500), 4096)
    fitted = score_pairs(model, store, high, low, seed)
    return model, {"arm": arm, "seed": seed, "steps": steps, "seconds": round(time.monotonic() - started, 1),
        "loss_first100": float(losses[:100].mean()), "loss_last100": float(losses[-100:].mean()),
        "fitting_pair_accuracy": float(np.mean((fitted > 0) + 0.5 * (fitted == 0)))}


@torch.no_grad()
def score_pairs(model: Comparator, store: TokenStore, left: np.ndarray, right: np.ndarray, seed: int) -> np.ndarray:
    """推理期为每条边打 f(left,right)；置换臂用固定生成器，结果可复现。"""
    model.eval()
    generator = torch.Generator(device="cuda")
    generator.manual_seed(10_000 + seed)
    result = np.empty(len(left), dtype=np.float64)
    for start in range(0, len(left), 256):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            values = model(store(left[start:start + 256]), store(right[start:start + 256]), generator)
        result[start:start + 256] = values.float().cpu().numpy()
    model.train()
    if not np.isfinite(result).all():
        raise ValueError("比较器输出非有限")
    return result


def pair_accuracy(score: np.ndarray, left: np.ndarray, right: np.ndarray, tiers: np.ndarray,
        events: np.ndarray, boundary: int | None = None) -> dict:
    """跨档对准确率（事件宏平均与合并）；boundary 只看跨越 ≥k 门槛的对；分数相等记 0.5。"""
    if boundary is None:
        cross = tiers[left] != tiers[right]
        truth = np.sign(tiers[left] - tiers[right])
    else:
        upper_left, upper_right = tiers[left] >= boundary, tiers[right] >= boundary
        cross = upper_left != upper_right
        truth = np.where(upper_left, 1, -1)
    sign = np.sign(score[cross])
    correct = np.where(sign == 0, 0.5, (sign == truth[cross]).astype(np.float64))
    owner = events[left[cross]]
    by_event = dict((event, float(correct[owner == event].mean())) for event in sorted(set(owner.tolist())))
    return {"pairs": int(cross.sum()), "pooled": float(correct.mean()),
        "event_macro": float(np.mean(list(by_event.values()))), "by_event": by_event}


def bootstrap(candidate: dict, reference: dict) -> dict:
    """同事件差的固定种子 bootstrap，返回均差、95% 区间与正向事件数。"""
    events = sorted(candidate)
    if events != sorted(reference):
        raise ValueError("对照事件不一致")
    differences = np.array([candidate[event] - reference[event] for event in events])
    samples = np.random.default_rng(BOOTSTRAP_SEED).choice(differences, size=(10000, len(differences))).mean(axis=1)
    return {"mean": float(differences.mean()), "event_bootstrap95": np.quantile(samples, (0.025, 0.975)).tolist(),
        "positive_events": int((differences > 0).sum()), "events": len(differences)}


def context_signal(edge_score: np.ndarray, left: np.ndarray, right: np.ndarray, total: int) -> tuple[np.ndarray, np.ndarray]:
    """每张照片对其同段相似邻居的平均比较优势；无邻居记 0，同时返回邻居数。"""
    sums, counts = np.zeros(total), np.zeros(total)
    np.add.at(sums, left, edge_score)
    np.add.at(sums, right, -edge_score)
    np.add.at(counts, left, 1)
    np.add.at(counts, right, 1)
    return np.divide(sums, counts, out=np.zeros(total), where=counts > 0), counts


def tilt(probabilities: np.ndarray, signal: np.ndarray, gamma: float) -> np.ndarray:
    """在四档对数概率上加 γ·signal·档序后重新归一化；γ=0 即原单图头，不改档位定义。"""
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) + gamma * signal[:, None] * np.arange(4.0)
    logits -= logits.max(axis=1, keepdims=True)
    fused = np.exp(logits)
    return fused / fused.sum(axis=1, keepdims=True)


def measure_rows(metadata: dict, fingerprints: list[str], probabilities: np.ndarray, indices: np.ndarray,
        events: list[str]) -> dict:
    """用统一协议评价指定照片子集的四档概率。"""
    return rb.measure(metadata, dict((fingerprints[index], probabilities[index]) for index in indices), events)


def selection_key(metrics: dict, gamma: float) -> tuple:
    """γ 选择：拟合事件 AP≥3 优先，同分比四档平均召回，再取较小 γ。"""
    return (metrics["ranking"]["3"]["ap_macro"], metrics["rating"]["balanced_accuracy_macro"], -gamma)


def choose_gamma(metadata, fingerprints, probabilities, signal, event_of, events) -> tuple[float, list]:
    """只在给定事件上逐 γ 评价融合结果，返回最优 γ 与完整网格记录。"""
    indices = np.flatnonzero(np.isin(event_of, events))
    grid = []
    for gamma in GAMMAS:
        metrics = measure_rows(metadata, fingerprints, tilt(probabilities, signal, gamma), indices, events)
        grid.append((selection_key(metrics, gamma), gamma, metrics["ranking"]["3"]["ap_macro"],
            metrics["ranking"]["1"]["ap_macro"], metrics["rating"]["balanced_accuracy_macro"]))
    best = max(grid)
    return best[1], [{"gamma": row[1], "ap3": row[2], "ap1": row[3], "balanced": row[4]} for row in grid]


def compare_metrics(candidate: dict, reference: dict) -> dict:
    """逐边界 AP 与四档平均召回的事件配对差。"""
    result = dict()
    for threshold in ("1", "2", "3"):
        result[f"ap{threshold}"] = bootstrap(
            dict((event, row["ap"]) for event, row in candidate["ranking"][threshold]["by_event"].items()),
            dict((event, row["ap"]) for event, row in reference["ranking"][threshold]["by_event"].items()))
    result["balanced_accuracy"] = bootstrap(
        dict((event, row["balanced_accuracy"]) for event, row in candidate["rating"]["by_event"].items()),
        dict((event, row["balanced_accuracy"]) for event, row in reference["rating"]["by_event"].items()))
    return result


def headline(metrics: dict) -> dict:
    """摘取三条边界 AP（含随机/闭合）、四档平均与逐档召回及 12.5% 预算精度/召回。"""
    ranking = metrics["ranking"]
    budget = metrics["budget"]["0.125"]
    return {**dict((f"ap{k}", ranking[k]["ap_macro"]) for k in ("1", "2", "3")),
        **dict((f"random_ap{k}", ranking[k]["random_ap_macro"]) for k in ("1", "2", "3")),
        "ap3_closure": ranking["3"]["ap_closure"], "balanced_accuracy": metrics["rating"]["balanced_accuracy_macro"],
        "class_recall": metrics["rating"]["class_recall_macro"], "precision_at_12.5": budget["precision_macro"],
        "recall_at_12.5": budget["photo_recall_macro"], "random_precision_at_12.5": budget["random_precision_macro"]}


def main() -> int:
    """按登记顺序执行 Stage A → B →（可选）val，每道闸自动判定，不据 val/test 调参。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patches", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--open-val", action="store_true")
    parser.add_argument("--steps", type=int, default=STEPS,
        help="只用于登记的训练预算稳健性复核；非默认值时只报 Stage A，不执行 Stage B / val")
    cli = parser.parse_args()
    inputs = argparse.Namespace(photos=ROOT / "audit/out/m3_pairs_v2_20260924b/photos.csv",
        split_json=ROOT / "data/split-v2-20260924.json", features=ROOT / "train/out/recovery_work/corrected_features.npz",
        local=ROOT / "train/out/recovery_work/rescored_lora_v2/scores.csv",
        global_scores=ROOT / "train/out/recovery_work/corrected_rank/clip_scores.csv")
    formal = ROOT / "train/out/rating_baseline_20261014_balanced"
    metadata, fingerprints, features, _ = rb.load_inputs(inputs)
    patch_identity = json.loads((cli.patches / "identity.json").read_text(encoding="utf-8"))
    patch_complete = json.loads((cli.patches / "complete.json").read_text(encoding="utf-8"))
    if patch_identity["decode_version"] != "display-plane-v3" or patch_identity["features_sha256"] != digest(inputs.features):
        raise ValueError("patch 缓存身份与修复特征不一致")
    if digest(cli.patches / "patches.f16.npy") != patch_complete["patch_sha256"]:
        raise ValueError("patch 缓存内容哈希不符")
    cli.out.mkdir(parents=True, exist_ok=False)
    hashed = [inputs.photos, inputs.split_json, inputs.features, Path(__file__), ROOT / "train/rating_baseline.py",
        ROOT / "audit/evaluation_protocol.py", formal / "joint_scores.csv", formal / "joint_0.1_oof.json"]
    plan = {"registration": "EXECUTION-LOG 2026-11-11 同场景差异定位与空间比较器", "similar_cosine": SIMILAR,
        "arms": ARMS, "seeds": SEEDS, "steps": cli.steps, "batch": BATCH, "width": WIDTH, "learning_rate": LEARNING_RATE,
        "weight_decay": DECAY, "warmup": WARMUP, "gammas": GAMMAS, "gate_a": GATE_A, "gate_b": GATE_B,
        "patch_cache": str(cli.patches), "patch_sha256": patch_complete["patch_sha256"],
        "input_sha256": dict((str(path.resolve()), digest(path)) for path in hashed)}
    write_json(cli.out / "plan.json", plan)
    result = {"plan": plan}

    tiers = np.array([min(int(metadata[fp]["rating_raw"]), 3) for fp in fingerprints])
    event_of = np.array([metadata[fp]["event"] for fp in fingerprints])
    split_of = np.array([metadata[fp]["split"] for fp in fingerprints])
    segment_key = np.array([metadata[fp]["event"] + "|" + metadata[fp]["seg_id"] for fp in fingerprints])
    dino = features["dino"] / np.sqrt(384)
    train_events = sorted(set(event_of[split_of == "train"].tolist()),
        key=lambda event: hashlib.sha256(("20260926:" + event).encode()).hexdigest())
    folds = [train_events[fold::3] for fold in range(3)]
    train_indices = np.flatnonzero(split_of == "train")

    baseline = np.full((len(fingerprints), 4), np.nan)
    threadpool_limits(2)
    for held in folds:
        fitting = np.isin(event_of, [event for event in train_events if event not in held])
        testing = np.flatnonzero(np.isin(event_of, held))
        coefficient, _ = rb.fit_classifier(features["joint"][fitting], tiers[fitting], event_of[fitting].tolist(),
            0.1, "event_class")
        baseline[testing] = rb.predict(features["joint"][testing], coefficient)
    baseline_metrics = measure_rows(metadata, fingerprints, baseline, train_indices, train_events)
    formal_metrics = json.loads((formal / "joint_0.1_oof.json").read_text(encoding="utf-8"))["metrics"]
    for threshold in ("1", "2", "3"):
        if abs(baseline_metrics["ranking"][threshold]["ap_macro"] - formal_metrics["ranking"][threshold]["ap_macro"]) > 1e-6:
            raise ValueError("单图基线 OOF 未复现正式类别均衡臂")
    result["baseline_train_oof"] = headline(baseline_metrics)
    print(f"baseline reproduced {result['baseline_train_oof']}", flush=True)

    left, right, cosine = similar_edges(segments_of(train_indices, segment_key), dino)
    expected = baseline @ np.arange(4.0)
    needed = np.unique(np.concatenate((left, right)))
    print(f"train edges {len(left)} cross {int((tiers[left] != tiers[right]).sum())} photos {len(needed)}", flush=True)
    stores = {"cls": TokenStore(torch.from_numpy(dino[:, None, :] * np.sqrt(384)).float().cuda(),
        np.arange(len(fingerprints)), len(fingerprints))}
    stores["patch"] = load_patch_store(cli.patches, fingerprints, needed)
    stores["shuffle"] = stores["patch"]
    print(f"patch store on GPU {torch.cuda.memory_allocated() / 2 ** 30:.1f} GiB", flush=True)

    edge_scores = dict((arm, np.full((len(SEEDS), len(left)), np.nan)) for arm in ARMS)
    traces = []
    for fold, held in enumerate(folds):
        fitting_events = [event for event in train_events if event not in held]
        fit_edges = np.isin(event_of[left], fitting_events)
        held_edges = np.flatnonzero(np.isin(event_of[left], held))
        sampler = PairSampler(left[fit_edges], right[fit_edges], tiers, event_of)
        for arm in ARMS:
            for slot, seed in enumerate(SEEDS):
                model, trace = train_model(arm, stores[arm], sampler, seed + 100 * fold, cli.steps)
                edge_scores[arm][slot, held_edges] = score_pairs(model, stores[arm], left[held_edges],
                    right[held_edges], seed + 100 * fold)
                trace.update(fold=fold, held_out_events=held, fitting_pairs=sampler.pairs)
                traces.append(trace)
                print(f"fold {fold} {arm} seed {seed}: {trace}", flush=True)
                del model
    if any(np.isnan(scores).any() for scores in edge_scores.values()):
        raise ValueError("留出边未全部打分")
    np.savez(cli.out / "edge_scores.npz", left=left, right=right, cosine=cosine, baseline_expected_tier=expected,
        **dict((arm, scores) for arm, scores in edge_scores.items()))

    stage_a = {"traces": traces, "arms": dict()}
    references = {"single_image_joint": expected[left] - expected[right]}
    for arm in ARMS:
        references[arm] = edge_scores[arm].mean(axis=0)
    for name, score in references.items():
        entry = {"overall": pair_accuracy(score, left, right, tiers, event_of)}
        for boundary in (1, 2, 3):
            boundary_score = score
            if name == "single_image_joint":
                upper = baseline[:, boundary:].sum(axis=1)
                boundary_score = upper[left] - upper[right]
            entry[f"boundary{boundary}"] = pair_accuracy(boundary_score, left, right, tiers, event_of, boundary)
        if name in ARMS:
            entry["per_seed"] = [pair_accuracy(edge_scores[name][slot], left, right, tiers, event_of)["event_macro"]
                for slot in range(len(SEEDS))]
        stage_a["arms"][name] = entry
    patch_events = stage_a["arms"]["patch"]["overall"]["by_event"]
    stage_a["comparisons"] = dict((f"patch_minus_{name}", bootstrap(patch_events, stage_a["arms"][name]["overall"]["by_event"]))
        for name in ("single_image_joint", "cls", "shuffle"))
    accuracy = stage_a["arms"]["patch"]["overall"]["event_macro"]
    lower_bounds = [row["event_bootstrap95"][0] for row in stage_a["comparisons"].values()]
    passed_a = accuracy >= GATE_A["accuracy"] and all(bound > 0 for bound in lower_bounds)
    stage_a["gate"] = {"patch_event_macro": accuracy, "lower_bounds": lower_bounds, "passed": passed_a,
        "killed": accuracy < GATE_A["kill_below"] or any(bound <= 0 for bound in lower_bounds)}
    result["stage_a"] = stage_a
    write_json(cli.out / "result.json", result)
    print(f"STAGE A gate {stage_a['gate']}", flush=True)
    for name, entry in stage_a["arms"].items():
        print(f"  {name}: event-macro {entry['overall']['event_macro']:.4f} pooled {entry['overall']['pooled']:.4f} "
            f"b1 {entry['boundary1']['event_macro']:.4f} b3 {entry['boundary3']['event_macro']:.4f}", flush=True)
    if cli.steps != STEPS:
        print("训练预算复核只报 Stage A，不执行 Stage B / val", flush=True)
        return 0
    if not passed_a:
        print("Stage A 未过闸，按登记停止，不执行 Stage B / val", flush=True)
        return 0

    signal, counts = context_signal(references["patch"], left, right, len(fingerprints))
    fused = baseline.copy()
    stage_b = {"folds": []}
    for held in folds:
        fitting_events = [event for event in train_events if event not in held]
        fitting_photos = np.isin(event_of, fitting_events) & (counts > 0)
        scale = float(signal[fitting_photos].std())
        gamma, grid = choose_gamma(metadata, fingerprints, baseline, signal / scale, event_of, fitting_events)
        held_photos = np.flatnonzero(np.isin(event_of, held))
        fused[held_photos] = tilt(baseline[held_photos], signal[held_photos] / scale, gamma)
        stage_b["folds"].append({"held_out_events": held, "scale": scale, "gamma": gamma, "grid": grid})
    fused_metrics = measure_rows(metadata, fingerprints, fused, train_indices, train_events)
    stage_b["fused_train_oof"] = headline(fused_metrics)
    stage_b["baseline_train_oof"] = result["baseline_train_oof"]
    stage_b["differences"] = compare_metrics(fused_metrics, baseline_metrics)
    stage_b["photos_with_neighbours"] = float((counts[train_indices] > 0).mean())
    differences = stage_b["differences"]
    passed_b = (differences["ap3"]["mean"] >= GATE_B["ap3"] and differences["ap3"]["event_bootstrap95"][0] > 0
        and differences["ap1"]["mean"] >= GATE_B["ap1"] and differences["ap1"]["event_bootstrap95"][0] > 0
        and differences["balanced_accuracy"]["mean"] >= GATE_B["balanced_tolerance"])
    stage_b["gate"] = {"passed": passed_b}
    result["stage_b"] = stage_b
    write_json(cli.out / "result.json", result)
    print(f"STAGE B fused {stage_b['fused_train_oof']}\n  differences {differences}\n  gate {passed_b}", flush=True)
    if not passed_b or not cli.open_val:
        print("Stage B 未过闸或未授权 val，停止", flush=True)
        return 0

    full_scale = float(signal[train_indices][counts[train_indices] > 0].std())
    val_gamma, val_grid = choose_gamma(metadata, fingerprints, baseline, signal / full_scale, event_of, train_events)
    full_sampler = PairSampler(left, right, tiers, event_of)
    models = [train_model("patch", stores["patch"], full_sampler, 900 + seed)[0] for seed in SEEDS]
    del stores
    torch.cuda.empty_cache()
    val_indices = np.flatnonzero(split_of == "val")
    val_events = sorted(set(event_of[val_indices].tolist()))
    val_left, val_right, val_cosine = similar_edges(segments_of(val_indices, segment_key), dino)
    val_store = load_patch_store(cli.patches, fingerprints, np.unique(np.concatenate((val_left, val_right))))
    val_edge = np.mean([score_pairs(model, val_store, val_left, val_right, 900 + seed)
        for model, seed in zip(models, SEEDS)], axis=0)
    formal_scores = dict()
    with (formal / "joint_scores.csv").open(encoding="utf-8") as stream:
        header = stream.readline().strip().split(",")
        for line in stream:
            values = line.strip().split(",")
            formal_scores[values[0]] = np.array([float(values[header.index(name)]) for name in ("p0", "p1", "p2", "p_ge3")])
    val_baseline = np.full((len(fingerprints), 4), np.nan)
    for index in val_indices:
        val_baseline[index] = formal_scores[fingerprints[index]]
    val_signal, val_counts = context_signal(val_edge, val_left, val_right, len(fingerprints))
    val_fused = val_baseline.copy()
    val_fused[val_indices] = tilt(val_baseline[val_indices], val_signal[val_indices] / full_scale, val_gamma)
    val_baseline_metrics = measure_rows(metadata, fingerprints, val_baseline, val_indices, val_events)
    val_fused_metrics = measure_rows(metadata, fingerprints, val_fused, val_indices, val_events)
    val_expected = val_baseline @ np.arange(4.0)
    result["val"] = {"gamma": val_gamma, "scale": full_scale, "grid_on_train_oof": val_grid,
        "edges": int(len(val_left)), "photos_with_neighbours": float((val_counts[val_indices] > 0).mean()),
        "baseline": headline(val_baseline_metrics), "fused": headline(val_fused_metrics),
        "differences": compare_metrics(val_fused_metrics, val_baseline_metrics),
        "pair_accuracy": {"patch": pair_accuracy(val_edge, val_left, val_right, tiers, event_of),
            "single_image_joint": pair_accuracy(val_expected[val_left] - val_expected[val_right], val_left, val_right,
                tiers, event_of)}}
    np.savez(cli.out / "val_edge_scores.npz", left=val_left, right=val_right, cosine=val_cosine, patch=val_edge)
    write_json(cli.out / "result.json", result)
    print(f"VAL {result['val']['baseline']}\n -> {result['val']['fused']}\n  diff {result['val']['differences']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
