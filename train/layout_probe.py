"""内容无关布局描述读团内取舍：只用单图 patch 两两相似度与网格位置构造构图量，检验同场景取舍能否跨事件迁移。

实验登记见 EXECUTION-LOG 2026-11-12。描述只依赖逐 token L2 归一化后的单图 Gram 矩阵与 32×32 位置，对特征空间
任意正交旋转不变，不含“是什么”。Stage A：train 事件三折留出的同段 cos≥0.88 跨档对与 golden 盲评对上，
布局分对照位置置换、等维内容摘要、单图 joint 头及其融合；Stage B：逐图布局分倾斜单图四档概率，报完整事件
AP/四档。--open-val 只在两道闸均通过后执行一次，不打开 test。星级只按 min(星,3) 档序比较，不作数值回归。
--control random|truth 把布局描述换成噪声 / 带噪档位，只用于验证管线，输出不参与任何判定。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "audit"))
import rating_baseline as rb
import scene_comparator as sc
from evaluation_protocol import digest, write_json
from golden_label_check import pair_rows, read_groups
from within_scene_audit import baseline_oof

GRID = 32
BATCH = 128
BAND = 3
PENALTIES = (0.001, 0.01, 0.1, 1.0, 10.0)
ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
ARMS = ("layout", "shuffle", "content")
COSINE_BANDS = ((0.88, 0.92), (0.92, 0.95), (0.95, 0.98), (0.98, 1.01))
SHUFFLE_SEED = 20261112
CONTROL_SEED = 7
GATE_A = dict(layout=0.58, kill_below=0.55, golden=0.60)
GATE_B = dict(ap3=0.02)
NAMES = (["salient_cx", "salient_cy", "salient_sx", "salient_sy", "salient_corr"]
    + [f"thirds_{row}{col}" for row in "tmb" for col in "lcr"]
    + ["band_top", "band_bottom", "band_left", "band_right", "salient_entropy", "salient_to_thirds_point",
       "salient_to_center", "salient_offset_x", "salient_offset_y", "rarity_mean", "rarity_std"]
    + [f"spectrum_{k}" for k in range(1, 6)] + ["spectrum_participation", "spectrum_total"]
    + ["fore_area", "fore_cx", "fore_cy", "fore_touch_top", "fore_touch_bottom", "fore_touch_left", "fore_touch_right",
       "fore_components", "fore_largest_share", "fore_box_width", "fore_box_height"]
    + ["neighbour_coherence", "horizontal_line_position", "horizontal_line_strength", "vertical_line_position",
       "vertical_line_strength", "border_boundary", "interior_boundary", "mirror_left_right", "mirror_top_bottom"])


def grid_constants(device: str) -> dict:
    """返回 token 中心坐标、九宫格与四边带的独热矩阵、边界强度计数与边缘带掩码。"""
    row, col = np.divmod(np.arange(GRID * GRID), GRID)
    x, y = (col + 0.5) / GRID, (row + 0.5) / GRID
    thirds = np.zeros((GRID * GRID, 9))
    thirds[np.arange(GRID * GRID), np.minimum((y * 3).astype(int), 2) * 3 + np.minimum((x * 3).astype(int), 2)] = 1
    bands = np.stack((row < BAND, row >= GRID - BAND, col < BAND, col >= GRID - BAND), axis=1).astype(float)
    counts = np.zeros((GRID, GRID))
    counts[:, :-1] += 1
    counts[:, 1:] += 1
    counts[:-1, :] += 1
    counts[1:, :] += 1
    border = (bands.sum(axis=1) > 0).reshape(GRID, GRID)
    as_tensor = lambda value, dtype=torch.float32: torch.as_tensor(value, dtype=dtype, device=device)
    return {"x": as_tensor(x), "y": as_tensor(y), "thirds": as_tensor(thirds), "bands": as_tensor(bands),
        "counts": as_tensor(counts), "border": as_tensor(border, torch.bool)}


def component_stats(foreground: np.ndarray) -> np.ndarray:
    """前景掩码的 4 连通块统计：log(1+块数)、最大块占比、最大块外接框宽高（归一化）。"""
    result = np.zeros((len(foreground), 4))
    for slot, mask in enumerate(foreground):
        labels, count = ndimage.label(mask)
        if count == 0:
            continue
        sizes = np.bincount(labels.ravel())[1:]
        largest = int(np.argmax(sizes)) + 1
        rows, cols = np.nonzero(labels == largest)
        result[slot] = (math.log1p(count), sizes.max() / sizes.sum(),
            (cols.max() - cols.min() + 1) / GRID, (rows.max() - rows.min() + 1) / GRID)
    return result


@torch.no_grad()
def describe(tokens: torch.Tensor, constants: dict) -> np.ndarray:
    """一批 [B,1024,384] patch 网格 → [B,52] 布局描述；全部量只由单图 Gram 矩阵与位置决定。"""
    batch = tokens.shape[0]
    unit = tokens.float()
    unit = unit / unit.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    mean = unit.mean(dim=1)
    rarity = 1 - torch.einsum("btd,bd->bt", unit, mean)
    mass = torch.relu(rarity - rarity.median(dim=1, keepdim=True).values) + 1e-6
    mass = mass / mass.sum(dim=1, keepdim=True)
    x, y = constants["x"], constants["y"]
    cx, cy = mass @ x, mass @ y
    dx, dy = x[None] - cx[:, None], y[None] - cy[:, None]
    sx, sy = (mass * dx ** 2).sum(1).sqrt(), (mass * dy ** 2).sum(1).sqrt()
    corr = (mass * dx * dy).sum(1) / (sx * sy).clamp_min(1e-6)
    points = torch.tensor([(a, b) for a in (1 / 3, 2 / 3) for b in (1 / 3, 2 / 3)], device=tokens.device)
    to_point = ((cx[:, None] - points[None, :, 0]) ** 2 + (cy[:, None] - points[None, :, 1]) ** 2).sqrt().min(1).values
    salient = torch.cat((torch.stack((cx, cy, sx, sy, corr), 1), mass @ constants["thirds"], mass @ constants["bands"],
        torch.stack((-(mass * mass.log()).sum(1) / math.log(GRID * GRID), to_point,
        ((cx - 0.5) ** 2 + (cy - 0.5) ** 2).sqrt(), (cx - 0.5).abs(), (cy - 0.5).abs(), rarity.mean(1), rarity.std(1)), 1)), 1)

    centered = unit - mean[:, None, :]
    values, vectors = torch.linalg.eigh(centered.transpose(1, 2) @ centered / (GRID * GRID))
    values = values.flip(-1).clamp_min(0)
    total = values.sum(-1).clamp_min(1e-12)
    spectrum = torch.cat((values[:, :5] / total[:, None],
        torch.stack((total ** 2 / (values ** 2).sum(-1).clamp_min(1e-24) / values.shape[-1], total), 1)), 1)

    projection = torch.einsum("btd,bd->bt", centered, vectors[:, :, -1])
    positive = projection > 0
    mean_rarity = lambda mask: (rarity * mask).sum(1) / mask.sum(1).clamp_min(1)
    foreground = torch.where((mean_rarity(positive) >= mean_rarity(~positive))[:, None], positive, ~positive)
    weight = foreground.float()
    area = weight.sum(1)
    fore_cx = torch.where(area > 0, (weight @ x) / area.clamp_min(1), torch.full_like(area, 0.5))
    fore_cy = torch.where(area > 0, (weight @ y) / area.clamp_min(1), torch.full_like(area, 0.5))
    plane = weight.view(batch, GRID, GRID)
    partition = torch.stack((area / (GRID * GRID), fore_cx, fore_cy, plane[:, 0].mean(1), plane[:, -1].mean(1),
        plane[:, :, 0].mean(1), plane[:, :, -1].mean(1)), 1)

    field = unit.view(batch, GRID, GRID, -1)
    horizontal = (field[:, :, :-1] * field[:, :, 1:]).sum(-1)
    vertical = (field[:, :-1] * field[:, 1:]).sum(-1)
    row_edge, col_edge = (1 - vertical).mean(2), (1 - horizontal).mean(1)
    boundary = torch.zeros((batch, GRID, GRID), device=tokens.device)
    boundary[:, :, :-1] += 1 - horizontal
    boundary[:, :, 1:] += 1 - horizontal
    boundary[:, :-1, :] += 1 - vertical
    boundary[:, 1:, :] += 1 - vertical
    boundary = boundary / constants["counts"]
    border = constants["border"]
    structure = torch.stack(((horizontal.mean((1, 2)) + vertical.mean((1, 2))) / 2,
        (row_edge.argmax(1).float() + 1) / GRID, row_edge.max(1).values / row_edge.mean(1).clamp_min(1e-6),
        (col_edge.argmax(1).float() + 1) / GRID, col_edge.max(1).values / col_edge.mean(1).clamp_min(1e-6),
        boundary[:, border].mean(1), boundary[:, ~border].mean(1),
        (field * field.flip(2)).sum(-1)[:, :, :GRID // 2].mean((1, 2)),
        (field * field.flip(1)).sum(-1)[:, :GRID // 2].mean((1, 2))), 1)

    gpu_part = torch.cat((salient, spectrum, partition), 1).double().cpu().numpy()
    components = component_stats(plane.bool().cpu().numpy())
    result = np.concatenate((gpu_part, components, structure.double().cpu().numpy()), axis=1)
    if result.shape[1] != len(NAMES) or not np.isfinite(result).all():
        raise ValueError("布局描述维数或有限性错误")
    return result


def extract(cache: Path, fingerprints: list[str], indices: np.ndarray) -> dict:
    """按缓存行序分块读取指定照片的 patch 网格，返回布局 / 位置置换布局 / 平均 token 三组逐图表示。"""
    cached = json.loads((cache / "fingerprints.json").read_text(encoding="utf-8"))
    row_of = dict((fingerprint, row) for row, fingerprint in enumerate(cached))
    array = np.load(cache / "patches.f16.npy", mmap_mode="r")
    rows = np.array([row_of[fingerprints[index]] for index in indices])
    order = np.argsort(rows)
    constants = grid_constants("cuda")
    layout, shuffle = np.empty((len(indices), len(NAMES))), np.empty((len(indices), len(NAMES)))
    content = np.empty((len(indices), 384))
    for start in range(0, len(indices), BATCH):
        pick = order[start:start + BATCH]
        block = torch.from_numpy(np.ascontiguousarray(array[rows[pick]])).cuda()
        layout[pick] = describe(block, constants)
        permutation = np.stack([np.random.default_rng([SHUFFLE_SEED, int(indices[slot])]).permutation(GRID * GRID)
            for slot in pick])
        index = torch.from_numpy(permutation).cuda()[..., None].expand_as(block)
        shuffle[pick] = describe(torch.gather(block, 1, index), constants)
        unit = block.float() / block.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        content[pick] = unit.mean(1).double().cpu().numpy()
        if start // BATCH % 20 == 0:
            print(f"  described {min(start + BATCH, len(indices))}/{len(indices)}", flush=True)
    return {"indices": indices, "layout": layout, "shuffle": shuffle, "content": content}


def pair_weights(high: np.ndarray, low: np.ndarray, event_of: np.ndarray) -> np.ndarray:
    """与 scene_comparator.PairSampler 同口径的对权重：事件等权 → 事件内照片等权 → 平分到该照片的各伙伴。"""
    degree = Counter(high.tolist()) + Counter(low.tolist())
    photos = defaultdict(set)
    for photo in degree:
        photos[event_of[photo]].add(photo)
    share = lambda photo: 1 / (len(photos) * len(photos[event_of[photo]]) * degree[photo])
    weights = np.array([share(winner) + share(loser) for winner, loser in zip(high.tolist(), low.tolist())])
    return weights / weights.sum()


def fit_scorer(z: np.ndarray, high: np.ndarray, low: np.ndarray, weights: np.ndarray, penalty: float) -> np.ndarray:
    """无截距加权成对 logistic：P(高档胜) = σ(β·(z_高 − z_低))，返回逐图线性打分系数 β。"""
    difference = z[high] - z[low]

    def objective(beta):
        """返回加权负对数似然 + L2 及梯度。"""
        margin = difference @ beta
        loss = float(weights @ np.logaddexp(0, -margin) + penalty * beta @ beta / 2)
        return loss, -difference.T @ (weights * expit(-margin)) + penalty * beta

    result = minimize(objective, np.zeros(z.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    return result.x


def oriented(left: np.ndarray, right: np.ndarray, tiers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """跨档边按档位定向为 (高档, 低档)。"""
    cross = tiers[left] != tiers[right]
    upper = tiers[left] > tiers[right]
    return np.where(upper, left, right)[cross], np.where(upper, right, left)[cross]


def macro(score: np.ndarray, left: np.ndarray, right: np.ndarray, tiers: np.ndarray, event_of: np.ndarray) -> float:
    """逐图分数在给定边上的跨档对事件宏平均准确率。"""
    return sc.pair_accuracy(score[left] - score[right], left, right, tiers, event_of)["event_macro"]


def standardizer(values: np.ndarray, fitting: np.ndarray, components: int | None) -> callable:
    """在拟合照片上估计标准化（content 臂另做 PCA 到给定维数），返回作用于全体照片的变换。"""
    center = values[fitting].mean(axis=0)
    basis = None
    if components is not None:
        _, _, vt = np.linalg.svd(values[fitting] - center, full_matrices=False)
        basis = vt[:components].T
    project = lambda data: (data - center) if basis is None else (data - center) @ basis
    scale = project(values[fitting]).std(axis=0)
    scale[scale < 1e-12] = 1.0
    return lambda data: project(data) / scale


def halves(events: list[str]) -> list[list[str]]:
    """拟合事件按哈希分两半（只用于内层选正则与融合权重）。"""
    tagged = [int(hashlib.sha256(("inner:" + event).encode()).hexdigest()[:8], 16) % 2 for event in events]
    return [[event for event, tag in zip(events, tagged) if tag == part] for part in (0, 1)]


def fit_arm(values, fitting_events, left, right, tiers, event_of, components):
    """单折单臂：内层两半互评选正则、得内层交叉分；再用全部拟合事件重拟合，返回逐图分函数与记录。"""
    transform = standardizer(values, np.isin(event_of, fitting_events), components)
    z = transform(values)
    fit_edges = np.isin(event_of[left], fitting_events)
    inner = halves(fitting_events)
    grid, inner_scores = [], dict()
    for penalty in PENALTIES:
        score = np.full(len(values), np.nan)
        for part in (0, 1):
            train_edges = fit_edges & np.isin(event_of[left], inner[1 - part])
            high, low = oriented(left[train_edges], right[train_edges], tiers)
            beta = fit_scorer(z, high, low, pair_weights(high, low, event_of), penalty)
            members = np.isin(event_of, inner[part])
            score[members] = z[members] @ beta
        accuracy = macro(score, left[fit_edges], right[fit_edges], tiers, event_of)
        grid.append({"penalty": penalty, "inner_accuracy": accuracy})
        inner_scores[penalty] = score
    best = max(grid, key=lambda row: (row["inner_accuracy"], row["penalty"]))["penalty"]
    high, low = oriented(left[fit_edges], right[fit_edges], tiers)
    beta = fit_scorer(z, high, low, pair_weights(high, low, event_of), best)
    final = z @ beta
    return {"score": final, "inner_score": inner_scores[best], "beta": beta, "penalty": best, "grid": grid,
        "fitting_accuracy": macro(final, left[fit_edges], right[fit_edges], tiers, event_of)}


def choose_alpha(single: np.ndarray, layout_inner: np.ndarray, left, right, tiers, event_of) -> tuple[float, list]:
    """在拟合事件边上用内层交叉布局分选融合权重 α（同分取较小 α）。"""
    single_scale = np.std(single[left] - single[right])
    layout_scale = np.std(layout_inner[left] - layout_inner[right])
    grid = []
    for alpha in ALPHAS:
        fused = single / single_scale + alpha * layout_inner / layout_scale
        grid.append({"alpha": alpha, "accuracy": macro(fused, left, right, tiers, event_of)})
    return max(grid, key=lambda row: (row["accuracy"], -row["alpha"]))["alpha"], grid


def arm_entry(score: np.ndarray, left, right, tiers, event_of, cosine, boundary_scores=None) -> dict:
    """一组逐图分数的跨档对准确率：总体、≥1/≥2/≥3 边界与余弦分带。"""
    entry = {"overall": sc.pair_accuracy(score[left] - score[right], left, right, tiers, event_of)}
    for boundary in (1, 2, 3):
        boundary_score = score if boundary_scores is None else boundary_scores[boundary]
        entry[f"boundary{boundary}"] = sc.pair_accuracy(boundary_score[left] - boundary_score[right], left, right,
            tiers, event_of, boundary)
    entry["bands"] = dict()
    for low, high in COSINE_BANDS:
        mask = (cosine >= low) & (cosine < high)
        result = sc.pair_accuracy(score[left[mask]] - score[right[mask]], left[mask], right[mask], tiers, event_of)
        entry["bands"][f"{low:.2f}-{min(high, 1.0):.2f}"] = {"pairs": result["pairs"], "event_macro": result["event_macro"],
            "pooled": result["pooled"]}
    return entry


def golden_entry(rows: list[dict], score: np.ndarray) -> dict:
    """golden 盲评对上逐图分数的准确率（合并与事件宏平均，相等记 0.5）。"""
    by_event = defaultdict(list)
    for row in rows:
        margin = score[row["winner"]] - score[row["loser"]]
        by_event[row["event"]].append(1.0 if margin > 0 else (0.5 if margin == 0 else 0.0))
    values = [value for items in by_event.values() for value in items]
    return {"pairs": len(values), "pooled": float(np.mean(values)),
        "event_macro": float(np.mean([np.mean(items) for items in by_event.values()])),
        "by_event": dict((event, float(np.mean(items))) for event, items in sorted(by_event.items()))}


def reliability(values: np.ndarray, left: np.ndarray, right: np.ndarray) -> dict:
    """近重对（cos≥0.98）上逐描述的左右 Pearson 相关，检验描述是否稳定而非噪声。"""
    result = dict()
    for column, name in enumerate(NAMES):
        a, b = values[left, column], values[right, column]
        result[name] = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None
    finite = [value for value in result.values() if value is not None]
    return {"median": float(np.median(finite)), "by_descriptor": result}


def load_descriptors(path: Path, total: int) -> dict:
    """读回已提取的逐图表示，铺回全体照片下标（未覆盖处为 NaN）。"""
    with np.load(path) as cached:
        indices = cached["indices"]
        result = {"indices": indices, "patch_sha256": str(cached["patch_sha256"])}
        for name in ARMS:
            full = np.full((total, cached[name].shape[1]), np.nan)
            full[indices] = cached[name]
            result[name] = full
    return result


def main() -> int:
    """按登记顺序执行 Stage A → B →（可选）val，闸门自动判定，不据 val/test 调参。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patches", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("D:/PhotoDB/dataset"))
    parser.add_argument("--descriptors", type=Path, help="复用既有 descriptors.npz，跳过 train 提取")
    parser.add_argument("--control", choices=("none", "random", "truth"), default="none")
    parser.add_argument("--open-val", action="store_true")
    cli = parser.parse_args()
    threadpool_limits(2)
    inputs = argparse.Namespace(photos=ROOT / "audit/out/m3_pairs_v2_20260924b/photos.csv",
        split_json=ROOT / "data/split-v2-20260924.json", features=ROOT / "train/out/recovery_work/corrected_features.npz",
        local=ROOT / "train/out/recovery_work/rescored_lora_v2/scores.csv",
        global_scores=ROOT / "train/out/recovery_work/corrected_rank/clip_scores.csv")
    formal = ROOT / "train/out/rating_baseline_20261014_balanced"
    metadata, fingerprints, features, _ = rb.load_inputs(inputs)
    identity = json.loads((cli.patches / "identity.json").read_text(encoding="utf-8"))
    complete = json.loads((cli.patches / "complete.json").read_text(encoding="utf-8"))
    if identity["decode_version"] != "display-plane-v3" or identity["features_sha256"] != digest(inputs.features):
        raise ValueError("patch 缓存身份与修复特征不一致")
    cli.out.mkdir(parents=True, exist_ok=False)
    hashed = [inputs.photos, inputs.split_json, inputs.features, Path(__file__), ROOT / "train/scene_comparator.py",
        ROOT / "train/rating_baseline.py", ROOT / "audit/evaluation_protocol.py", ROOT / "audit/golden_label_check.py",
        formal / "joint_scores.csv", formal / "joint_0.1_oof.json"]
    plan = {"registration": "EXECUTION-LOG 2026-11-12 内容无关布局描述读团内取舍", "control": cli.control,
        "descriptors_source": str(cli.descriptors) if cli.descriptors else "extracted (descriptors.npz)",
        "similar_cosine": sc.SIMILAR, "descriptors": list(NAMES), "penalties": PENALTIES, "alphas": ALPHAS,
        "gammas": sc.GAMMAS, "shuffle_seed": SHUFFLE_SEED, "gate_a": GATE_A, "gate_b": GATE_B,
        "patch_cache": str(cli.patches), "patch_sha256": complete["patch_sha256"],
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

    baseline = baseline_oof(features, tiers, event_of, train_events)
    baseline_metrics = sc.measure_rows(metadata, fingerprints, baseline, train_indices, train_events)
    formal_metrics = json.loads((formal / "joint_0.1_oof.json").read_text(encoding="utf-8"))["metrics"]
    for threshold in ("1", "2", "3"):
        if abs(baseline_metrics["ranking"][threshold]["ap_macro"] - formal_metrics["ranking"][threshold]["ap_macro"]) > 1e-6:
            raise ValueError("单图基线 OOF 未复现正式类别均衡臂")
    result["baseline_train_oof"] = sc.headline(baseline_metrics)
    expected = baseline @ np.arange(4.0)
    print(f"baseline reproduced {result['baseline_train_oof']}", flush=True)

    if cli.descriptors is None:
        if digest(cli.patches / "patches.f16.npy") != complete["patch_sha256"]:
            raise ValueError("patch 缓存内容哈希不符")
        extracted = extract(cli.patches, fingerprints, train_indices)
        np.savez(cli.out / "descriptors.npz", patch_sha256=complete["patch_sha256"], names=np.array(NAMES),
            **dict((name, extracted[name]) for name in ("indices", *ARMS)))
        stored = load_descriptors(cli.out / "descriptors.npz", len(fingerprints))
    else:
        stored = load_descriptors(cli.descriptors, len(fingerprints))
        if stored["patch_sha256"] != complete["patch_sha256"] or not np.isin(train_indices, stored["indices"]).all():
            raise ValueError("复用的描述与 patch 缓存身份或 train 覆盖不一致")
    if cli.control != "none":
        rng = np.random.default_rng(CONTROL_SEED)
        noise = rng.standard_normal(stored["layout"].shape)
        if cli.control == "truth":
            noise[:, 0] = tiers + rng.normal(0, 1.5, len(tiers))
        stored["layout"] = noise
        print(f"CONTROL {cli.control}: 布局描述已替换，输出不参与判定", flush=True)

    left, right, cosine = sc.similar_edges(sc.segments_of(train_indices, segment_key), dino)
    print(f"train edges {len(left)} cross {int((tiers[left] != tiers[right]).sum())}", flush=True)
    golden_rows = pair_rows(read_groups(cli.dataset), dict((fp, index) for index, fp in enumerate(fingerprints)),
        event_of, set(train_events))

    oof = dict((arm, np.full(len(fingerprints), np.nan)) for arm in ARMS)
    fused = dict((arm, np.full(len(fingerprints), np.nan)) for arm in ARMS)
    signal_fit = [dict() for _ in folds]
    records = []
    for fold, held in enumerate(folds):
        fitting_events = [event for event in train_events if event not in held]
        fit_edges = np.isin(event_of[left], fitting_events)
        held_photos = np.isin(event_of, held)
        for arm in ARMS:
            components = len(NAMES) if arm == "content" else None
            fitted = fit_arm(stored[arm], fitting_events, left, right, tiers, event_of, components)
            alpha, alpha_grid = choose_alpha(expected, fitted["inner_score"], left[fit_edges], right[fit_edges],
                tiers, event_of)
            single_scale = np.std(expected[left[fit_edges]] - expected[right[fit_edges]])
            layout_scale = np.std(fitted["score"][left[fit_edges]] - fitted["score"][right[fit_edges]])
            oof[arm][held_photos] = fitted["score"][held_photos]
            fused[arm][held_photos] = (expected / single_scale + alpha * fitted["score"] / layout_scale)[held_photos]
            fitting_photos = np.isin(event_of, fitting_events)
            signal_fit[fold][arm] = (fitted["inner_score"], fitted["score"], fitting_photos)
            records.append({"fold": fold, "arm": arm, "held_out_events": held, "penalty": fitted["penalty"],
                "penalty_grid": fitted["grid"], "alpha": alpha, "alpha_grid": alpha_grid,
                "fitting_accuracy": fitted["fitting_accuracy"],
                "beta": dict(zip(NAMES, fitted["beta"].tolist())) if arm != "content" else None})
            print(f"fold {fold} {arm}: penalty {fitted['penalty']} alpha {alpha} fit {fitted['fitting_accuracy']:.4f}",
                flush=True)
    for name in (*ARMS,):
        if np.isnan(oof[name][train_indices]).any():
            raise ValueError("留出照片未全部打分")

    high_boundary = dict((boundary, baseline[:, boundary:].sum(axis=1)) for boundary in (1, 2, 3))
    scores = {"single_image_joint": expected, **oof, **dict((f"fused_{arm}", fused[arm]) for arm in ARMS)}
    stage_a = {"records": records, "arms": dict(), "golden": dict()}
    for name, score in scores.items():
        stage_a["arms"][name] = arm_entry(score, left, right, tiers, event_of, cosine,
            high_boundary if name == "single_image_joint" else None)
        stage_a["golden"][name] = golden_entry(golden_rows, score)
    events_of = lambda name: stage_a["arms"][name]["overall"]["by_event"]
    stage_a["comparisons"] = {
        "layout_minus_shuffle": sc.bootstrap(events_of("layout"), events_of("shuffle")),
        "layout_minus_content": sc.bootstrap(events_of("layout"), events_of("content")),
        "layout_minus_single": sc.bootstrap(events_of("layout"), events_of("single_image_joint")),
        "fused_layout_minus_single": sc.bootstrap(events_of("fused_layout"), events_of("single_image_joint")),
        "fused_shuffle_minus_single": sc.bootstrap(events_of("fused_shuffle"), events_of("single_image_joint")),
        "fused_content_minus_single": sc.bootstrap(events_of("fused_content"), events_of("single_image_joint")),
        "golden_layout_minus_single": sc.bootstrap(stage_a["golden"]["layout"]["by_event"],
            stage_a["golden"]["single_image_joint"]["by_event"]),
        "golden_fused_layout_minus_single": sc.bootstrap(stage_a["golden"]["fused_layout"]["by_event"],
            stage_a["golden"]["single_image_joint"]["by_event"])}
    layout_accuracy = stage_a["arms"]["layout"]["overall"]["event_macro"]
    golden_accuracy = stage_a["golden"]["layout"]["event_macro"]
    versus_shuffle = stage_a["comparisons"]["layout_minus_shuffle"]["event_bootstrap95"]
    versus_single = stage_a["comparisons"]["fused_layout_minus_single"]["event_bootstrap95"]
    passed_a = (layout_accuracy >= GATE_A["layout"] and versus_shuffle[0] > 0 and versus_single[0] > 0
        and golden_accuracy >= GATE_A["golden"])
    killed_a = layout_accuracy < GATE_A["kill_below"] or versus_single[0] <= 0 <= versus_single[1]
    stage_a["gate"] = {"layout_event_macro": layout_accuracy, "golden_event_macro": golden_accuracy,
        "layout_minus_shuffle_95": versus_shuffle, "fused_minus_single_95": versus_single,
        "verdict": "pass" if passed_a else ("kill" if killed_a else "weak")}
    near = cosine >= 0.98
    stage_a["descriptor_reliability_near_duplicates"] = {"pairs": int(near.sum()),
        "layout": reliability(stored["layout"], left[near], right[near]),
        "shuffle": reliability(stored["shuffle"], left[near], right[near])}
    layout_betas = np.array([[row["beta"][name] for name in NAMES] for row in records if row["arm"] == "layout"])
    stage_a["layout_coefficients"] = sorted(({"descriptor": name, "mean": float(layout_betas[:, k].mean()),
        "same_sign_folds": int(max((layout_betas[:, k] > 0).sum(), (layout_betas[:, k] < 0).sum()))}
        for k, name in enumerate(NAMES)), key=lambda row: -abs(row["mean"]))
    result["stage_a"] = stage_a
    write_json(cli.out / "result.json", result)
    print(f"STAGE A gate {stage_a['gate']}", flush=True)
    for name in scores:
        entry = stage_a["arms"][name]
        print(f"  {name}: event-macro {entry['overall']['event_macro']:.4f} pooled {entry['overall']['pooled']:.4f} "
            f"b1 {entry['boundary1']['event_macro']:.4f} b3 {entry['boundary3']['event_macro']:.4f} "
            f"golden {stage_a['golden'][name]['event_macro']:.4f}", flush=True)
    for name, row in stage_a["comparisons"].items():
        print(f"  {name}: {row['mean']:+.4f} {np.round(row['event_bootstrap95'], 4).tolist()} "
            f"({row['positive_events']}/{row['events']})", flush=True)

    tilted = baseline.copy()
    stage_b = {"folds": []}
    for fold, held in enumerate(folds):
        fitting_events = [event for event in train_events if event not in held]
        inner_score, outer_score, fitting_photos = signal_fit[fold]["layout"]
        inner_z = (inner_score - inner_score[fitting_photos].mean()) / inner_score[fitting_photos].std()
        outer_z = (outer_score - outer_score[fitting_photos].mean()) / outer_score[fitting_photos].std()
        gamma, grid = sc.choose_gamma(metadata, fingerprints, baseline, inner_z, event_of, fitting_events)
        held_photos = np.flatnonzero(np.isin(event_of, held))
        tilted[held_photos] = sc.tilt(baseline[held_photos], outer_z[held_photos], gamma)
        stage_b["folds"].append({"held_out_events": held, "gamma": gamma, "grid": grid})
    tilted_metrics = sc.measure_rows(metadata, fingerprints, tilted, train_indices, train_events)
    stage_b["fused_train_oof"] = sc.headline(tilted_metrics)
    stage_b["baseline_train_oof"] = result["baseline_train_oof"]
    stage_b["differences"] = sc.compare_metrics(tilted_metrics, baseline_metrics)
    differences = stage_b["differences"]
    passed_b = (differences["ap3"]["mean"] >= GATE_B["ap3"] and differences["ap3"]["event_bootstrap95"][0] > 0
        and differences["ap1"]["mean"] >= 0 and differences["balanced_accuracy"]["mean"] >= 0)
    stage_b["gate"] = {"passed": passed_b, "counts_as_evidence": passed_a}
    result["stage_b"] = stage_b
    write_json(cli.out / "result.json", result)
    print(f"STAGE B fused {stage_b['fused_train_oof']}\n  differences "
        f"{dict((k, (round(v['mean'], 4), np.round(v['event_bootstrap95'], 4).tolist())) for k, v in differences.items())}"
        f"\n  gate {passed_b} (counts only if A passed: {passed_a})", flush=True)
    if cli.control != "none" or not (passed_a and passed_b and cli.open_val):
        print("未满足 val 条件（对照运行 / 闸门未全过 / 未授权），停止", flush=True)
        return 0

    full_events = train_events
    final = fit_arm(stored["layout"], full_events, left, right, tiers, event_of, None)
    val_indices = np.flatnonzero(split_of == "val")
    val_events = sorted(set(event_of[val_indices].tolist()))
    val_layout = np.full((len(fingerprints), len(NAMES)), np.nan)
    val_layout[val_indices] = extract(cli.patches, fingerprints, val_indices)["layout"]
    transform = standardizer(stored["layout"], np.isin(event_of, full_events), None)
    val_score = np.full(len(fingerprints), np.nan)
    val_score[val_indices] = transform(val_layout[val_indices]) @ final["beta"]
    train_score = final["score"]
    scale_center, scale = train_score[train_indices].mean(), train_score[train_indices].std()
    oof_z = np.full(len(fingerprints), np.nan)
    for fold, held in enumerate(folds):
        _, outer_score, fitting_photos = signal_fit[fold]["layout"]
        held_photos = np.isin(event_of, held)
        oof_z[held_photos] = ((outer_score - outer_score[fitting_photos].mean()) / outer_score[fitting_photos].std())[held_photos]
    val_gamma, val_grid = sc.choose_gamma(metadata, fingerprints, baseline, oof_z, event_of, train_events)
    formal_scores = dict()
    with (formal / "joint_scores.csv").open(encoding="utf-8") as stream:
        header = stream.readline().strip().split(",")
        for line in stream:
            values = line.strip().split(",")
            formal_scores[values[0]] = np.array([float(values[header.index(name)]) for name in ("p0", "p1", "p2", "p_ge3")])
    val_baseline = np.full((len(fingerprints), 4), np.nan)
    for index in val_indices:
        val_baseline[index] = formal_scores[fingerprints[index]]
    val_tilted = val_baseline.copy()
    val_tilted[val_indices] = sc.tilt(val_baseline[val_indices], (val_score[val_indices] - scale_center) / scale, val_gamma)
    val_baseline_metrics = sc.measure_rows(metadata, fingerprints, val_baseline, val_indices, val_events)
    val_tilted_metrics = sc.measure_rows(metadata, fingerprints, val_tilted, val_indices, val_events)
    val_left, val_right, _ = sc.similar_edges(sc.segments_of(val_indices, segment_key), dino)
    val_expected = val_baseline @ np.arange(4.0)
    result["val"] = {"penalty": final["penalty"], "gamma": val_gamma, "grid_on_train_oof": val_grid,
        "baseline": sc.headline(val_baseline_metrics), "fused": sc.headline(val_tilted_metrics),
        "differences": sc.compare_metrics(val_tilted_metrics, val_baseline_metrics),
        "pair_accuracy": {"layout": sc.pair_accuracy(val_score[val_left] - val_score[val_right], val_left, val_right,
            tiers, event_of), "single_image_joint": sc.pair_accuracy(val_expected[val_left] - val_expected[val_right],
            val_left, val_right, tiers, event_of)}}
    write_json(cli.out / "result.json", result)
    print(f"VAL {result['val']['baseline']}\n -> {result['val']['fused']}\n  diff {result['val']['differences']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
