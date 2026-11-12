"""逐图 PCA 关系图 + 小卷积读法：把单图 patch 关系结构整张交给学习式读法，检验同场景取舍能否跨事件迁移。

实验登记见 EXECUTION-LOG 2026-11-12（第二试）。输入只由单图 Gram 关系与位置决定（显著度、逐图前三主成分投影、
相邻边界强度 + 坐标通道），对特征空间旋转不变；读法为约 1.3 万参数的逐图打分卷积网，成对 logistic、三种子、
固定步数。对照位置置换关系图、单图 joint 头及融合；闸门、对、折、golden 与 layout_probe 完全相同。
--open-val 只在两道闸均通过后执行一次，不打开 test。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "audit"))
import layout_probe as lp
import rating_baseline as rb
import scene_comparator as sc
from evaluation_protocol import digest, write_json
from golden_label_check import pair_rows, read_groups
from within_scene_audit import baseline_oof

GRID = lp.GRID
CHANNELS = 5
ARMS = ("map", "map_shuffle")
SEEDS = (0, 1, 2)
STEPS, BATCH, WIDTH, LEARNING_RATE, DECAY, WARMUP = 1500, 256, 24, 1e-3, 1e-2, 100


@torch.no_grad()
def relation_maps(tokens: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """一批 [B,1024,384] patch 网格 → [B,5,32,32]：显著度、前三主成分投影（与显著度同号）、相邻边界强度。"""
    batch = tokens.shape[0]
    unit = tokens.float()
    unit = unit / unit.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    mean = unit.mean(dim=1)
    rarity = 1 - torch.einsum("btd,bd->bt", unit, mean)
    centered = unit - mean[:, None, :]
    values, vectors = torch.linalg.eigh(centered.transpose(1, 2) @ centered / (GRID * GRID))
    total = values.clamp_min(0).sum(-1).clamp_min(1e-12)
    projection = torch.einsum("btd,bdk->btk", centered, vectors[:, :, -3:].flip(-1)) / total.sqrt()[:, None, None]
    rarity_centered = rarity - rarity.mean(1, keepdim=True)
    sign = torch.sign((projection * rarity_centered[..., None]).sum(1))
    sign[sign == 0] = 1
    projection = projection * sign[:, None, :]
    field = unit.view(batch, GRID, GRID, -1)
    horizontal = 1 - (field[:, :, :-1] * field[:, :, 1:]).sum(-1)
    vertical = 1 - (field[:, :-1] * field[:, 1:]).sum(-1)
    boundary = torch.zeros((batch, GRID, GRID), device=tokens.device)
    boundary[:, :, :-1] += horizontal
    boundary[:, :, 1:] += horizontal
    boundary[:, :-1, :] += vertical
    boundary[:, 1:, :] += vertical
    boundary = boundary / counts
    maps = torch.cat((rarity[:, None], projection.transpose(1, 2)), 1).view(batch, 4, GRID, GRID)
    return torch.cat((maps, boundary[:, None]), 1)


def extract_maps(cache: Path, fingerprints: list[str], indices: np.ndarray) -> dict:
    """按缓存行序读取指定照片，返回真实与位置置换两组关系图（float16，[N,5,32,32]）。"""
    cached = json.loads((cache / "fingerprints.json").read_text(encoding="utf-8"))
    row_of = dict((fingerprint, row) for row, fingerprint in enumerate(cached))
    array = np.load(cache / "patches.f16.npy", mmap_mode="r")
    rows = np.array([row_of[fingerprints[index]] for index in indices])
    order = np.argsort(rows)
    counts = lp.grid_constants("cuda")["counts"]
    result = dict((arm, np.empty((len(indices), CHANNELS, GRID, GRID), dtype=np.float16)) for arm in ARMS)
    for start in range(0, len(indices), lp.BATCH):
        pick = order[start:start + lp.BATCH]
        block = torch.from_numpy(np.ascontiguousarray(array[rows[pick]])).cuda()
        result["map"][pick] = relation_maps(block, counts).half().cpu().numpy()
        permutation = np.stack([np.random.default_rng([lp.SHUFFLE_SEED, int(indices[slot])]).permutation(GRID * GRID)
            for slot in pick])
        index = torch.from_numpy(permutation).cuda()[..., None].expand_as(block)
        result["map_shuffle"][pick] = relation_maps(torch.gather(block, 1, index), counts).half().cpu().numpy()
        if start // lp.BATCH % 20 == 0:
            print(f"  mapped {min(start + lp.BATCH, len(indices))}/{len(indices)}", flush=True)
    return result


class MapScorer(nn.Module):
    """逐图打分小卷积网：关系图 + 坐标 → 三层 3×3 卷积 → 1×1 降维 → 保留 8×8 位置展平 → 线性分数。"""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(CHANNELS + 2, WIDTH, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.GELU(),
            nn.Conv2d(WIDTH, 8, 1), nn.GELU(), nn.Flatten(), nn.Dropout(0.1))
        self.head = nn.Linear(8 * 8 * 8, 1)
        grid = (torch.arange(GRID).float() + 0.5) / GRID - 0.5
        self.register_buffer("coordinates", torch.stack(torch.meshgrid(grid, grid, indexing="ij"))[None])

    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        """返回每张图的标量分数。"""
        inputs = torch.cat((maps, self.coordinates.expand(len(maps), -1, -1, -1)), 1)
        return self.head(self.body(inputs)).squeeze(-1)


def train_scorer(maps: torch.Tensor, sampler: sc.PairSampler, seed: int) -> MapScorer:
    """固定步数训练一个逐图打分网（f(A,B)=s(A)−s(B) 成对 logistic，无早停）。"""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MapScorer().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=DECAY)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / WARMUP))
    model.train()
    for _ in range(STEPS):
        high, low = sampler.draw(rng, BATCH)
        swap = rng.random(BATCH) < 0.5
        left, right = np.where(swap, low, high), np.where(swap, high, low)
        target = torch.as_tensor((~swap).astype(np.float32), device="cuda")
        logit = model(maps[left]) - model(maps[right])
        loss = F.binary_cross_entropy_with_logits(logit, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
    if not torch.isfinite(loss):
        raise ValueError("训练损失非有限")
    return model


@torch.no_grad()
def score_photos(models: list[MapScorer], maps: torch.Tensor, photos: np.ndarray, total: int) -> np.ndarray:
    """多种子平均逐图分数，未打分处为 NaN。"""
    result = np.full(total, np.nan)
    values = np.zeros(len(photos))
    for model in models:
        model.eval()
        for start in range(0, len(photos), 1024):
            values[start:start + 1024] += model(maps[photos[start:start + 1024]]).double().cpu().numpy()
    result[photos] = values / len(models)
    return result


def fit_arm(maps, photos, fitting_events, left, right, tiers, event_of, seed_base):
    """单折单臂：内层两半互评得内层交叉分；再用全部拟合事件训练三种子，返回逐图分与拟合准确率。"""
    fit_edges = np.isin(event_of[left], fitting_events)
    inner_score = np.full(len(event_of), np.nan)
    for part, half in enumerate(lp.halves(fitting_events)):
        other = [event for event in fitting_events if event not in half]
        train_edges = fit_edges & np.isin(event_of[left], other)
        sampler = sc.PairSampler(left[train_edges], right[train_edges], tiers, event_of)
        models = [train_scorer(maps, sampler, seed_base + 10 * (part + 1) + seed) for seed in SEEDS]
        members = np.flatnonzero(np.isin(event_of, half))
        inner_score[members] = score_photos(models, maps, members, len(event_of))[members]
    sampler = sc.PairSampler(left[fit_edges], right[fit_edges], tiers, event_of)
    models = [train_scorer(maps, sampler, seed_base + seed) for seed in SEEDS]
    score = score_photos(models, maps, photos, len(event_of))
    return {"score": score, "inner_score": inner_score, "models": models,
        "fitting_accuracy": lp.macro(score, left[fit_edges], right[fit_edges], tiers, event_of)}


def main() -> int:
    """按登记顺序执行 Stage A → B →（可选）val，闸门沿用 layout_probe，不据 val/test 调参。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patches", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("D:/PhotoDB/dataset"))
    parser.add_argument("--maps", type=Path, help="复用既有 maps.npz，跳过 train 提取")
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
    hashed = [inputs.photos, inputs.split_json, inputs.features, Path(__file__), ROOT / "train/layout_probe.py",
        ROOT / "train/scene_comparator.py", ROOT / "train/rating_baseline.py", ROOT / "audit/evaluation_protocol.py",
        ROOT / "audit/golden_label_check.py", formal / "joint_scores.csv", formal / "joint_0.1_oof.json"]
    plan = {"registration": "EXECUTION-LOG 2026-11-12 逐图 PCA 关系图 + 小卷积读法", "maps_source": str(cli.maps or "extracted"),
        "channels": ["rarity", "pc1", "pc2", "pc3", "boundary", "x", "y"], "seeds": SEEDS, "steps": STEPS, "batch": BATCH,
        "width": WIDTH, "learning_rate": LEARNING_RATE, "weight_decay": DECAY, "warmup": WARMUP,
        "parameters": sum(p.numel() for p in MapScorer().parameters()), "alphas": lp.ALPHAS, "gammas": sc.GAMMAS,
        "gate_a": lp.GATE_A, "gate_b": lp.GATE_B, "patch_cache": str(cli.patches), "patch_sha256": complete["patch_sha256"],
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
    print(f"baseline reproduced ap3 {result['baseline_train_oof']['ap3']:.6f}", flush=True)

    if cli.maps is None:
        if digest(cli.patches / "patches.f16.npy") != complete["patch_sha256"]:
            raise ValueError("patch 缓存内容哈希不符")
        extracted = extract_maps(cli.patches, fingerprints, train_indices)
        np.savez(cli.out / "maps.npz", indices=train_indices, patch_sha256=complete["patch_sha256"], **extracted)
        maps_path = cli.out / "maps.npz"
    else:
        maps_path = cli.maps
    with np.load(maps_path) as cached:
        if str(cached["patch_sha256"]) != complete["patch_sha256"] or not np.array_equal(cached["indices"], train_indices):
            raise ValueError("关系图与 patch 缓存身份或 train 覆盖不一致")
        raw = dict((arm, cached[arm].astype(np.float32)) for arm in ARMS)
    stores = dict()
    for arm in ARMS:
        values = raw[arm]
        center = values.mean(axis=(0, 2, 3), keepdims=True)
        scale = values.std(axis=(0, 2, 3), keepdims=True)
        full = torch.zeros((len(fingerprints), CHANNELS, GRID, GRID), device="cuda")
        full[torch.from_numpy(train_indices).cuda()] = torch.from_numpy((values - center) / scale).cuda()
        stores[arm] = full
    del raw

    left, right, cosine = sc.similar_edges(sc.segments_of(train_indices, segment_key), dino)
    golden_rows = pair_rows(read_groups(cli.dataset), dict((fp, index) for index, fp in enumerate(fingerprints)),
        event_of, set(train_events))
    print(f"train edges {len(left)} cross {int((tiers[left] != tiers[right]).sum())} golden {len(golden_rows)}", flush=True)

    oof = dict((arm, np.full(len(fingerprints), np.nan)) for arm in ARMS)
    fused = dict((arm, np.full(len(fingerprints), np.nan)) for arm in ARMS)
    signal_fit = [dict() for _ in folds]
    records = []
    for fold, held in enumerate(folds):
        fitting_events = [event for event in train_events if event not in held]
        fit_edges = np.isin(event_of[left], fitting_events)
        held_photos = np.isin(event_of, held)
        fitting_photos = np.isin(event_of, fitting_events)
        for arm in ARMS:
            fitted = fit_arm(stores[arm], train_indices, fitting_events, left, right, tiers, event_of, 1000 * fold)
            alpha, alpha_grid = lp.choose_alpha(expected, fitted["inner_score"], left[fit_edges], right[fit_edges],
                tiers, event_of)
            single_scale = np.std(expected[left[fit_edges]] - expected[right[fit_edges]])
            map_scale = np.std(fitted["score"][left[fit_edges]] - fitted["score"][right[fit_edges]])
            oof[arm][held_photos] = fitted["score"][held_photos]
            fused[arm][held_photos] = (expected / single_scale + alpha * fitted["score"] / map_scale)[held_photos]
            signal_fit[fold][arm] = (fitted["inner_score"], fitted["score"], fitting_photos)
            records.append({"fold": fold, "arm": arm, "held_out_events": held, "alpha": alpha, "alpha_grid": alpha_grid,
                "fitting_accuracy": fitted["fitting_accuracy"],
                "inner_accuracy": lp.macro(fitted["inner_score"], left[fit_edges], right[fit_edges], tiers, event_of)})
            print(f"fold {fold} {arm}: fit {fitted['fitting_accuracy']:.4f} inner {records[-1]['inner_accuracy']:.4f} "
                f"alpha {alpha}", flush=True)
            del fitted
    for arm in ARMS:
        if np.isnan(oof[arm][train_indices]).any():
            raise ValueError("留出照片未全部打分")

    high_boundary = dict((boundary, baseline[:, boundary:].sum(axis=1)) for boundary in (1, 2, 3))
    scores = {"single_image_joint": expected, **oof, **dict((f"fused_{arm}", fused[arm]) for arm in ARMS)}
    stage_a = {"records": records, "arms": dict(), "golden": dict()}
    for name, score in scores.items():
        stage_a["arms"][name] = lp.arm_entry(score, left, right, tiers, event_of, cosine,
            high_boundary if name == "single_image_joint" else None)
        stage_a["golden"][name] = lp.golden_entry(golden_rows, score)
    events_of = lambda name: stage_a["arms"][name]["overall"]["by_event"]
    stage_a["comparisons"] = {
        "map_minus_shuffle": sc.bootstrap(events_of("map"), events_of("map_shuffle")),
        "map_minus_single": sc.bootstrap(events_of("map"), events_of("single_image_joint")),
        "fused_map_minus_single": sc.bootstrap(events_of("fused_map"), events_of("single_image_joint")),
        "fused_shuffle_minus_single": sc.bootstrap(events_of("fused_map_shuffle"), events_of("single_image_joint")),
        "golden_map_minus_single": sc.bootstrap(stage_a["golden"]["map"]["by_event"],
            stage_a["golden"]["single_image_joint"]["by_event"]),
        "golden_fused_map_minus_single": sc.bootstrap(stage_a["golden"]["fused_map"]["by_event"],
            stage_a["golden"]["single_image_joint"]["by_event"])}
    accuracy = stage_a["arms"]["map"]["overall"]["event_macro"]
    golden_accuracy = stage_a["golden"]["map"]["event_macro"]
    versus_shuffle = stage_a["comparisons"]["map_minus_shuffle"]["event_bootstrap95"]
    versus_single = stage_a["comparisons"]["fused_map_minus_single"]["event_bootstrap95"]
    passed_a = (accuracy >= lp.GATE_A["layout"] and versus_shuffle[0] > 0 and versus_single[0] > 0
        and golden_accuracy >= lp.GATE_A["golden"])
    killed_a = accuracy < lp.GATE_A["kill_below"] or versus_single[0] <= 0 <= versus_single[1]
    stage_a["gate"] = {"map_event_macro": accuracy, "golden_event_macro": golden_accuracy,
        "map_minus_shuffle_95": versus_shuffle, "fused_minus_single_95": versus_single,
        "verdict": "pass" if passed_a else ("kill" if killed_a else "weak")}
    result["stage_a"] = stage_a
    write_json(cli.out / "result.json", result)
    print(f"STAGE A gate {stage_a['gate']}", flush=True)
    for name in scores:
        entry = stage_a["arms"][name]
        print(f"  {name}: event-macro {entry['overall']['event_macro']:.4f} pooled {entry['overall']['pooled']:.4f} "
            f"bands {dict((k, round(v['event_macro'], 3)) for k, v in entry['bands'].items())} "
            f"golden {stage_a['golden'][name]['event_macro']:.4f}", flush=True)
    for name, row in stage_a["comparisons"].items():
        print(f"  {name}: {row['mean']:+.4f} {np.round(row['event_bootstrap95'], 4).tolist()} "
            f"({row['positive_events']}/{row['events']})", flush=True)

    tilted = baseline.copy()
    stage_b = {"folds": []}
    for fold, held in enumerate(folds):
        fitting_events = [event for event in train_events if event not in held]
        inner_score, outer_score, fitting_photos = signal_fit[fold]["map"]
        inner_z = (inner_score - inner_score[fitting_photos].mean()) / inner_score[fitting_photos].std()
        outer_z = (outer_score - outer_score[fitting_photos].mean()) / outer_score[fitting_photos].std()
        gamma, grid = sc.choose_gamma(metadata, fingerprints, baseline, inner_z, event_of, fitting_events)
        held_photos = np.flatnonzero(np.isin(event_of, held))
        tilted[held_photos] = sc.tilt(baseline[held_photos], outer_z[held_photos], gamma)
        stage_b["folds"].append({"held_out_events": held, "gamma": gamma, "grid": grid})
    tilted_metrics = sc.measure_rows(metadata, fingerprints, tilted, train_indices, train_events)
    stage_b["fused_train_oof"] = sc.headline(tilted_metrics)
    stage_b["differences"] = sc.compare_metrics(tilted_metrics, baseline_metrics)
    differences = stage_b["differences"]
    passed_b = (differences["ap3"]["mean"] >= lp.GATE_B["ap3"] and differences["ap3"]["event_bootstrap95"][0] > 0
        and differences["ap1"]["mean"] >= 0 and differences["balanced_accuracy"]["mean"] >= 0)
    stage_b["gate"] = {"passed": passed_b, "counts_as_evidence": passed_a}
    result["stage_b"] = stage_b
    write_json(cli.out / "result.json", result)
    print(f"STAGE B ap3 {stage_b['fused_train_oof']['ap3']:.4f} differences "
        f"{dict((k, (round(v['mean'], 4), np.round(v['event_bootstrap95'], 4).tolist())) for k, v in differences.items())}"
        f"\n  gate {passed_b} (counts only if A passed: {passed_a})", flush=True)
    if not (passed_a and passed_b and cli.open_val):
        print("未满足 val 条件（闸门未全过 / 未授权），停止", flush=True)
        return 0

    sampler = sc.PairSampler(left, right, tiers, event_of)
    models = [train_scorer(stores["map"], sampler, 9000 + seed) for seed in SEEDS]
    train_score = score_photos(models, stores["map"], train_indices, len(fingerprints))
    val_indices = np.flatnonzero(split_of == "val")
    val_events = sorted(set(event_of[val_indices].tolist()))
    with np.load(maps_path) as cached:
        reference = cached["map"].astype(np.float32)
    center, scale = reference.mean(axis=(0, 2, 3), keepdims=True), reference.std(axis=(0, 2, 3), keepdims=True)
    val_maps = (extract_maps(cli.patches, fingerprints, val_indices)["map"].astype(np.float32) - center) / scale
    stores["map"][torch.from_numpy(val_indices).cuda()] = torch.from_numpy(val_maps).cuda()
    val_score = score_photos(models, stores["map"], val_indices, len(fingerprints))
    oof_z = np.full(len(fingerprints), np.nan)
    for fold, held in enumerate(folds):
        _, outer_score, fitting_photos = signal_fit[fold]["map"]
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
    val_z = (val_score - train_score[train_indices].mean()) / train_score[train_indices].std()
    val_tilted = val_baseline.copy()
    val_tilted[val_indices] = sc.tilt(val_baseline[val_indices], val_z[val_indices], val_gamma)
    val_baseline_metrics = sc.measure_rows(metadata, fingerprints, val_baseline, val_indices, val_events)
    val_tilted_metrics = sc.measure_rows(metadata, fingerprints, val_tilted, val_indices, val_events)
    val_left, val_right, _ = sc.similar_edges(sc.segments_of(val_indices, segment_key), dino)
    val_expected = val_baseline @ np.arange(4.0)
    result["val"] = {"gamma": val_gamma, "grid_on_train_oof": val_grid, "baseline": sc.headline(val_baseline_metrics),
        "fused": sc.headline(val_tilted_metrics), "differences": sc.compare_metrics(val_tilted_metrics, val_baseline_metrics),
        "pair_accuracy": {"map": sc.pair_accuracy(val_score[val_left] - val_score[val_right], val_left, val_right,
            tiers, event_of), "single_image_joint": sc.pair_accuracy(val_expected[val_left] - val_expected[val_right],
            val_left, val_right, tiers, event_of)}}
    write_json(cli.out / "result.json", result)
    print(f"VAL {result['val']['baseline']}\n -> {result['val']['fused']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
