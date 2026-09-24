"""固定特征的监督机制对照：四配方、等步数、多种子、整事件开发留出，不查看 test。"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audit"))
from evaluation_protocol import abs_metrics, budget_metrics, digest, local_metrics, mean_or_none, pair_metrics, read_rows, verify_snapshot, write_json
from supervision_data import SourceBatchSampler, build_preferences, choose_development_events, load_training_labels, quarantine_conflicts

ARMS = ("legacy", "quarantine", "balanced", "source_loss")


def load_features(database: Path, model_id: str, metadata: dict) -> tuple[list[str], np.ndarray]:
    """只读加载指定版本 CLS 并逐张归一化；特征缺失或不有限时拒绝训练。"""
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        vectors = {fingerprint: np.frombuffer(blob, dtype="<f4").copy()
                   for fingerprint, blob in connection.execute(
                       "SELECT fingerprint, cls_vector FROM photo_features WHERE model_id=?", (model_id,))}
    finally:
        connection.close()
    fingerprints = sorted(metadata)
    missing = set(fingerprints) - vectors.keys()
    if missing:
        raise ValueError(f"缺少 {model_id} 特征: {len(missing)}")
    features = np.stack([vectors[fingerprint] for fingerprint in fingerprints])
    if not np.isfinite(features).all() or np.any(np.linalg.norm(features, axis=1) == 0):
        raise ValueError("特征包含非有限数或全零向量")
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    return fingerprints, features


def scores_for(model, features: torch.Tensor, fingerprints: list[str]) -> dict[str, float]:
    """关闭 dropout 后对固定特征打分，返回逐指纹有限分数。"""
    model.eval()
    with torch.no_grad():
        chunks = [model(chunk).squeeze(-1).cpu().numpy() for chunk in features.split(4096)]
    scores = np.concatenate(chunks)
    if not np.isfinite(scores).all():
        raise ValueError("模型产生非有限分数")
    return dict(zip(fingerprints, scores.astype(float).tolist()))


def run_arm(args, arm: str, seed: int, pairs, features, fingerprints, labels, metadata, development, abs_development, abs_validation) -> dict:
    """从相同种子初始化 MLP，固定步数训练指定配方，返回开发集指标及曝光账本。"""
    torch.manual_seed(seed)
    random_sampler_seed = seed + 20260921
    selected = pairs if arm == "legacy" else quarantine_conflicts(pairs)
    index_of = {fingerprint: index for index, fingerprint in enumerate(fingerprints)}
    winners = torch.tensor([index_of[pair.winner] for pair in selected], device=features.device)
    losers = torch.tensor([index_of[pair.loser] for pair in selected], device=features.device)
    source = torch.tensor([pair.source == "golden" for pair in selected], device=features.device)
    weights = torch.tensor([pair.weight for pair in selected], dtype=torch.float32, device=features.device)
    model = torch.nn.Sequential(torch.nn.Linear(features.shape[1], 128), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                                torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)).to(features.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sampler = SourceBatchSampler(selected, random_sampler_seed, args.batch_size,
                                 args.golden_per_batch if arm in ("balanced", "source_loss") else 0)
    total_loss = golden_coefficient = 0.0
    history = []
    started = time.monotonic()
    model.train()
    for step in range(1, args.steps + 1):
        sampled = torch.as_tensor(sampler.next_batch(), device=features.device)
        golden = source[sampled]
        logits = model(features[torch.cat((winners[sampled], losers[sampled]))]).squeeze(-1)
        difference = logits[:args.batch_size] - logits[args.batch_size:]
        per_pair = torch.nn.functional.softplus(-difference)
        batch_weights = weights[sampled]
        if arm == "source_loss":
            old_loss = (per_pair[~golden] * batch_weights[~golden]).sum() / batch_weights[~golden].sum()
            golden_loss = (per_pair[golden] * batch_weights[golden]).sum() / batch_weights[golden].sum()
            loss = (1 - args.golden_loss_share) * old_loss + args.golden_loss_share * golden_loss
            coefficient = args.golden_loss_share
        else:
            batch_weights = batch_weights * torch.where(golden, args.golden_weight, 1.0)
            loss = (per_pair * batch_weights).sum() / batch_weights.sum()
            coefficient = float(batch_weights[golden].sum() / batch_weights.sum())
        if not torch.isfinite(loss):
            raise ValueError(f"损失非有限: {arm}/{seed}/{step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach())
        golden_coefficient += coefficient
        if step % 500 == 0 or step == args.steps:
            history.append({"step": step, "loss_mean": total_loss / step,
                            "golden_coefficient_mean": golden_coefficient / step})
            print(f"{arm} seed={seed} {step}/{args.steps} loss={total_loss / step:.4f} "
                  f"golden系数={golden_coefficient / step:.4f}", flush=True)
    scores = scores_for(model, features, fingerprints)
    dev_labels = [row for row in labels if row["event"] in development]
    train_labels = [row for row in labels if row["event"] not in development]
    training_golden = [{"winner": pair.winner, "loser": pair.loser, "event": pair.event}
                       for pair in selected if pair.source == "golden"]
    result = {"arm": arm, "seed": seed, "seconds": time.monotonic() - started,
              "training_pairs": len(selected), "sampler": sampler.report(), "history": history,
              "golden_coefficient_mean": golden_coefficient / args.steps,
              "train_golden": pair_metrics(training_golden, scores),
              "train_local": local_metrics(train_labels, scores), "development_local": local_metrics(dev_labels, scores),
              "development_abs": abs_metrics(abs_development, metadata, scores),
              "original_val_abs": abs_metrics(abs_validation, metadata, scores),
              "development_budget_proxy": budget_metrics(metadata, scores, list(development))}
    destination = args.out / f"{arm}_seed{seed}"
    destination.mkdir()
    torch.save(model.state_dict(), destination / "head.pt")
    with (destination / "scores_development.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["fingerprint", "score"])
        for fingerprint, value in scores.items():
            if metadata[fingerprint]["event"] in development or metadata[fingerprint]["split"] == "val":
                writer.writerow([fingerprint, format(value, ".9g")])
    write_json(destination / "result.json", result)
    return result


def summarize(results: list[dict], output: Path) -> None:
    """汇总预先指定全部配方的种子均值，不按开发成绩选择 epoch 或隐藏失败配方。"""
    summary = {}
    for arm in ARMS:
        rows = [row for row in results if row["arm"] == arm]
        summary[arm] = {
            "seeds": len(rows),
            "dev_best1": [row["development_local"]["unique_best"]["best1"] for row in rows],
            "dev_best1_event_macro": [row["development_local"]["unique_best"]["best1_event_macro"] for row in rows],
            "dev_best2": [row["development_local"]["unique_best"]["best2"] for row in rows],
            "dev_abs_cross": [row["development_abs"]["cross_event"]["accuracy"] for row in rows],
            "train_golden": [row["train_golden"]["accuracy"] for row in rows],
            "golden_coefficient": [row["golden_coefficient_mean"] for row in rows],
        }
    write_json(output / "summary.json", summary)
    lines = ["# 固定特征监督对照", "", "固定 CLS + 同一 MLP；无 derived；全新初始化；按事件留出；所有配方固定步数。",
             "不评价 test，不据此宣称 LoRA 或全局美学能力达标。种子重复不是新增独立事件。", "",
             "| 配方 | 开发冠军逐种子 | 事件均值逐种子 | 训练干净对均值 | 干净损失系数 |", "|---|---|---|---:|---:|"]
    for arm, row in summary.items():
        values = "/".join(f"{value:.3f}" for value in row["dev_best1"])
        macro = "/".join(f"{value:.3f}" for value in row["dev_best1_event_macro"])
        lines.append(f"| {arm} | {values} | {macro} | {mean_or_none(row['train_golden']):.3f} | {mean_or_none(row['golden_coefficient']):.3f} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main() -> None:
    """预注册开发事件与预算，加载冻结特征，依次执行所有种子和四个对照配方。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=Path("D:/PhotoDB/dataset/photos_dataset.db"))
    parser.add_argument("--model-id", default="dinov3_vits16_f32_518_v1+clhe2.0ycc1.0")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--development-events", type=int, default=4)
    parser.add_argument("--development-seed", type=int, default=20260921)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--golden-per-batch", type=int, default=4)
    parser.add_argument("--golden-weight", type=float, default=20.0)
    parser.add_argument("--golden-loss-share", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.steps < 1 or len(args.seeds) != len(set(args.seeds)) or not args.seeds:
        raise ValueError("非法训练步数/种子")
    if not 0 < args.golden_loss_share < 1 or not 0 < args.golden_per_batch < args.batch_size:
        raise ValueError("非法干净监督份额")
    if args.lr <= 0 or args.golden_weight <= 0:
        raise ValueError("学习率和监督权重必须为正")
    verify_snapshot(args.snapshot)
    metadata = {row["fingerprint"]: row for row in read_rows(args.snapshot / "photos.csv")}
    labels = load_training_labels(args.snapshot)
    development = set(choose_development_events(labels, args.development_events, args.development_seed))
    pairs, audit = build_preferences(args.snapshot, metadata, labels, development)
    args.out.mkdir(parents=True, exist_ok=False)
    plan = {"arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "development_events": sorted(development), "arms": list(ARMS),
            "snapshot_sha256": digest(args.snapshot / "manifest.json"),
            "code_sha256": {filename: digest(Path(__file__).parent / filename) for filename in ("supervision_ablation.py", "supervision_data.py")},
            "eval_code_sha256": digest(Path(__file__).resolve().parents[1] / "audit/evaluation_protocol.py"),
            "scope": "冻结 CLS 机制探针，排除 derived 防新开发事件经教师泄漏；不等同 fw2 重训；不评价 test",
            "selection": "所有预定配方、种子固定最终步数；无早停或最优epoch挑选",
            "quarantine": "所有冲突方向均隔离，不认定新标签必真"}
    write_json(args.out / "plan.json", plan)
    write_json(args.out / "supervision_audit.json", audit)
    print("开发事件:", sorted(development), "训练来源:", audit["counts"], "冲突:", len(audit["conflicts"]), flush=True)
    fingerprints, raw_features = load_features(args.db, args.model_id, metadata)
    train_mask = np.array([metadata[fingerprint]["split"] == "train" and metadata[fingerprint]["event"] not in development for fingerprint in fingerprints])
    center = raw_features[train_mask].mean(axis=0)
    scale = raw_features[train_mask].std(axis=0)
    scale[scale < 1e-6] = 1.0
    feature_hash = hashlib.sha256("\n".join(fingerprints).encode() + raw_features.tobytes()).hexdigest()
    np.savez(args.out / "features.npz", fingerprints=np.array(fingerprints), features=raw_features, center=center, scale=scale)
    write_json(args.out / "features_manifest.json", {"sha256": feature_hash, "model_id": args.model_id,
               "photos": len(fingerprints), "dimensions": raw_features.shape[1], "normalization_train_photos": int(train_mask.sum()),
               "artifact_sha256": digest(args.out / "features.npz")})
    torch.set_num_threads(2)
    features = torch.tensor((raw_features - center) / scale, device=args.device)
    abs_development = [row for row in read_rows(args.snapshot / "abs_train.csv")
                       if all(metadata[row[field]]["event"] in development for field in ("fp_i", "fp_j"))]
    abs_validation = read_rows(args.snapshot / "abs_val.csv")
    results = []
    for seed in args.seeds:
        for arm in ARMS:
            results.append(run_arm(args, arm, seed, pairs, features, fingerprints, labels, metadata,
                                   development, abs_development, abs_validation))
    summarize(results, args.out)


if __name__ == "__main__":
    main()
