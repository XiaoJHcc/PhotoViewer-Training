"""v2 固定特征供料探针：同初始化、同训练步数，比较弱教师与已有直接横评。"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audit"))
from evaluation_protocol import digest, read_rows, write_json
from supervision_ablation import load_features, scores_for

ARMS = ("legacy", "no_derived", "direct_abs")


def load_preferences(m3: Path, absolute: Path, metadata: dict) -> tuple[list, list]:
    """读取训练来源并验证事件隔离，返回定向旧监督和直接横评，不消费验证标签。"""
    old, direct = [], []
    for path, target in ((m3 / "pairs_train.csv", old), (absolute / "pairs_train.csv", direct)):
        for row in read_rows(path):
            left, right = row["fp_i"], row["fp_j"]
            if any(metadata[fingerprint]["split"] != "train" for fingerprint in (left, right)):
                raise ValueError("训练对含留出照片")
            if left == right or float(row["weight"]) <= 0:
                raise ValueError("自配对或无效权重")
            if target is direct:
                winner, loser = left, right
            else:
                field = "abs_score" if row["ptype"] == "derived" else "rating_raw"
                difference = float(metadata[left][field]) - float(metadata[right][field])
                if not difference:
                    raise ValueError("配对无方向")
                winner, loser = (left, right) if difference > 0 else (right, left)
            target.append((winner, loser, float(row["weight"]), row["ptype"]))
    if not old or not direct:
        raise ValueError("监督源为空")
    return old, direct


def train_arm(args, arm: str, seed: int, old: list, direct: list,
              fingerprints: list[str], features: torch.Tensor) -> dict:
    """固定步数训练一个 MLP；直接横评臂每批四分之一曝光，记录实际来源计数。"""
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    selected = old if arm == "legacy" else [pair for pair in old if pair[3] != "derived"]
    pool = selected + direct if arm == "direct_abs" else selected
    index_of = {fingerprint: index for index, fingerprint in enumerate(fingerprints)}
    winners = torch.tensor([index_of[pair[0]] for pair in pool])
    losers = torch.tensor([index_of[pair[1]] for pair in pool])
    weights = torch.tensor([pair[2] for pair in pool], dtype=torch.float32)
    model = torch.nn.Sequential(torch.nn.Linear(features.shape[1], 128), torch.nn.ReLU(),
                                torch.nn.Dropout(0.1), torch.nn.Linear(128, 64), torch.nn.ReLU(),
                                torch.nn.Linear(64, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    exposure = Counter()
    loss_sum = 0.0
    for step in range(args.steps):
        direct_count = args.batch_size // 4 if arm == "direct_abs" else 0
        indexes = generator.integers(0, len(selected), args.batch_size - direct_count)
        if direct_count:
            indexes = np.concatenate((indexes, generator.integers(len(selected), len(pool), direct_count)))
            generator.shuffle(indexes)
        exposure.update(pool[index][3] for index in indexes)
        sampled = torch.from_numpy(indexes)
        logits = model(features[torch.cat((winners[sampled], losers[sampled]))]).squeeze(-1)
        losses = torch.nn.functional.softplus(-(logits[:args.batch_size] - logits[args.batch_size:]))
        loss = (losses * weights[sampled]).sum() / weights[sampled].sum()
        if not torch.isfinite(loss):
            raise ValueError("训练损失非有限")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.detach())
    scores = scores_for(model, features, fingerprints)
    destination = args.out / f"{arm}_s{seed}"
    destination.mkdir()
    torch.save(model.state_dict(), destination / "head.pt")
    with (destination / "scores.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["fingerprint", "score"])
        writer.writerows((fingerprint, format(score, ".9g")) for fingerprint, score in scores.items())
    result = {"arm": arm, "seed": seed, "loss": loss_sum / args.steps,
              "steps": args.steps, "batch_size": args.batch_size, "source_exposure": dict(exposure),
              "training_pairs": dict(Counter(pair[3] for pair in pool))}
    write_json(destination / "run.json", result)
    print(f"{arm} seed={seed} loss={result['loss']:.4f} done", flush=True)
    return result


def main() -> int:
    """读取训练配置，执行三配方三种子固定特征实验并保存可复算的输入哈希。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m3", type=Path, required=True)
    parser.add_argument("--abs-pairs", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=Path("D:/PhotoDB/dataset/photos_dataset.db"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.steps <= 0 or args.batch_size < 4:
        raise ValueError("步数应为正，batch 至少为四")
    torch.set_num_threads(2)
    metadata = {row["fingerprint"]: row for row in read_rows(args.m3 / "photos.csv")}
    old, direct = load_preferences(args.m3, args.abs_pairs, metadata)
    model_id = "dinov3_vits16_f32_518_v1+clhe2.0ycc1.0"
    fingerprints, features = load_features(args.db, model_id, metadata)
    features = torch.from_numpy(features)
    args.out.mkdir(parents=True, exist_ok=False)
    results = [train_arm(args, arm, seed, old, direct, fingerprints, features)
               for seed in (0, 1, 2) for arm in ARMS]
    paths = [args.m3 / "photos.csv", args.m3 / "pairs_train.csv", args.abs_pairs / "pairs_train.csv"]
    write_json(args.out / "manifest.json", {"runs": results, "model_id": model_id,
               "sha256": {str(path): digest(path) for path in paths},
               "feature_sha256": hashlib.sha256(features.numpy().tobytes()).hexdigest(),
               "scope": "冻结 CLS 探针；direct_abs 为去掉 derived 后加入 25% 横评曝光；没有测试集调参或新增人工标注。"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
