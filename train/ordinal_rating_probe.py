"""有序评分受控训练：共享分数与递增门槛，不作等距星级回归。

完整train事件三折三种子，单图/无标签邻域与线性/非线性二乘二对照。
固定预算不早停、不读val/test，按统一AP与全部低分档诊断结果解释。
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "audit"))
from evaluation_protocol import RATING_VERSION, digest, write_json
from rating_baseline import measure, write_scores
from context_rating_probe import load_inputs, contextual_features, compare


class OrdinalHead(nn.Module):
    """单一潜在分数与严格递增的三个门槛，产生天然单调的累计概率。"""

    def __init__(self, dimension: int, nonlinear: bool):
        """创建线性或固定128维GELU评分网络；门槛间隔以softplus确保为正。"""
        super().__init__()
        self.network = nn.Sequential(nn.Linear(dimension, 128), nn.GELU(), nn.Dropout(0.2),
                                     nn.Linear(128, 1)) if nonlinear else nn.Linear(dimension, 1)
        self.base = nn.Parameter(torch.tensor(-1.0))
        self.raw_gaps = nn.Parameter(torch.zeros(2))

    def forward(self, features):
        """返回三个累计二元logit，门槛递增且不使用标签差的数值距离。"""
        thresholds = torch.cat((self.base[None], self.base + torch.cumsum(functional.softplus(self.raw_gaps) + 1e-4, dim=0)))
        return self.network(features) - thresholds[None]


def probabilities_from_logits(logits):
    """由单调累计logit还原四类概率，拒绝产生负值或总和偏离一的非法结果。"""
    cumulative = torch.sigmoid(logits)
    values = torch.cat((1 - cumulative[:, :1], cumulative[:, :-1] - cumulative[:, 1:], cumulative[:, -1:]), dim=1)
    if not torch.isfinite(values).all() or torch.any(values < -1e-7):
        raise ValueError("有序头产生非法概率")
    return values


def fit_fold(features, truth, event_of, training_events, held_out, nonlinear, seed):
    """固定1500步训练事件/类别均衡有序头，返回留出概率与训练损失轨迹。"""
    torch.manual_seed(seed)
    generator = np.random.default_rng(20261015 + seed)
    fitting = np.flatnonzero(np.isin(event_of, sorted(set(training_events) - set(held_out))))
    testing = np.flatnonzero(np.isin(event_of, held_out))
    domains = Counter((event_of[index], int(truth[index])) for index in fitting)
    classes_per_event = Counter(event for event, category in domains)
    sampling = np.array([1 / (domains[(event_of[index], int(truth[index]))] * classes_per_event[event_of[index]])
                         for index in fitting])
    sampling /= sampling.sum()
    model = OrdinalHead(features.shape[1], nonlinear).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    targets = torch.as_tensor(truth[:, None] >= np.array([1, 2, 3])[None], device="cuda", dtype=torch.float32)
    history = []
    model.train()
    for step in range(1500):
        selected = generator.choice(fitting, size=256, replace=True, p=sampling)
        selected_tensor = torch.as_tensor(selected, device="cuda")
        logits = model(features[selected_tensor])
        loss = functional.binary_cross_entropy_with_logits(logits, targets[selected_tensor])
        if not torch.isfinite(loss):
            raise ValueError("非有限有序训练损失")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 500 == 0 or step == 1499:
            history.append({"step": step + 1, "loss": float(loss.detach().cpu())})
    model.eval()
    with torch.inference_mode():
        probabilities = probabilities_from_logits(model(features[torch.as_tensor(testing, device="cuda")])).cpu().numpy()
    return testing, probabilities, {"seed": seed, "held_out_events": held_out, "history": history,
        "training_photos": len(fitting), "testing_photos": len(testing)}


def main() -> int:
    """执行预登记二乘二三种子完整训练事件OOF，保存概率和每种子结果，不打开val/test。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photos", type=Path, default=ROOT / "audit/out/m3_pairs_v2_20260924b/photos.csv")
    parser.add_argument("--split-json", type=Path, default=ROOT / "data/split-v2-20260924.json")
    parser.add_argument("--features", type=Path, default=ROOT / "train/out/recovery_work/corrected_features.npz")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise ValueError("固定GPU运行环境不可用")
    torch.set_num_threads(2)
    threadpool_limits(2)
    started = time.monotonic()
    metadata, fingerprints, visual, dino = load_inputs(args)
    train_fingerprints = [fingerprint for fingerprint in fingerprints if metadata[fingerprint]["split"] == "train"]
    indexes = [index for index, fingerprint in enumerate(fingerprints) if metadata[fingerprint]["split"] == "train"]
    metadata = {fingerprint: metadata[fingerprint] for fingerprint in train_fingerprints}
    event_of = np.array([metadata[fingerprint]["event"] for fingerprint in train_fingerprints])
    truth = np.array([min(int(metadata[fingerprint]["rating_raw"]), 3) for fingerprint in train_fingerprints])
    events = sorted(set(event_of.tolist()), key=lambda event: hashlib.sha256(("20260926:" + event).encode()).hexdigest())
    folds = [events[index::3] for index in range(3)]
    single = visual[indexes]
    context, context_identity = contextual_features(single, dino[indexes], train_fingerprints, event_of, False)
    inputs = [args.photos, args.split_json, args.features, args.features.with_suffix(".json"), Path(__file__),
              ROOT / "train/rating_baseline.py", ROOT / "train/context_rating_probe.py", ROOT / "audit/evaluation_protocol.py"]
    plan = {"protocol": RATING_VERSION, "folds": folds, "seeds": [0, 1, 2], "steps": 1500,
            "batch": 256, "learning_rate": 0.001, "weight_decay": 0.01, "hidden": 128,
            "dropout": 0.2, "scope": "完整train事件OOF；有序分类非等距回归；不打开val/test",
            "context_identity": context_identity, "input_sha256": {str(path.resolve()): digest(path) for path in inputs}}
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "plan.json", plan)
    result = {"plan": plan, "models": {}}
    for name, array, nonlinear in (("single_linear", single, False), ("context_linear", context, False),
                                    ("single_mlp", single, True), ("context_mlp", context, True)):
        features = torch.tensor(array, device="cuda", dtype=torch.float32)
        seed_probabilities, seed_results = [], []
        for seed in plan["seeds"]:
            probabilities = np.full((len(train_fingerprints), 4), np.nan, dtype=np.float64)
            traces = []
            for held_out in folds:
                testing, values, trace = fit_fold(features, truth, event_of, events, held_out, nonlinear, seed)
                probabilities[testing] = values
                traces.append(trace)
            if not np.isfinite(probabilities).all():
                raise ValueError("OOF覆盖不完整")
            metrics = measure(metadata, dict(zip(train_fingerprints, probabilities.tolist())), events)
            seed_results.append({"seed": seed, "metrics": metrics, "traces": traces})
            seed_probabilities.append(probabilities)
            print(f"{name} seed={seed} AP>=3={metrics['ranking']['3']['ap_macro']:.4f} "
                  f"balanced_accuracy={metrics['rating']['balanced_accuracy_macro']:.4f}", flush=True)
        ensemble = np.mean(seed_probabilities, axis=0)
        metrics = measure(metadata, dict(zip(train_fingerprints, ensemble.tolist())), events)
        result["models"][name] = {"metrics": metrics, "seeds": seed_results}
        write_scores(args.out / f"{name}_oof_scores.csv", train_fingerprints, ensemble)
        write_json(args.out / f"{name}.json", result["models"][name])
        print(f"{name} ensemble AP>=3={metrics['ranking']['3']['ap_macro']:.4f} "
              f"balanced_accuracy={metrics['rating']['balanced_accuracy_macro']:.4f}", flush=True)
        del features
    reference = json.loads((ROOT / "train/out/context_rating_probe_20261014/result.json").read_text(encoding="utf-8"))
    result["softmax_reference_sha256"] = digest(ROOT / "train/out/context_rating_probe_20261014/result.json")
    result["difference_vs_softmax"] = {name: compare(model["metrics"], reference["models"][
        "similar_context" if name.startswith("context") else "single"]["metrics"])
        for name, model in result["models"].items()}
    if any(digest(path) != plan["input_sha256"][str(path.resolve())] for path in inputs):
        raise ValueError("运行期间输入漂移")
    result["seconds"] = time.monotonic() - started
    write_json(args.out / "result.json", result)
    print(f"完成有序探针 {result['seconds']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
