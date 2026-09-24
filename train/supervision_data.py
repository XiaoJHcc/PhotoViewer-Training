"""监督来源审计与抽样：隔离矛盾、按事件及组均衡，并记录真实曝光。"""
from __future__ import annotations

import hashlib
from itertools import combinations
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audit"))
from evaluation_protocol import read_rows
from protocol_eval import joined_labels


@dataclass(frozen=True)
class Preference:
    """一条有来源的定向偏好，数值权重仅沿用历史策略，不解释成审美级距。"""
    winner: str
    loser: str
    weight: float
    source: str
    event: str
    group: str


def identity(pair: Preference) -> tuple[str, str]:
    """返回无向照片对身份，用于检测不同来源的同向或反向重合。"""
    return tuple(sorted((pair.winner, pair.loser)))


def load_training_labels(snapshot: Path) -> list[dict]:
    """连接两批已冻结的训练盲评，校验指纹组身份后返回逐张标注。"""
    rows = []
    for batch in (2, 3):
        current = joined_labels(snapshot / f"batch{batch}_key.csv", snapshot / f"batch{batch}.tsv")
        rows.extend(row for row in current if row["split"] == "train")
    identities = [(row["gid"], row["fingerprint"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("同一标注组存在重复照片")
    return rows


def choose_development_events(labels: list[dict], count: int, seed: int) -> list[str]:
    """仅根据事件身份与标注覆盖确定开发事件，不读取模型分数或标签高低。"""
    groups = defaultdict(set)
    for row in labels:
        groups[row["event"]].add(row["gid"])
    eligible = [event for event, identities in groups.items() if len(identities) >= 5]
    if len(eligible) <= count or count < 2:
        raise ValueError("开发事件数不足或会耗尽训练事件")
    return sorted(eligible, key=lambda event: hashlib.sha256(f"{seed}:{event}".encode()).hexdigest())[:count]


def build_preferences(snapshot: Path, metadata: dict, labels: list[dict], development: set[str]) -> tuple[list[Preference], dict]:
    """构建无开发事件的直接监督；冻结探针排除依赖 M2 全库拟合的 derived。"""
    result = []
    skipped = Counter()
    for row in read_rows(snapshot / "pairs_train.csv"):
        if row["ptype"] == "derived":
            skipped["derived_excluded_to_avoid_teacher_leakage"] += 1
            continue
        left, right = metadata[row["fp_i"]], metadata[row["fp_j"]]
        if left["split"] != "train" or right["split"] != "train":
            raise ValueError("原始训练对跨入 val/test")
        if left["event"] in development or right["event"] in development:
            skipped["development_pair"] += 1
            continue
        if left["event"] != right["event"]:
            raise ValueError("旧星级对跨事件")
        difference = int(left["rating_raw"]) - int(right["rating_raw"])
        if difference == 0:
            raise ValueError("旧监督存在无方向对")
        winner, loser = (row["fp_i"], row["fp_j"]) if difference > 0 else (row["fp_j"], row["fp_i"])
        group = ":".join(sorted((left["cluster_id"], right["cluster_id"])))
        result.append(Preference(winner, loser, float(row["weight"]), row["ptype"], left["event"], group))
    grouped = defaultdict(list)
    for row in labels:
        grouped[row["gid"]].append(row)
    reconstructed = []
    for group_id, members in grouped.items():
        for left, right in combinations(members, 2):
            difference = int(left["user_star"]) - int(right["user_star"])
            if difference == 0:
                continue
            winner, loser = (left, right) if difference > 0 else (right, left)
            weight = 0.5 if abs(difference) == 1 else 1.0 if abs(difference) == 2 else 1.5
            reconstructed.append(Preference(winner["fingerprint"], loser["fingerprint"], weight, "golden", left["event"], group_id))
    original = Counter((row["fp_i"], row["fp_j"], float(row["weight"])) for row in read_rows(snapshot / "golden_pairs.csv"))
    if Counter((pair.winner, pair.loser, pair.weight) for pair in reconstructed) != original:
        raise ValueError("逐组重建的盲评来源与冻结的训练对不一致")
    for pair in reconstructed:
        left, right = metadata[pair.winner], metadata[pair.loser]
        if left["split"] != "train" or right["split"] != "train" or left["event"] != right["event"]:
            raise ValueError("盲评训练对边界错误")
        if pair.event not in development:
            result.append(pair)
    seen = defaultdict(list)
    for pair in result:
        if not np.isfinite(pair.weight) or pair.weight <= 0:
            raise ValueError("偏好权重非正或不有限")
        seen[identity(pair)].append(pair)
    conflicts = {key for key, rows in seen.items() if len({pair.winner for pair in rows}) > 1}
    audit = {"counts": dict(Counter(pair.source for pair in result)), "skipped": dict(skipped),
             "conflicts": [{"left": key[0], "right": key[1],
                            "sources": [{"source": pair.source, "winner": pair.winner, "weight": pair.weight} for pair in seen[key]]}
                           for key in sorted(conflicts)]}
    return result, audit


def quarantine_conflicts(pairs: list[Preference]) -> list[Preference]:
    """双向冲突对的所有来源均暂不训练，避免假定单次新标注一定正确。"""
    directions = defaultdict(set)
    for pair in pairs:
        directions[identity(pair)].add(pair.winner)
    return [pair for pair in pairs if len(directions[identity(pair)]) == 1]


class SourceBatchSampler:
    """生成等步数批次；均衡模式固定盲评份额并按事件、组分层抽样。"""

    def __init__(self, pairs: list[Preference], seed: int, batch_size: int, golden_per_batch: int = 0):
        """初始化来源层级与独立 RNG；批大小及盲评份额必须合法。"""
        if not 0 <= golden_per_batch < batch_size or batch_size <= 0:
            raise ValueError("非法批大小/盲评份额")
        self.pairs = pairs
        self.random = np.random.default_rng(seed)
        self.batch_size = batch_size
        self.golden_per_batch = golden_per_batch
        self.tree = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for index, pair in enumerate(pairs):
            domain = "golden" if pair.source == "golden" else "old"
            self.tree[domain][pair.event][pair.group].append(index)
        if set(self.tree) != {"golden", "old"}:
            raise ValueError("抽样需要旧监督与盲评两类来源")
        self.pools = {domain: [list(groups.values()) for groups in events.values()]
                      for domain, events in self.tree.items()}
        self.exposure = Counter()
        self.events = Counter()
        self.unique_pairs = set()
        self.unique_photos = set()
        self.batches = 0
        self.batches_with_golden = 0

    def hierarchical_pick(self, domain: str) -> int:
        """先均匀抽事件再抽组，最后抽组内对，返回原始偏好索引。"""
        events = self.pools[domain]
        groups = events[self.random.integers(len(events))]
        candidates = groups[self.random.integers(len(groups))]
        return candidates[self.random.integers(len(candidates))]

    def next_batch(self) -> np.ndarray:
        """返回一个批次并累计真实来源、事件、独立照片和成对曝光。"""
        if self.golden_per_batch:
            indexes = [self.hierarchical_pick("golden" if slot < self.golden_per_batch else "old")
                       for slot in range(self.batch_size)]
            self.random.shuffle(indexes)
            indexes = np.array(indexes)
        else:
            indexes = self.random.integers(len(self.pairs), size=self.batch_size)
        current = [self.pairs[index] for index in indexes]
        self.batches += 1
        self.batches_with_golden += any(pair.source == "golden" for pair in current)
        for pair in current:
            self.exposure[pair.source] += 1
            self.events[pair.event] += 1
            self.unique_pairs.add(identity(pair))
            self.unique_photos.update((pair.winner, pair.loser))
        return indexes

    def report(self) -> dict:
        """返回可序列化抽样账本；不把损失系数误称为梯度贡献。"""
        return {"batches": self.batches, "batches_with_golden": self.batches_with_golden,
                "exposure": dict(self.exposure), "events": dict(self.events),
                "unique_pairs": len(self.unique_pairs), "unique_photos": len(self.unique_photos)}
