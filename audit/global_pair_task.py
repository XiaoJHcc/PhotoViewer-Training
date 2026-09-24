"""生成跨组精品直接比较题，避免把不同组星级当成统一数值。"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    """读取候选池键。"""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def make_pair(a: dict[str, str], b: dict[str, str], role: str, rng: random.Random, pair_id: str) -> dict[str, str]:
    """随机左右顺序，构造一题跨组二选一。"""
    if rng.random() < 0.5:
        a, b = b, a
    return {"pair_id": pair_id, "event": a["event"], "gid_a": a["gid"], "anon_a": a["anon"],
            "fingerprint_a": a["fingerprint"], "gid_b": b["gid"], "anon_b": b["anon"],
            "fingerprint_b": b["fingerprint"], "ens_a": a["ens"], "ens_b": b["ens"],
            "fw2_a": a["fw2"], "fw2_b": b["fw2"], "task_role": role}


def select_pairs(rows: list[dict[str, str]], pairs_per_event: int, seed: int) -> list[dict[str, str]]:
    """每事件抽取近分、模型分歧和随机跨组题。"""
    rng = random.Random(seed)
    by_event: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_event[row["event"]].append(row)
    output = []
    for event, candidates in sorted(by_event.items()):
        pair_pool = []
        for index, left in enumerate(candidates):
            for right in candidates[index + 1:]:
                if left["gid"] == right["gid"]:
                    continue
                ens_gap = abs(float(left["ens"]) - float(right["ens"]))
                fw_gap = abs(float(left["fw2"]) - float(right["fw2"]))
                opposite = (float(left["ens"]) - float(right["ens"])) * (float(left["fw2"]) - float(right["fw2"])) < 0
                role = "disagreement" if opposite else "near" if ens_gap < 0.08 else "random"
                pair_pool.append((role, ens_gap + fw_gap, left, right))
        if len(pair_pool) < pairs_per_event:
            raise ValueError(f"事件 {event} 可用跨组题不足: {len(pair_pool)}")
        chosen = []
        for role in ("near", "disagreement"):
            role_pool = [item for item in pair_pool if item[0] == role]
            role_pool.sort(key=lambda item: item[1])
            take = min(len(role_pool), pairs_per_event // 3)
            chosen.extend(role_pool[:take])
        remaining = [item for item in pair_pool if item not in chosen]
        rng.shuffle(remaining)
        chosen.extend(remaining[:pairs_per_event - len(chosen)])
        rng.shuffle(chosen)
        for left_index, (_, _, left, right) in enumerate(chosen):
            pair_id = f"{event}-Q{left_index + 1:03d}"
            output.append(make_pair(left, right, chosen[left_index][0], rng, pair_id))
    return output


def main() -> int:
    """生成跨组比较键和空白读回模板。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs-per-event", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"拒绝覆盖已有跨组题: {args.out}")
    rows = read_rows(args.key)
    pairs = select_pairs(rows, args.pairs_per_event, args.seed)
    args.out.mkdir(parents=True)
    fields = list(pairs[0])
    with (args.out / "global_pair_key.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pairs)
    with (args.out / "global_pair_readback.tsv").open("w", encoding="utf-8", newline="") as stream:
        stream.write("pair_id\twinner\tstatus\tnote\n")
        for row in pairs:
            stream.write(f"{row['pair_id']}\t\t\t\n")
    (args.out / "README.md").write_text(
        "# 跨组精品直接比较题\n\n"
        f"共 {len(pairs)} 题，每事件 {args.pairs_per_event} 题。左右顺序已随机化，题目只允许选择 A、B 或 tie。\n\n"
        "这是跨组监督的独立题型，不读取旧星级，也不把各组 keep/drop 标签转换成跨组偏好。\n",
        encoding="utf-8")
    print(f"跨组题完成: {args.out} · {len(pairs)} 题")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
