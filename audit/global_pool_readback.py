"""校验全局精品试点读回，并生成只含组内关系的训练候选。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


DECISIONS = {"keep", "drop", "tie"}
GROUP_STATUSES = {"ranked", "tie"}


def read_delimited(path: Path) -> list[dict[str, str]]:
    """按扩展名读取 CSV 或 TSV。"""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
        return list(csv.DictReader(stream, delimiter=delimiter))


def sha256(path: Path) -> str:
    """计算输入文件 SHA256。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_key(path: Path) -> tuple[list[dict[str, str]], dict[tuple[str, str], dict[str, str]]]:
    """读取候选键并检查组内匿名位与指纹唯一性。"""
    rows = read_delimited(path)
    required = {"gid", "anon", "fingerprint", "event"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"候选键缺少字段: {sorted(required)}")
    by_key = {}
    by_gid: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row["gid"], row["anon"])
        if key in by_key or not row["fingerprint"]:
            raise ValueError(f"候选键重复或缺指纹: {key}")
        by_key[key] = row
        by_gid[row["gid"]].append(row)
    for gid, group in by_gid.items():
        if len(group) != 6 or len({row["fingerprint"] for row in group}) != 6:
            raise ValueError(f"{gid} 不是六张不同照片")
        if len({row["event"] for row in group}) != 1:
            raise ValueError(f"{gid} 混入多个事件")
    return rows, by_key


def parse_readback(path: Path, key: dict[tuple[str, str], dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    """读取人工读回并拒绝遗漏、重复和非法标签。"""
    rows = read_delimited(path)
    required = {"gid", "anon", "fingerprint", "decision", "group_status"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"读回文件缺少字段: {sorted(required)}")
    result = {}
    statuses: dict[str, str] = {}
    for row in rows:
        item_key = (row["gid"], row["anon"])
        if item_key not in key:
            raise ValueError(f"读回出现未知照片: {item_key}")
        if item_key in result or row["fingerprint"] != key[item_key]["fingerprint"]:
            raise ValueError(f"读回键不匹配或重复: {item_key}")
        if row["decision"] not in DECISIONS or row["group_status"] not in GROUP_STATUSES:
            raise ValueError(f"非法标签: {item_key} {row['decision']} / {row['group_status']}")
        if row["gid"] in statuses and statuses[row["gid"]] != row["group_status"]:
            raise ValueError(f"同组混用 group_status: {row['gid']}")
        statuses[row["gid"]] = row["group_status"]
        if row["group_status"] == "tie" and row["decision"] != "tie":
            raise ValueError(f"tie 组必须每张标 tie: {item_key}")
        if row["group_status"] == "ranked" and row["decision"] == "tie":
            raise ValueError(f"ranked 组不能混用 tie: {item_key}")
        result[item_key] = row
    if set(result) != set(key):
        missing = sorted(set(key) - set(result))
        raise ValueError(f"读回缺少 {len(missing)} 张照片，例如 {missing[:3]}")
    return result


def write_outputs(out: Path, key_rows: list[dict[str, str]], readback: dict[tuple[str, str], dict[str, str]],
                  key_path: Path, readback_path: Path) -> dict:
    """保存带来源的照片标签、组内偏好对和审计报告。"""
    out.mkdir(parents=True, exist_ok=False)
    labels = []
    by_gid: dict[str, list[dict[str, str]]] = defaultdict(list)
    key_by_pair = {(row["gid"], row["anon"]): row for row in key_rows}
    for item_key, label in readback.items():
        row = dict(key_by_pair[item_key])
        row.update(decision=label["decision"], group_status=label["group_status"], note=label.get("note", ""))
        labels.append(row)
        by_gid[row["gid"]].append(row)
    label_fields = list(labels[0])
    with (out / "photo_labels.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=label_fields)
        writer.writeheader()
        writer.writerows(sorted(labels, key=lambda row: (row["gid"], row["anon"])))

    pairs = []
    for gid, group in sorted(by_gid.items()):
        winners = [row for row in group if row["decision"] == "keep"]
        losers = [row for row in group if row["decision"] == "drop"]
        for winner in winners:
            for loser in losers:
                pairs.append({"gid": gid, "winner": winner["fingerprint"], "loser": loser["fingerprint"],
                              "winner_anon": winner["anon"], "loser_anon": loser["anon"],
                              "source": "global_pool_readback"})
    with (out / "within_group_pairs.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["gid", "winner", "loser", "winner_anon", "loser_anon", "source"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pairs)

    report = {"protocol": "global-pool-readback-v1", "groups": len(by_gid), "photos": len(labels),
              "groups_ranked": sum(next(iter(group))["group_status"] == "ranked" for group in by_gid.values()),
              "groups_tie": sum(next(iter(group))["group_status"] == "tie" for group in by_gid.values()),
              "keep": sum(row["decision"] == "keep" for row in labels),
              "drop": sum(row["decision"] == "drop" for row in labels),
              "tie": sum(row["decision"] == "tie" for row in labels),
              "within_group_pairs": len(pairs), "key_sha256": sha256(key_path),
              "readback_sha256": sha256(readback_path),
              "warning": "不同组的星级不能直接转换为跨组偏好；跨组任务必须另行同题比较。"}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(
        "# 全局精品试点读回\n\n"
        "`ranked` 组只表示 keep/drop 集合，允许多个 keep；`tie` 组整组不可判，不生成偏好对。\n\n"
        "本工具只生成同组 keep 胜 drop 的关系。不同组的星级不能直接比较；需要跨组精品监督时，必须另建同题跨组比较任务。\n",
        encoding="utf-8")
    return report


def main() -> int:
    """解析候选键与读回文件并生成审计产物。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--readback", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    key_rows, key = load_key(args.key)
    readback = parse_readback(args.readback, key)
    report = write_outputs(args.out, key_rows, readback, args.key, args.readback)
    print(f"读回通过: {args.out} · {report['groups']} 组 / {report['within_group_pairs']} 组内偏好对")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
