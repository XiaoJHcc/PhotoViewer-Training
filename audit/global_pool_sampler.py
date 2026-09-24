"""生成跨组精品偏好试点池：只用模型分数组织候选，不把旧星级用于抽样。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_M3 = ROOT / "audit/out/m3_pairs"
DEFAULT_DB = Path("D:/PhotoDB/dataset/photos_dataset.db")
DEFAULT_ENS = ROOT / "train/out/m8_best/scores_ens.csv"
DEFAULT_FW2 = ROOT / "train/out/m5_cvfuse_hz20_fw2/scores_ep2.csv"
DEFAULT_OUT = ROOT / "audit/out/global_pool_20260921"
LEGACY_EVENT_LABEL = "2024-2-12 雾天风光"


def read_rows(path: Path) -> list[dict]:
    """读取 UTF-8 CSV；输入缺失由调用方直接报错。"""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_scores(path: Path) -> dict[str, float]:
    """加载有限模型分数并拒绝重复指纹。"""
    result = {}
    for row in read_rows(path):
        fingerprint = row["fingerprint"]
        value = float(row["score"])
        if fingerprint in result or not np.isfinite(value):
            raise ValueError(f"分数重复或非有限: {path} {fingerprint}")
        result[fingerprint] = value
    return result


def load_used(data: Path) -> set[str]:
    """读取历史盲评/考卷键，避免把已有标注照片混入新池。"""
    used = set()
    for path in (data / "golden_exam/golden_exam_key.csv",
                 data / "golden_star2/golden_batch2_key.csv",
                 data / "golden_star3/golden_batch3_key.csv",
                 data / "golden_clusters/golden_clusters_key.csv"):
        if path.exists():
            used.update(row["fingerprint"] for row in read_rows(path))
    return used


def normalize_event(raw_event: str) -> tuple[str, str]:
    """把旧批空事件名映射为已知事件，并保留来源标记。"""
    if raw_event:
        return raw_event, "m3"
    return LEGACY_EVENT_LABEL, "legacy-empty"


def select_candidates(rows: list[dict], ens: dict[str, float], fw2: dict[str, float], used: set[str],
                      groups_per_event: int, group_size: int, seed: int) -> list[dict]:
    """每事件生成固定数量六人组：高分、分歧、中段、低分和随机探索混合。"""
    rng = np.random.default_rng(seed)
    members = defaultdict(list)
    for row in rows:
        if row["split"] != "train" or row["fingerprint"] in used:
            continue
        fingerprint = row["fingerprint"]
        if fingerprint not in ens or fingerprint not in fw2:
            continue
        event, event_source = normalize_event(row["event"])
        members[(event, row["cluster_id"])].append(dict(row, event=event,
                                                         event_raw=row["event"],
                                                         event_source=event_source))
    representatives = []
    for (event, cluster), cluster_rows in members.items():
        selected = min(cluster_rows, key=lambda row: (-ens[row["fingerprint"]], row["fingerprint"]))
        fingerprint = selected["fingerprint"]
        representatives.append({"event": event, "cluster_id": cluster, "fingerprint": fingerprint,
                                "ens": ens[fingerprint], "fw2": fw2[fingerprint],
                                "disagreement": abs(ens[fingerprint] - fw2[fingerprint]),
                                "rating_raw": selected["rating_raw"], "seg_id": selected["seg_id"],
                                "event_raw": selected["event_raw"], "event_source": selected["event_source"]})
    by_event = defaultdict(list)
    for row in representatives:
        by_event[row["event"]].append(row)
    output = []
    for event, candidates in sorted(by_event.items()):
        if len(candidates) < group_size:
            continue
        candidates.sort(key=lambda row: row["ens"])
        quantile = lambda fraction: candidates[min(len(candidates) - 1, max(0, round((len(candidates) - 1) * fraction)))]
        selected = []
        for group_index in range(groups_per_event):
            anchor = quantile((group_index + 0.5) / groups_per_event)
            used_fingerprints = {row["fingerprint"] for row in selected}
            pool = [row for row in candidates if row["fingerprint"] not in used_fingerprints]
            if len(pool) < group_size:
                break
            buckets = {
                "high": sorted(pool, key=lambda row: (-row["ens"], row["fingerprint"])),
                "disagreement": sorted(pool, key=lambda row: (-row["disagreement"], row["fingerprint"])),
                "mid": sorted(pool, key=lambda row: (abs(row["ens"] - anchor["ens"]), row["fingerprint"])),
                "low": sorted(pool, key=lambda row: (row["ens"], row["fingerprint"])),
                "random": list(pool),
            }
            rng.shuffle(buckets["random"])
            chosen = []
            chosen_fingerprints = set()
            for bucket_name in ("high", "disagreement", "mid", "low", "random"):
                available = [row for row in buckets[bucket_name]
                             if row["fingerprint"] not in chosen_fingerprints]
                if not available:
                    continue
                candidate = dict(available[0], sampling_role=bucket_name)
                chosen.append(candidate)
                chosen_fingerprints.add(candidate["fingerprint"])
                if len(chosen) >= group_size - 1:
                    break
            remaining = [row for row in pool if row["fingerprint"] not in chosen_fingerprints]
            rng.shuffle(remaining)
            for row in remaining:
                chosen.append(dict(row, sampling_role="random"))
                chosen_fingerprints.add(row["fingerprint"])
                if len(chosen) == group_size:
                    break
            if len(chosen) != group_size:
                break
            selected.extend(chosen)
            gid = f"P{len(output) // group_size + 1:03d}"
            for anon, row in zip("ABCDEF", chosen):
                output.append(dict(gid=gid, anon=anon, **row))
    return output


def validate_pool(rows: list[dict], group_size: int, used: set[str]) -> None:
    """验证候选池的组内唯一性、事件一致性和历史排除。"""
    if not rows:
        raise ValueError("候选池为空")
    by_gid = defaultdict(list)
    for row in rows:
        by_gid[row["gid"]].append(row)
    for gid, group in by_gid.items():
        if len(group) != group_size:
            raise ValueError(f"{gid} 张数错误: {len(group)}")
        if len({row["event"] for row in group}) != 1:
            raise ValueError(f"{gid} 混入多个事件")
        if len({row["fingerprint"] for row in group}) != group_size:
            raise ValueError(f"{gid} 存在重复指纹")
        if len({row["cluster_id"] for row in group}) != group_size:
            raise ValueError(f"{gid} 存在重复相似团")
        if {row["anon"] for row in group} != set("ABCDEF"):
            raise ValueError(f"{gid} 匿名位不完整")
        if any(row["fingerprint"] in used for row in group):
            raise ValueError(f"{gid} 混入历史标注照片")


def enrich_paths(rows: list[dict], db: Path) -> None:
    """补充产品库中的文件名和事件相对路径，便于后续导出而不复制原图。"""
    connection = sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True)
    try:
        path_of = {row[0]: (row[1], row[2], row[3]) for row in connection.execute(
            "SELECT fingerprint, filename_noext, source_rel_path, event_label FROM photos")}
    finally:
        connection.close()
    for row in rows:
        filename, relative, event = path_of.get(row["fingerprint"], ("", "", ""))
        row["filename_noext"], row["source_rel_path"], row["db_event"] = filename, relative, event
        if not filename:
            raise ValueError(f"照片库找不到指纹: {row['fingerprint']}")


def main() -> None:
    """解析候选池参数并生成不可直接训练的全局标注试点清单。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m3", type=Path, default=DEFAULT_M3)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--ens", type=Path, default=DEFAULT_ENS)
    parser.add_argument("--fw2", type=Path, default=DEFAULT_FW2)
    parser.add_argument("--data", type=Path, default=Path("D:/PhotoDB/dataset"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--groups-per-event", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"拒绝覆盖已有候选池: {args.out}")
    if args.groups_per_event < 1 or args.group_size != 6:
        raise ValueError("当前标注协议固定每事件至少1组、每组6张")
    rows = read_rows(args.m3 / "photos.csv")
    ens, fw2 = load_scores(args.ens), load_scores(args.fw2)
    selected = select_candidates(rows, ens, fw2, load_used(args.data), args.groups_per_event, args.group_size, args.seed)
    used = load_used(args.data)
    validate_pool(selected, args.group_size, used)
    enrich_paths(selected, args.db)
    args.out.mkdir(parents=True)
    with (args.out / "global_pool_key.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    manifest = {"purpose": "直接全局精品偏好试点，不进入训练直到独立读回和冲突审计",
                "groups": len(selected) // args.group_size, "photos": len(selected),
                "events": sorted({row["event"] for row in selected}), "group_size": args.group_size,
                "groups_per_event": args.groups_per_event, "seed": args.seed,
                "sampling": "每组同一事件、不同模型代表：高分/模型分歧/中段/低分/随机探索；抽样不读取 rating_raw",
                "labeling": "组内可并留、不可判则同星或全0；跨组星级不作数值比较；读回后另做跨组成对偏好",
                "legacy_event_label": LEGACY_EVENT_LABEL,
                "legacy_event_rows": sum(row["event_source"] == "legacy-empty" for row in selected),
                "used_excluded": len(used),
                "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.m3 / "photos.csv", args.ens, args.fw2)}}
    (args.out / "README.md").write_text(
        "# 全局精品偏好试点池\n\n"
        f"共 {manifest['groups']} 组、{manifest['photos']} 张，来自 {len(manifest['events'])} 个训练事件。\n\n"
        "每组 6 张来自同一事件的不同相似团代表，抽样混合模型高分、模型分歧、中段、低分与随机探索。"
        "生成过程不读取旧 rating 作为抽样依据，旧标注只在读回后做分层审计。\n\n"
        f"旧批在 M3 中事件名为空，已按批次台账映射为“{LEGACY_EVENT_LABEL}”，原始空值保存在 event_raw。\n\n"
        "这不是训练文件。标注时只在组内表达：可以并留多张；无法判断就同星或全0。不要把不同组的星级当成数值直接比较。"
        "读回后再把不同组代表组成跨组偏好对，并保留独立复测组。\n", encoding="utf-8")
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.out / "readback_template.tsv").open("w", encoding="utf-8", newline="") as stream:
        stream.write("gid\tanon\tfingerprint\tdecision\tgroup_status\tnote\n")
        for row in selected:
            stream.write(f"{row['gid']}\t{row['anon']}\t{row['fingerprint']}\t\t\t\n")
    print(f"候选池完成: {args.out} · {manifest['groups']} 组 / {manifest['photos']} 张 / {len(manifest['events'])} 事件", flush=True)


if __name__ == "__main__":
    main()
