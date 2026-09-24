"""冻结评估输入并复算候选：新输出目录不可覆盖，任何输入漂移或缺分数即失败。"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path

from evaluation_protocol import VERSION, abs_metrics, budget_metrics, digest, load_scores, local_metrics, read_rows, verify_snapshot, write_json

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("D:/PhotoDB/dataset")


def freeze_inputs(output: Path, data: Path, m3: Path) -> None:
    """复制固定范围的现有标签、元数据与训练来源，生成带来源路径的哈希清单。"""
    sources = {
        "photos.csv": m3 / "photos.csv", "split.json": m3 / "split.json",
        "pairs_train.csv": m3 / "pairs_train.csv",
        "pairs_val.csv": m3 / "pairs_val.csv", "pairs_test.csv": m3 / "pairs_test.csv",
        "golden_pairs.csv": ROOT / "audit/out/golden_pairs/pairs_train.csv",
        "abs_train.csv": ROOT / "audit/out/abs_pairs/pairs_train.csv",
        "abs_val.csv": ROOT / "audit/out/abs_pairs/pairs_val.csv",
        "abs_test.csv": ROOT / "audit/out/abs_pairs/pairs_test.csv",
        "exam_key.csv": data / "golden_exam/golden_exam_key.csv",
        "exam.tsv": data / "golden_exam/golden_exam_readback.tsv",
        "retest_report.md": data / "golden_retest/retest_report.md",
        "batch2_key.csv": data / "golden_star2/golden_batch2_key.csv",
        "batch2.tsv": data / "golden_star2/golden_star_readback.tsv",
        "batch3_key.csv": data / "golden_star3/golden_batch3_key.csv",
        "batch3.tsv": data / "golden_star3/golden_star_readback.tsv",
    }
    for source in sources.values():
        if not source.is_file():
            raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for filename, source in sources.items():
        original_hash = digest(source)
        shutil.copyfile(source, output / filename)
        if digest(output / filename) != original_hash:
            raise ValueError(f"复制期间输入改变: {source}")
        files[filename] = {"source": str(source.resolve()), "sha256": original_hash}
    write_json(output / "manifest.json", {"protocol": VERSION, "files": files,
               "role": "既有开发回归与监督消融输入，不是新终考",
               "budget_basis": "原始事件照片张数；模型选每团代表；旧>=4星只作评价代理",
               "human_basis": "既有复测报告的双决胜组；不宣称独立终考或多数决真值"})
    verify_snapshot(output)
    print(f"冻结完成: {output} ({len(files)} 文件)", flush=True)


def joined_labels(key_path: Path, labels_path: Path) -> list[dict]:
    """按组号与匿名位置连接标注及事件，要求两表一一对应且指纹完全一致。"""
    keys = {(row["gid"], row["anon"]): row for row in read_rows(key_path)}
    rows = read_rows(labels_path, "\t")
    seen = set()
    for row in rows:
        identity = (row["gid"], row["anon"])
        if identity in seen or identity not in keys or keys[identity]["fingerprint"] != row["fingerprint"]:
            raise ValueError(f"标注连接不一致: {identity}")
        seen.add(identity)
        row["event"] = keys[identity]["event"]
        row["split"] = keys[identity]["split"]
    if seen != keys.keys():
        raise ValueError("标注表缺少样本")
    return rows


def evaluate(snapshot: Path, specifications: list[str], output: Path, local_only: bool) -> None:
    """在固定 test 开发回归集评估所有模型，保存分数副本及结构化报告。"""
    verify_snapshot(snapshot)
    metadata = {row["fingerprint"]: row for row in read_rows(snapshot / "photos.csv")}
    split = json.loads((snapshot / "split.json").read_text(encoding="utf-8"))
    labels = joined_labels(snapshot / "exam_key.csv", snapshot / "exam.tsv")
    report_text = (snapshot / "retest_report.md").read_text(encoding="utf-8")
    matched_ids = set(re.findall(r"^([GHIR]\d{3}):", report_text, re.MULTILINE))
    matched = [row for row in labels if row["gid"] in matched_ids]
    if not matched_ids or {row["gid"] for row in matched} != matched_ids:
        raise ValueError("复测报告组号未能完整对齐")
    fingerprints = {row["fingerprint"] for row in labels} if local_only else {fingerprint for fingerprint, row in metadata.items() if row["split"] == "test"}
    constant = dict.fromkeys(fingerprints, 0.0)
    models = {"constant": constant}
    source_paths = {}
    for specification in specifications:
        name, raw_path = specification.split("=", 1)
        if name in models or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(f"重复或非法模型名: {name}")
        source_paths[name] = Path(raw_path)
        candidate_rows = read_rows(Path(raw_path))
        if candidate_rows and "gid" in candidate_rows[0]:
            if not local_only:
                raise ValueError("按组分数只可用于局部评估，不能伪装全局标尺")
            contextual_scores = {(row["gid"], row["fingerprint"]): float(row["score"]) for row in candidate_rows}
            if len(contextual_scores) != len(candidate_rows) or not all(math.isfinite(value) for value in contextual_scores.values()):
                raise ValueError("按组分数重复或非有限")
            models[name] = contextual_scores
        else:
            models[name] = load_scores(Path(raw_path))
    result = {"protocol": VERSION, "snapshot_manifest_sha256": digest(snapshot / "manifest.json"),
              "local_only": local_only, "human_matched_groups": len(matched_ids), "models": {}}
    for name, scores in models.items():
        contextual = isinstance(next(iter(scores)), tuple)
        current = {"local": local_metrics(labels, scores, contextual), "matched_human_groups": local_metrics(matched, scores, contextual)}
        if not local_only:
            current["direct_abs"] = abs_metrics(read_rows(snapshot / "abs_test.csv"), metadata, scores)
            current["budget_proxy"] = budget_metrics(metadata, scores, split["test"])
        result["models"][name] = current
    output.mkdir(parents=True, exist_ok=False)
    for name, source in source_paths.items():
        destination = output / f"scores_{name}.csv"
        shutil.copyfile(source, destination)
        if digest(source) != digest(destination):
            raise ValueError(f"评估期间分数发生变化: {source}")
        result["models"][name]["scores_sha256"] = digest(destination)
    write_json(output / "report.json", result)
    lines = ["# 固定协议开发回归结果", "", "这是既有考卷回归，不是新事件终考。模型等分不再计为胜出。", "",
             "| 模型 | 冠军命中 | 保留两张含冠军 | 所选在人类前二 | 组数 |", "|---|---:|---:|---:|---:|"]
    for name, current in result["models"].items():
        metrics = current["local"]["unique_best"]
        lines.append(f"| {name} | {metrics['best1']:.3f} | {metrics['best2']:.3f} | {metrics['accepted_top2']:.3f} | {metrics['groups']} |")
        print(name, json.dumps(metrics, ensure_ascii=False), flush=True)
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def contextual_mix(snapshot: Path, base: Path, expert: Path, output: Path) -> None:
    """按冻结组的相似带选专家，以组号加指纹保存分数，禁止跨组覆盖同图分数。"""
    import csv
    verify_snapshot(snapshot)
    base_scores, expert_scores = load_scores(base), load_scores(expert)
    keys = read_rows(snapshot / "exam_key.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["gid", "fingerprint", "score", "expert"])
        for row in keys:
            use_expert = bool(row["cos"]) and float(row["cos"]) >= 0.96
            chosen = expert_scores if use_expert else base_scores
            writer.writerow([row["gid"], row["fingerprint"], chosen[row["fingerprint"]], int(use_expert)])
    write_json(output.with_suffix(".manifest.json"), {"snapshot": digest(snapshot / "manifest.json"),
               "base_sha256": digest(base), "expert_sha256": digest(expert), "threshold": 0.96,
               "scores_sha256": digest(output), "scope": "局部上下文分数，不作跨组全局分数"})


def main() -> None:
    """解析冻结/评估子命令并执行，输出存在时拒绝覆盖。"""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--out", type=Path, required=True)
    freeze.add_argument("--data", type=Path, default=DATA)
    freeze.add_argument("--m3", type=Path, default=ROOT / "audit/out/m3_pairs")
    run = commands.add_parser("evaluate")
    run.add_argument("--snapshot", type=Path, required=True)
    run.add_argument("--scores", nargs="+", required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--local-only", action="store_true")
    mix = commands.add_parser("mix")
    mix.add_argument("--snapshot", type=Path, required=True)
    mix.add_argument("--base", type=Path, required=True)
    mix.add_argument("--expert", type=Path, required=True)
    mix.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze_inputs(args.out, args.data, args.m3)
    elif args.command == "mix":
        contextual_mix(args.snapshot, args.base, args.expert, args.out)
    else:
        evaluate(args.snapshot, args.scores, args.out, args.local_only)


if __name__ == "__main__":
    main()
