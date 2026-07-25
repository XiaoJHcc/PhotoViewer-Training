"""
m2_pool_builder.py — M2 标注池生成（plan-3-2 §6 v1.0 决策 3：相似团代表制）

从 `cluster_mine.py` 产出的 clusters.csv 选代表，复制成中性名副本并抹掉内嵌 XMP 星级，
交付用户**排序制盲评**（水平相近同档、高档恒优于低档；数值无意义，宪法 §0.3 v1.7）。

选取规则（§6 v1.0，与 abs_set 去重——abs_set 141 张团顶已双职、59 张仅校验）：
    1. big_cluster_top：16+ 张团中团顶不在 abs_set 的，取团顶（并列取居中度最高）——预期 ~64；
    2. isolated_hi：单张团（孤立照）且 rating ≥3★、不在 abs_set——全取，预期 ~52（场景 5 命门）；
    3. isolated_lo：单张团且 rating ≤2★、不在 abs_set——按事件分层抽 ~30，约束低权（低端跨段
       对比历史上很少真实发生，宪法 §0.3 v1.7）；
    4. seg_top：异常段（段大小 <8 张 ∪ 段封顶 ≤2★）的段顶（段内最高星，并列取居中度最高；
       已被前几条选中的跳过）——异常段无锦标赛形状，人工直接定位。
    5. 暗放重复件（M2 GATE 自洽用）：从池中无角色偏抽 ~12%，同命名序列二次复制（盲评者
       无法分辨），key 里记 is_dup_of。

抹星与验证沿用 abs_set_sampler（等长字节替换 xmp:Rating→0 + 残留 grep 校验，两形式）；
副本命名 C0001..C####（shuffle 后编号，不泄露角色/事件/星级）；不复制 .xmp sidecar。

用法（仓根 D:/Git/PhotoViewer 下）：
    PYTHONUTF8=1 Tools/.venv/Scripts/python.exe Training/audit/m2_pool_builder.py
    （--scan 追加 dotnet scan-only 验证；--seed/--out 可改默认）
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "Training" / "audit"))
from abs_set_sampler import (  # noqa: E402
    OLD_BATCH_TAG, Resolver, load_photos, scrub_file,
)

CLUSTERS_DEFAULT = "D:/Git/PhotoViewer/Training/audit/out/clusters/clusters.csv"
ABS_KEY_DEFAULT = "D:/PhotoDB/dataset/abs_set_key.csv"
MANIFEST_DEFAULT = "D:/PhotoDB/dataset/manifest.2026-07-19.json"
OUT_DIR_DEFAULT = "D:/PhotoDB/dataset"
POOL_SUBDIR = "m2_pool"
KEY_CSV_NAME = "m2_pool_key.csv"
SEED = 7
BIG_CLUSTER_MIN = 16          # 大团阈值（§6：锦标赛形完整）
LOW_ISO_TOTAL = 30            # 低端孤立照抽样量
DUP_FRAC = 0.12               # 暗放重复件比例（M2 GATE 自洽）
SEG_SMALL = 8                 # 异常段定义 1：段大小 <8
SEG_CAP_MAX = 2               # 异常段定义 2：段封顶 ≤2★

ROLES = ("big_cluster_top", "isolated_hi", "isolated_lo", "seg_top", "dup")


def load_clusters(path: str) -> dict[str, dict]:
    """fingerprint → clusters.csv 行（数值列转 int/float）。"""
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        out[r["fingerprint"]] = dict(
            event=r["event"], seg_id=int(r["seg_id"]), cluster_id=int(r["cluster_id"]),
            cluster_size=int(r["cluster_size"]), rating=int(r["rating"]),
            is_cluster_top=int(r["is_cluster_top"]), centrality=float(r["centrality"]),
            cluster_cap=int(r["cluster_cap"]))
    return out


def select(clus: dict[str, dict], abs_fps: set[str], seed: int):
    """按 §6 规则选代表。返回 picks：fingerprint → (role, weight_class)。"""
    rng = random.Random(seed)
    picks: dict[str, tuple[str, str]] = {}

    def claim(fp: str, role: str, weight: str = "normal") -> bool:
        if fp in abs_fps or fp in picks:
            return False
        picks[fp] = (role, weight)
        return True

    # 规则 1：16+ 大团顶（团顶已入 abs_set 的整团跳过）
    by_cid: dict[int, list[dict]] = defaultdict(list)
    for fp, c in clus.items():
        by_cid[c["cluster_id"]].append(dict(fp=fp, **c))
    for cid, members in sorted(by_cid.items()):
        if members[0]["cluster_size"] < BIG_CLUSTER_MIN:
            continue
        tops = [m for m in members if m["is_cluster_top"]]
        if any(m["fp"] in abs_fps for m in tops):
            continue
        best = max(tops, key=lambda m: (m["centrality"], m["fp"]))
        claim(best["fp"], "big_cluster_top")

    # 规则 2：≥3★ 孤立照全取
    for fp, c in sorted(clus.items()):
        if c["cluster_size"] == 1 and c["rating"] >= 3:
            claim(fp, "isolated_hi")

    # 规则 3：≤2★ 孤立照按事件分层抽 LOW_ISO_TOTAL
    lo: dict[str, list[str]] = defaultdict(list)
    for fp, c in clus.items():
        if c["cluster_size"] == 1 and c["rating"] <= 2 and fp not in abs_fps:
            lo[c["event"]].append(fp)
    total_lo = sum(len(v) for v in lo.values())
    left = LOW_ISO_TOTAL
    for ev in sorted(lo, key=lambda e: -len(lo[e])):
        take = min(len(lo[ev]), max(1, round(LOW_ISO_TOTAL * len(lo[ev]) / total_lo)), left)
        for fp in sorted(rng.sample(lo[ev], take)):
            claim(fp, "isolated_lo", "low")
        left -= take
        if left <= 0:
            break

    # 规则 4：异常段（<8 张 ∪ 封顶 ≤2★）段顶
    by_seg: dict[int, list[dict]] = defaultdict(list)
    for fp, c in clus.items():
        by_seg[c["seg_id"]].append(dict(fp=fp, **c))
    for sid, members in sorted(by_seg.items()):
        cap = max(m["rating"] for m in members)
        if len(members) >= SEG_SMALL and cap > SEG_CAP_MAX:
            continue
        tops = [m for m in members if m["rating"] == cap]
        for m in sorted(tops, key=lambda m: (-m["centrality"], m["fp"])):
            if claim(m["fp"], "seg_top"):
                break

    # 规则 5：暗放重复件（从已选池抽，身份同序列、key 记 is_dup_of）
    pool_fps = sorted(picks)
    dups = sorted(rng.sample(pool_fps, max(1, round(len(pool_fps) * DUP_FRAC))))
    return picks, dups


def copy_and_scrub(picks: dict[str, tuple[str, str]], dups: list[str],
                   photos_by_fp, clus, resolver: Resolver, out_dir: Path,
                   rng: random.Random):
    """shuffle 编 C0001.. 中性名 → 复制 → 逐文件抹星校验 → (named, errors)。
    named = [(fp, new_name, role, weight, is_dup_of)]"""
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.iterdir():
        if re.fullmatch(r"C\d{4}\..+", stale.name):
            stale.unlink()

    units = [(fp, None) for fp in picks] + [(fp, fp) for fp in dups]   # (fp, is_dup_of)
    rng.shuffle(units)
    named, errors = [], []
    n_elem = n_attr = n_none = 0
    print(f"\n========== 复制 + 抹星（{len(units)} 件 → {out_dir}） ==========")
    for i, (fp, dup_of) in enumerate(units, 1):
        p = photos_by_fp[fp]
        src = resolver.resolve(p)
        if src is None:
            errors.append(f"{p.name}: 源文件不存在（{p.rel}）")
            continue
        ext = os.path.splitext(src)[1]
        new_name = f"C{i:04d}{ext}"
        dst = out_dir / new_name
        shutil.copyfile(src, dst)
        n_e, n_a, n_left = scrub_file(dst)
        if n_left > 0:
            errors.append(f"{new_name}: 抹除后仍有 {n_left} 处 xmp:Rating[1-5] 残留")
        if n_e:
            n_elem += 1
        elif n_a:
            n_attr += 1
        else:
            n_none += 1
        role, weight = picks[fp]
        named.append((fp, new_name, role, weight, dup_of or ""))
    print(f"抹除统计: 元素 {n_elem} · 属性 {n_attr} · 无标签 {n_none}；残留错误 {len(errors)}")
    return named, errors


def main() -> int:
    ap = argparse.ArgumentParser(description="M2 标注池生成（plan-3-2 §6 v1.0 决策 3）")
    ap.add_argument("--db", default="D:/PhotoDB/dataset/photos_dataset.db")
    ap.add_argument("--clusters", default=CLUSTERS_DEFAULT)
    ap.add_argument("--abs-key", default=ABS_KEY_DEFAULT)
    ap.add_argument("--manifest", default=MANIFEST_DEFAULT)
    ap.add_argument("--out", default=OUT_DIR_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    for p in (args.db, args.clusters, args.abs_key, args.manifest):
        if not Path(p).exists():
            print(f"[ERROR] 不存在: {p}", file=sys.stderr)
            return 1

    clus = load_clusters(args.clusters)
    abs_fps = {r["fingerprint"] for r in csv.DictReader(open(args.abs_key, encoding="utf-8-sig"))}
    photos = load_photos(args.db)
    photos_by_fp = {p.fp: p for p in photos}
    missing = [fp for fp in clus if fp not in photos_by_fp]
    if missing:
        print(f"[ERROR] clusters.csv 有 {len(missing)} 指纹不在 photos 表", file=sys.stderr)
        return 1

    picks, dups = select(clus, abs_fps, args.seed)
    n_role = defaultdict(int)
    for role, _ in picks.values():
        n_role[role] += 1
    print(f"选取: 大团顶 {n_role['big_cluster_top']} · ≥3★孤立 {n_role['isolated_hi']} · "
          f"低端孤立 {n_role['isolated_lo']} · 异常段顶 {n_role['seg_top']} = 池 {len(picks)}；"
          f"暗放重复件 {len(dups)}（{DUP_FRAC:.0%}）")

    resolver = Resolver(args.manifest)
    named, errors = copy_and_scrub(picks, dups, photos_by_fp, clus, resolver,
                                   Path(args.out) / POOL_SUBDIR, random.Random(args.seed + 1))
    if errors:
        print("\n[ERROR] 复制/抹除存在问题:")
        for e in errors:
            print("  " + e)
        return 1

    key_path = Path(args.out) / KEY_CSV_NAME
    # is_dup_of 指兄弟副本的 new_name（原件行该列为空；同指纹两行互认）
    orig_name = {fp: nn for fp, nn, _, _, d in named if d == ""}
    named = [(fp, nn, role, w, (orig_name.get(fp, "") if d else ""))
             for fp, nn, role, w, d in named]
    with open(key_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["new_name", "fingerprint", "event_label", "old_rating", "seg_id",
                    "cluster_id", "cluster_size", "role", "weight_class", "is_dup_of", "orig_path"])
        for fp, new_name, role, weight, dup_of in named:
            c = clus[fp]
            p = photos_by_fp[fp]
            w.writerow([new_name, fp, c["event"] or OLD_BATCH_TAG, c["rating"], c["seg_id"],
                        c["cluster_id"], c["cluster_size"], role, weight, dup_of, p.path])
    print(f"答案键: {key_path}（{len(named)} 行）")

    # 分布摘要
    by_ev_role = defaultdict(lambda: defaultdict(int))
    for fp, _, role, _, _ in named:
        by_ev_role[clus[fp]["event"] or OLD_BATCH_TAG][role] += 1
    print(f"\n{'事件':<26}{'大团顶':>7}{'孤立高':>7}{'孤立低':>7}{'异常段':>7}")
    for ev in sorted(by_ev_role):
        r = by_ev_role[ev]
        print(f"{ev:<26}{r['big_cluster_top']:>7}{r['isolated_hi']:>7}"
              f"{r['isolated_lo']:>7}{r['seg_top']:>7}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
