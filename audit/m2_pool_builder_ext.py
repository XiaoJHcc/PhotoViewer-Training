"""
m2_pool_builder_ext.py — M2 盲评扩锚池：2025 批 13 事件（derived 尺放进新事件的唯一门槛）

背景（2026-08-16/17）：2025 批入库后新事件无盲评锚点，m2_latent_ext.py 只能 NaN 占位；
飞轮首考后优先级修正为"干净团标签 > 盲评锚点 > 体量"，本池 = 扩锚动作。
用户裁定：每事件 15-20 张（≈240 + 12% 暗放重复件）。

选取规则（§6 v1.0 适配新事件，段覆盖优先——b_seg 逐段需锚点）：
    每事件配额 = clamp(round(240 × 事件段数 / 新事件总段数), 15, 20)；事件内按序取、
    每取一张优先落在**未覆盖段**（一段一锚先行）：
      1. 异常段顶（段 <8 张 ∪ 段封顶 ≤2★）——无锦标赛形状，人工直接定位；
      2. 大团顶（16+ 团，并列取居中度最高）——§6 3a 代表单位；
      3. ≥3★ 孤立照（场景 5 命门）；
      4. 其余未覆盖段的段顶（段内最高星，并列取居中度最高）；
      5. 配额未满时 ≤2★ 孤立照填充（weight=low，低端约束弱，宪法 §0.3）；
    排除已入 abs_set / 原 m2_pool 的指纹（新事件本就不在，防御性保留）。
    暗放重复件 12%（GATE 1 自洽用，同原池）。

抹星/校验复用 abs_set_sampler（scrub_file 残留校验）；命名续原池 C#### 之后；
输出 D:/PhotoDB/dataset/m2_pool_ext/ + m2_pool_ext_key.csv（schema 同 m2_pool_key.csv）。
**不动原 m2_pool/**；评完后 key/tsv 追加合并进原池再跑 m2_offset_fit。

用法（仓根 D:/Git/PhotoViewer 下）：
    PYTHONUTF8=1 Tools/.venv/Scripts/python.exe Training/audit/m2_pool_builder_ext.py
"""
from __future__ import annotations

import csv
import json
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
from m2_pool_builder import load_clusters  # noqa: E402

DB_DEFAULT = "D:/PhotoDB/dataset/photos_dataset.db"
CLUSTERS_DEFAULT = "D:/Git/PhotoViewer/Training/audit/out/clusters/clusters.csv"
ABS_KEY_DEFAULT = "D:/PhotoDB/dataset/abs_set_key.csv"
POOL_KEY_DEFAULT = "D:/PhotoDB/dataset/m2_pool_key.csv"
MANIFESTS = ["D:/PhotoDB/dataset/manifest.2026-07-19.json",
             "D:/PhotoDB/dataset/manifest.2025-pending.json"]
OUT_DIR = Path("D:/PhotoDB/dataset/m2_pool_ext")
KEY_CSV = Path("D:/PhotoDB/dataset/m2_pool_ext_key.csv")
SEED = 20260827

TOTAL = 240                   # 目标锚点总量（不含重复件）
PER_EVENT_MIN, PER_EVENT_MAX = 15, 20
BIG_CLUSTER_MIN = 16
SEG_SMALL = 8
SEG_CAP_MAX = 2
DUP_FRAC = 0.12
IS_NEW_EVENT = lambda ev: ev.startswith("2025")  # noqa: E731


def multi_resolver() -> Resolver:
    """合并两个 manifest 的 Resolver（2025 批源在 manifest.2025-pending.json，含 // 注释行）。"""
    r = Resolver(MANIFESTS[0])
    txt = Path(MANIFESTS[1]).read_text(encoding="utf-8")
    txt = "\n".join(ln for ln in txt.splitlines() if not ln.lstrip().startswith("//"))
    for folder in json.loads(txt)["folders"]:
        r.ev_roots[folder["eventLabel"]].append(folder["path"])
        r.all_roots.append(folder["path"])
    return r


def select(clus: dict[str, dict], exclude: set[str], seed: int):
    """返回 picks: fp → (role, weight)。新 13 事件、段覆盖优先。"""
    rng = random.Random(seed)
    by_ev = defaultdict(list)
    for fp, c in clus.items():
        if IS_NEW_EVENT(c["event"]) and fp not in exclude:
            by_ev[c["event"]].append(dict(fp=fp, **c))

    # 段计数 → 每事件配额（最大余数思路简化为 round + clamp）
    ev_segs = {ev: len({m["seg_id"] for m in ms}) for ev, ms in by_ev.items()}
    total_seg = sum(ev_segs.values())
    quota = {ev: max(PER_EVENT_MIN, min(PER_EVENT_MAX, round(TOTAL * n / total_seg)))
             for ev, n in ev_segs.items()}

    picks: dict[str, tuple[str, str]] = {}
    for ev in sorted(by_ev):
        members = by_ev[ev]
        q = quota[ev]
        by_seg = defaultdict(list)
        for m in members:
            by_seg[m["seg_id"]].append(m)
        covered: set[int] = set()

        def ev_count():
            return sum(1 for f in picks if clus[f]["event"] == ev)

        def claim(m, role, weight="normal"):
            if m["fp"] in picks or ev_count() >= q:
                return False
            picks[m["fp"]] = (role, weight)
            covered.add(m["seg_id"])
            return True

        # 1. 异常段顶
        for sid in sorted(by_seg):
            ms = by_seg[sid]
            cap = max(m["rating"] for m in ms)
            if len(ms) >= SEG_SMALL and cap > SEG_CAP_MAX:
                continue
            top = max(ms, key=lambda m: (m["rating"], m["centrality"], m["fp"]))
            if ev_count() < q:
                claim(top, "seg_top")
        # 2. 大团顶（只取未覆盖段）
        big = [m for m in members if m["cluster_size"] >= BIG_CLUSTER_MIN and m["is_cluster_top"]]
        for m in sorted(big, key=lambda m: (-m["cluster_size"], -m["centrality"], m["fp"])):
            if ev_count() >= q:
                break
            if m["seg_id"] not in covered:
                claim(m, "big_cluster_top")
        # 3. ≥3★ 孤立照（只取未覆盖段）
        iso_hi = [m for m in members if m["cluster_size"] == 1 and m["rating"] >= 3]
        for m in sorted(iso_hi, key=lambda m: (-m["rating"], m["fp"])):
            if ev_count() >= q:
                break
            if m["seg_id"] not in covered:
                claim(m, "isolated_hi")
        # 4. 其余未覆盖段的段顶
        for sid in sorted(by_seg):
            if sid in covered or ev_count() >= q:
                continue
            top = max(by_seg[sid], key=lambda m: (m["rating"], m["centrality"], m["fp"]))
            claim(top, "seg_top")
        # 5. 配额未满：剩余大团顶/≥3★孤立（已覆盖段也可），再 ≤2★ 孤立（低权）
        rest = [m for m in big + iso_hi if m["fp"] not in picks]
        iso_lo = [m for m in members if m["cluster_size"] == 1 and m["rating"] <= 2]
        rng.shuffle(iso_lo)
        for m in rest + iso_lo:
            if ev_count() >= q:
                break
            weight = "low" if m["rating"] <= 2 and m["cluster_size"] == 1 else "normal"
            claim(m, "isolated_lo" if weight == "low" else "big_cluster_top"
                  if m["cluster_size"] >= BIG_CLUSTER_MIN else "isolated_hi", weight)
    return picks, quota


def main() -> int:
    for p in (DB_DEFAULT, CLUSTERS_DEFAULT, ABS_KEY_DEFAULT, POOL_KEY_DEFAULT, *MANIFESTS):
        if not Path(p).exists():
            print(f"[ERROR] 不存在: {p}", file=sys.stderr)
            return 1

    clus = load_clusters(CLUSTERS_DEFAULT)
    exclude = {r["fingerprint"] for r in csv.DictReader(open(ABS_KEY_DEFAULT, encoding="utf-8-sig"))}
    old_key = list(csv.DictReader(open(POOL_KEY_DEFAULT, encoding="utf-8-sig")))
    exclude |= {r["fingerprint"] for r in old_key}
    c0 = max(int(re.match(r"C(\d+)", r["new_name"]).group(1)) for r in old_key)

    picks, quota = select(clus, exclude, SEED)
    n_role = defaultdict(int)
    for role, _ in picks.values():
        n_role[role] += 1
    print(f"选取: 大团顶 {n_role['big_cluster_top']} · ≥3★孤立 {n_role['isolated_hi']} · "
          f"低端孤立 {n_role['isolated_lo']} · 段顶 {n_role['seg_top']} = 池 {len(picks)}")

    rng = random.Random(SEED + 1)
    pool_fps = sorted(picks)
    dups = sorted(rng.sample(pool_fps, max(1, round(len(pool_fps) * DUP_FRAC))))

    photos = load_photos(DB_DEFAULT)
    photos_by_fp = {p.fp: p for p in photos}
    resolver = multi_resolver()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    units = [(fp, "") for fp in picks] + [(fp, fp) for fp in dups]
    rng.shuffle(units)
    named, errors = [], []
    n_elem = n_attr = n_none = 0
    for i, (fp, dup_of) in enumerate(units, 1):
        p = photos_by_fp.get(fp)
        if p is None:
            errors.append(f"{fp}: 不在 photos 表")
            continue
        src = resolver.resolve(p)
        if src is None:
            errors.append(f"{p.name}: 源文件不存在（{p.rel}）")
            continue
        ext = os.path.splitext(src)[1]
        new_name = f"C{c0 + i:04d}{ext}"
        dst = OUT_DIR / new_name
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
        named.append((fp, new_name, role, weight, dup_of))
    if errors:
        print("\n[ERROR] 复制/抹除存在问题:")
        for e in errors:
            print("  " + e)
        return 1
    print(f"抹除统计: 元素 {n_elem} · 属性 {n_attr} · 无标签 {n_none}；残留错误 0")

    orig_name = {fp: nn for fp, nn, _, _, d in named if d == ""}
    with open(KEY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["new_name", "fingerprint", "event_label", "old_rating", "seg_id",
                    "cluster_id", "cluster_size", "role", "weight_class", "is_dup_of", "orig_path"])
        for fp, new_name, role, weight, dup_of in named:
            c = clus[fp]
            p = photos_by_fp[fp]
            w.writerow([new_name, fp, c["event"] or OLD_BATCH_TAG, c["rating"], c["seg_id"],
                        c["cluster_id"], c["cluster_size"], role, weight,
                        orig_name.get(fp, "") if dup_of else "", p.path])
    print(f"答案键: {KEY_CSV}（{len(named)} 行 = 池 {len(picks)} + 重复件 {len(dups)}）")

    # 分布 + 段覆盖摘要
    by_ev = defaultdict(lambda: [0, set()])
    for fp in picks:
        by_ev[clus[fp]["event"]][0] += 1
    all_seg = defaultdict(set)
    for fp, c in clus.items():
        if IS_NEW_EVENT(c["event"]):
            all_seg[c["event"]].add(c["seg_id"])
    cov_seg = defaultdict(set)
    for fp in picks:
        cov_seg[clus[fp]["event"]].add(clus[fp]["seg_id"])
    print(f"\n{'事件':<26}{'配额':>5}{'实取':>5}{'段覆盖':>10}")
    for ev in sorted(by_ev):
        print(f"{ev:<26}{quota[ev]:>5}{by_ev[ev][0]:>5}"
              f"{len(cov_seg[ev])}/{len(all_seg[ev]):>5}")
    print(f"\n[OK] {OUT_DIR}（盲评后把 key/tsv 追加合并进原 m2_pool 再跑 m2_offset_fit）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
