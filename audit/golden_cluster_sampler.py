"""
golden_cluster_sampler.py — 金标准团内盲选集构建（B1：台阶② 换 ground truth 的唯一出路）

背景（宪法 v1.9 / 失败矩阵 §19）：台阶②（团内选优）死因在标签本身——连拍晋级的核级
标签多为"去重压低 + 整数舍入"产物，对该标签的任何学习都在学噪声。唯一出路 = 用户
按日常选片习惯对连拍团做**干净盲选**（真工作流里用户本来就能可靠选出团内最优，
只是锦标赛标签没把它记干净）。

设计（对齐宪法决策 11：金标准必须落在 test 事件）：
  - 候选 = test 三事件（茶博/虎跑/良渚版本）内 cluster_size≥2 的团；
  - 分层：Δ≥2「真胜负」团 30 + Δ=1 团 10 + 同星团 20（同星团测"真无差异判 tie"）；
    每段至多 2 团、三事件按比例；团内成员 >6 时截断（保留团顶 + centrality 分散抽样）；
  - 成员顺序随机化、匿名 A/B/C…，**拷贝原图**（用户全分辨率判读，同真实选片）；
  - 答案：winner = A/B/.../tie。

输出 D:/PhotoDB/dataset/golden_clusters/：
  G###/A.HIF …           匿名原图（每团一个文件夹）
  golden_clusters_key.csv        内部真值键（gid,anon,fingerprint,event,…,rating）——勿给用户
  golden_clusters_answers.tsv    用户填写模板（gid \t winner）
  README.txt                     标注说明

用法（仓根 D:/Git/PhotoViewer 下）：
    PYTHONUTF8=1 Tools/.venv/Scripts/python.exe Training/audit/golden_cluster_sampler.py
"""
from __future__ import annotations

import csv
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # Training/
sys.path.insert(0, str(ROOT / "train"))
from render_cache import LEGACY_ROOT, MANIFEST_DEFAULT, load_roots, resolve_photos  # noqa: E402

DB_DEFAULT = "D:/PhotoDB/dataset/photos_dataset.db"
OUT_DIR = Path("D:/PhotoDB/dataset/golden_clusters")
SEED = 20260728

QUOTA = {"d2p": 30, "d1": 10, "same": 20}      # 团类配额（max-min 星差 ≥2 / =1 / =0）
MAX_MEMBERS = 6
PER_SEG_CAP = 3


def main() -> int:
    split = json.load(open(ROOT / "audit" / "out" / "m3_pairs" / "split.json", encoding="utf-8"))
    test_events = set(split["test"])

    rows = list(csv.DictReader(open(ROOT / "audit" / "out" / "clusters" / "clusters.csv",
                                    encoding="utf-8-sig")))
    by_cluster = defaultdict(list)
    for r in rows:
        if r["event"] in test_events and int(r["cluster_size"]) >= 2:
            by_cluster[(r["event"], int(r["seg_id"]), int(r["cluster_id"]))].append(r)

    print(f"test 事件候选团: {len(by_cluster)}")
    roots = load_roots(MANIFEST_DEFAULT)
    ok_map, missing = resolve_photos(DB_DEFAULT, roots)
    print(f"源文件解析: {len(ok_map)} 命中 / {len(missing)} 缺失")

    rng = random.Random(SEED)
    buckets = {"d2p": [], "d1": [], "same": []}
    for key, members in by_cluster.items():
        if any(m["fingerprint"] not in ok_map for m in members):
            continue
        rs = [int(m["rating"]) for m in members]
        gap = max(rs) - min(rs)
        b = "d2p" if gap >= 2 else ("d1" if gap == 1 else "same")
        buckets[b].append((key, members))
    for b in buckets:
        rng.shuffle(buckets[b])
    print("团类库存:", {b: len(v) for b, v in buckets.items()})

    picked, seg_count, ev_count = [], defaultdict(int), defaultdict(int)
    picked_keys = set()
    total_target = sum(QUOTA.values())
    ev_total = defaultdict(int)
    for key, _ in [k for v in buckets.values() for k in v]:
        ev_total[key[0]] += 1
    bucket_got = defaultdict(int)
    for b, quota in QUOTA.items():
        for key, members in buckets[b]:
            if bucket_got[b] >= quota or len(picked) >= total_target:
                break
            ev, seg = key[0], key[1]
            # 事件按比例（软）：当前事件已选未超其占比×1.5 即可；段硬顶
            if seg_count[(ev, seg)] >= PER_SEG_CAP:
                continue
            if ev_count[ev] >= max(4, round(total_target * ev_total[ev] /
                                            sum(ev_total.values()) * 1.5)):
                continue
            picked.append((b, key, members))
            picked_keys.add(key)
            seg_count[(ev, seg)] += 1
            ev_count[ev] += 1
            bucket_got[b] += 1
    # 第二轮：事件软帽放开，只守段顶与类配额，尽量用足库存
    for b, quota in QUOTA.items():
        for key, members in buckets[b]:
            if bucket_got[b] >= quota or len(picked) >= total_target:
                break
            if key in picked_keys or seg_count[(key[0], key[1])] >= PER_SEG_CAP:
                continue
            picked.append((b, key, members))
            picked_keys.add(key)
            seg_count[(key[0], key[1])] += 1
            ev_count[key[0]] += 1
            bucket_got[b] += 1
    print(f"选中 {len(picked)} 团（{dict(ev_count)}）类分布 {dict(bucket_got)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    key_rows = []
    for ix, (b, (ev, seg, cid), members) in enumerate(picked, 1):
        gid = f"G{ix:03d}"
        gdir = OUT_DIR / gid
        gdir.mkdir(exist_ok=True)
        members = list(members)
        if len(members) > MAX_MEMBERS:
            top = [m for m in members if m["is_cluster_top"] == "1"]
            rest = sorted((m for m in members if m["is_cluster_top"] != "1"),
                          key=lambda m: -float(m["centrality"]))
            spread = rest[:: max(1, len(rest) // (MAX_MEMBERS - 1))][: MAX_MEMBERS - len(top)]
            members = (top + spread)[:MAX_MEMBERS]
        rng.shuffle(members)
        for anon, m in zip("ABCDEF", members):
            src = ok_map[m["fingerprint"]]
            ext = os.path.splitext(src)[1]
            shutil.copy(src, gdir / f"{anon}{ext}")
            key_rows.append(dict(gid=gid, anon=anon, fingerprint=m["fingerprint"],
                                 event=ev, seg_id=seg, cluster_id=cid, rating=m["rating"],
                                 cluster_size=m["cluster_size"], bucket=b))

    with open(OUT_DIR / "golden_clusters_key.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(key_rows[0]))
        w.writeheader()
        w.writerows(key_rows)
    with open(OUT_DIR / "golden_clusters_answers.tsv", "w", encoding="utf-8") as f:
        f.write("# 每行一团：winner 填 A/B/...（最好的那张）；真判不出填 tie\n")
        f.write("gid\twinner\n")
        for ix, (_, (_, _, _), members) in enumerate(picked, 1):
            f.write(f"G{ix:03d}\t\n")
    (OUT_DIR / "README.txt").write_text(
        "金标准团内盲选（台阶② ground truth）\n"
        "=====================================\n\n"
        "每个 G### 文件夹 = 一组连拍（同一拍摄点的多张相似照片），文件已匿名（A/B/C/...）。\n"
        "请按您日常选片的判断，为每组挑出**您会保留的那一张**（构图/瞬间/技术质量综合第一观感，"
        "可放大看细节，与日常选片一致）。\n\n"
        "- 在 golden_clusters_answers.tsv 填写：每组一行，winner 列填字母（如 B）；\n"
        "- 若整组真的无差别，填 tie（不要硬选）；\n"
        "- 顺序无意义，组别之间无关联，可分多次完成；\n"
        "- 请勿查看 golden_clusters_key.csv（那是答案键，看了就泄题）。\n",
        encoding="utf-8")

    print(f"[OK] {OUT_DIR}：{len(picked)} 团 × ≤{MAX_MEMBERS} 张 | "
          f"key={len(key_rows)} 行 | 答案模板 + README 已就位")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
