"""
golden_star_readback.py — 标星读回：组内最高星 = 用户选择 → 生成答卷 + 判决报告

读取 D:/PhotoDB/dataset/golden_star/ 中用户标星后的照片（内嵌 XMP 与 .xmp sidecar 两通路，
取 max，与入库 ResolveRating 同哲学）：
  - 组内最高星 >0 且唯一 → 胜者 = 该匿名位；
  - 最高星 >0 且多张并列 → tie（并列位记录）；
  - 全组 0 星 → tie0（真无差别 / 全不合格 / 未标——报告里与显式 tie 合并计但单列计数）。

产出：
  - golden_star/golden_star_readback.tsv（逐张明细：用户星 / 原星 / 是否原团顶 / 是否用户选）
  - golden_clusters/golden_clusters_answers.tsv（与 golden_cluster_eval.py 兼容的答卷）
  - 控制台判决 1：用户盲选 vs 锦标赛团顶一致率（标签噪声实测；仅"有唯一胜者"组参与）

用法（仓根 D:/Git/PhotoViewer 下）：
    PYTHONUTF8=1 Tools/.venv/Scripts/python.exe Training/audit/golden_star_readback.py
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAR_DIR = Path("D:/PhotoDB/dataset/golden_star")
GC_DIR = Path("D:/PhotoDB/dataset/golden_clusters")
KEY = GC_DIR / "golden_clusters_key.csv"

PAT_ELEM = re.compile(rb"<xmp:Rating>([-0-9]+)</xmp:Rating>")
PAT_ATTR = re.compile(rb'xmp:Rating="([-0-9]+)"')
PAT_NAME = re.compile(r"^(G\d{3})_([A-F])\.[^.]+$")


def read_rating(path: Path) -> int:
    """内嵌 + sidecar（.xmp 换后缀 / 直接追加两种约定）取 max；缺失 = 0。"""
    best = 0
    cands = [path]
    stem_sidecar = path.with_suffix(".xmp")
    if stem_sidecar.exists():
        cands.append(stem_sidecar)
    full_sidecar = path.with_name(path.name + ".xmp")
    if full_sidecar.exists():
        cands.append(full_sidecar)
    for c in cands:
        try:
            data = c.read_bytes()
        except OSError:
            continue
        for pat in (PAT_ELEM, PAT_ATTR):
            for m in pat.finditer(data):
                best = max(best, int(m.group(1)))
    return max(0, min(5, best))


def main() -> int:
    key = list(csv.DictReader(open(KEY, encoding="utf-8-sig")))
    slots = {(r["gid"], r["anon"]): r for r in key}

    stars = {}            # (gid, anon) -> user star
    seen = set()
    for f in sorted(STAR_DIR.iterdir()):
        m = PAT_NAME.match(f.name)
        if not m or f.suffix.lower() == ".xmp":
            continue
        gid, anon = m.group(1), m.group(2)
        stars[(gid, anon)] = read_rating(f)
        seen.add(gid)
    by_gid = defaultdict(dict)
    for (gid, anon), st in stars.items():
        by_gid[gid][anon] = st
    print(f"读回 {len(stars)} 张 / {len(by_gid)} 组（key 共 {len(slots)} 张 / "
          f"{len(set(r['gid'] for r in key))} 组）")

    rows, answers, stat = [], {}, Counter()
    agree = dec = 0
    for gid in sorted(by_gid):
        members = by_gid[gid]
        top = max(members.values())
        winners = [a for a, v in members.items() if v == top]
        orig_top = max(int(slots[(gid, a)]["rating"]) for a in members)
        if top == 0:
            status, ans = "tie0", "tie"
        elif len(winners) > 1:
            status, ans = "tie", "tie"
        else:
            status, ans = "win", winners[0]
            dec += 1
            if int(slots[(gid, winners[0])]["rating"]) == orig_top:
                agree += 1
        stat[status] += 1
        answers[gid] = ans
        for a, ust in sorted(members.items()):
            k = slots[(gid, a)]
            rows.append(dict(gid=gid, anon=a, fingerprint=k["fingerprint"],
                             user_star=ust, orig_rating=k["rating"],
                             is_orig_top=int(int(k["rating"]) == orig_top),
                             is_user_pick=int(ans == a or (ans == "tie" and ust == top and top > 0)),
                             group_status=status))

    with open(STAR_DIR / "golden_star_readback.tsv", "w", encoding="utf-8") as f:
        f.write("gid\tanon\tfingerprint\tuser_star\torig_rating\tis_orig_top\tis_user_pick\tgroup_status\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in
                              ("gid", "anon", "fingerprint", "user_star", "orig_rating",
                               "is_orig_top", "is_user_pick", "group_status")) + "\n")
    with open(GC_DIR / "golden_clusters_answers.tsv", "w", encoding="utf-8") as f:
        f.write("# 每行一团：winner 填 A/B/...（最好的那张）；真判不出填 tie\n")
        f.write("gid\twinner\n")
        for gid in sorted(answers):
            f.write(f"{gid}\t{answers[gid]}\n")

    print(f"\n组状态: {dict(stat)}（win=唯一胜者 / tie=并列 / tie0=全 0 星）")
    if dec:
        print(f"判决 1 · 用户盲选 vs 锦标赛团顶（标签噪声实测）: 一致 {agree}/{dec} = {agree / dec:.3f}")
        print("  （远低于 1 = 坐实团内核级标签为压低+舍入产物；≈1 = 锦标赛团内标签其实干净）")
    dist = Counter(v for v in stars.values())
    print(f"用户星级分布: {dict(sorted(dist.items()))}")
    print(f"\n[OK] 明细 {STAR_DIR / 'golden_star_readback.tsv'} + 答卷已生成")
    print("下一步：PYTHONUTF8=1 Tools/.venv/Scripts/python.exe Training/audit/golden_cluster_eval.py "
          "--scores ens=Training/train/out/m8_best/scores_ens.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
