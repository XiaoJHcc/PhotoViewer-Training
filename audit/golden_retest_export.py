"""
golden_retest_export.py — 团内人类线复测卷导出（同卷重标，记忆冲刷后测严格复测一致性）

目的（handover §5.4）：现用人类线是跨年上界（团内冠军 0.66-0.73 / 前二 0.93），
"达到人类线即可用"的判据未钉死。本脚本把 73 团金标准考卷按**新匿名身份**重新导出，
用户隔期重标后与原始标星对比，得同人同卷复测一致率。

防记忆锚定：组号重洗（R### ≠ 原 G/H###）、组内顺序重洗（匿名位与原 A-F 无关）；
从原始照片重新导出并剥星（不走 golden_star/ 已标副本）。

产出 D:/PhotoDB/dataset/golden_retest/：
  R###_X.ext 图像 + golden_retest_key.csv（r_gid,r_anon,orig_gid,orig_anon,fingerprint,event）+ README
复测分析：用户标完后跑 golden_retest_eval.py。

用法（仓根 D:/Git/PhotoViewer 下）：
    PYTHONUTF8=1 Tools/.venv/Scripts/python.exe Training/audit/golden_retest_export.py
"""
from __future__ import annotations

import csv
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # Training/
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "audit"))
from render_cache import DB_DEFAULT, resolve_photos  # noqa: E402
from golden_star_export import strip_rating  # noqa: E402
from golden_batch3_sampler import load_roots_jsonc  # noqa: E402

MANIFESTS = ["D:/PhotoDB/dataset/manifest.2026-07-19.json",
             "D:/PhotoDB/dataset/manifest.2025-pending.json"]
EXAM_KEY = Path("D:/PhotoDB/dataset/golden_exam/golden_exam_key.csv")
OUT_DIR = Path("D:/PhotoDB/dataset/golden_retest")
SEED = 20260827


def merged_roots():
    roots: dict[str | None, list[str]] = {}
    for m in MANIFESTS:
        for ev, rs in load_roots_jsonc(m).items():
            roots.setdefault(ev, []).extend(rs)
    return roots


def main() -> int:
    key = list(csv.DictReader(open(EXAM_KEY, encoding="utf-8-sig")))
    by_gid = defaultdict(list)
    for r in key:
        by_gid[r["gid"]].append(r)
    print(f"考卷 {len(by_gid)} 团 / {len(key)} 张")

    ok_map, missing = resolve_photos(DB_DEFAULT, merged_roots())
    need = {r["fingerprint"] for r in key}
    lack = need - set(ok_map)
    if lack:
        print(f"[ERROR] 源文件解析缺失 {len(lack)} 张")
        return 1

    rng = random.Random(SEED)
    gids = sorted(by_gid)
    rng.shuffle(gids)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    key_rows = []
    for ix, ogid in enumerate(gids, 1):
        rgid = f"R{ix:03d}"
        members = list(by_gid[ogid])
        rng.shuffle(members)
        for anon, m in zip("ABCDEF", members):
            src = ok_map[m["fingerprint"]]
            ext = os.path.splitext(src)[1]
            data, _ = strip_rating(open(src, "rb").read())
            open(OUT_DIR / f"{rgid}_{anon}{ext}", "wb").write(data)
            key_rows.append(dict(r_gid=rgid, r_anon=anon, orig_gid=ogid, orig_anon=m["anon"],
                                 fingerprint=m["fingerprint"], event=m["event"]))

    with open(OUT_DIR / "golden_retest_key.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(key_rows[0]))
        w.writeheader()
        w.writerows(key_rows)
    (OUT_DIR / "README.txt").write_text(
        "金标准团内盲选 · 复测卷（标星版）\n"
        "================================\n\n"
        f"本文件夹 {len(key_rows)} 张 = {len(gids)} 组连拍（R###=组号，X=组内匿名位）。\n"
        "规则同以往：每组把想留的标 ≥1 星、并列标同星、真无差别全留 0；\n"
        "星级只在组内比高低、跨组无意义。全部照片已预置 0 星。\n"
        "标完告诉我即可。\n",
        encoding="utf-8")
    print(f"[OK] {OUT_DIR}：{len(key_rows)} 张 / {len(gids)} 组 + 映射 key（星级已剥净）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
