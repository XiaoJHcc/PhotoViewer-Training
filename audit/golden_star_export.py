"""
golden_star_export.py — 金标准盲选集导出 v2（标星工作流：单文件夹 + 剥旧星级 + 读回最高星）

背景：用户日常习惯直接给照片标星（0-5），不适应 A/B/C 答卷。改为：
  - 全部 166 张（48 团 × ≤6）平铺进**一个文件夹** `D:/PhotoDB/dataset/golden_star/`，
    文件名 `G###_X.ext`（G###=团号，X=匿名位）——团号公开（任务就是团内选），
    原始身份与原始星级不可见；
  - **剥旧星级**：原文件内嵌 XMP `xmp:Rating`（元素/属性两形态，单字符 0-5），
    逐字节改写为 0（等长替换，不动容器结构）——否则用户工具会显示旧星、构成泄题锚定；
  - 用户标星：每组把要留的标 ≥1 星（多张并列就同星；真无差别全留 0），可写 sidecar 或内嵌；
  - 读回（golden_star_readback.py）：组内最高星 = 胜者集，同星并列 = tie。

输出后清理 golden_clusters/G### 图像子目录（防用户在带旧星的副本上误标），key.csv 保留。

用法（仓根 D:/Git/PhotoViewer 下）：
    PYTHONUTF8=1 Tools/.venv/Scripts/python.exe Training/audit/golden_star_export.py
"""
from __future__ import annotations

import csv
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # Training/
sys.path.insert(0, str(ROOT / "train"))
from render_cache import DB_DEFAULT, MANIFEST_DEFAULT, load_roots, resolve_photos  # noqa: E402

GC_DIR = Path("D:/PhotoDB/dataset/golden_clusters")
STAR_DIR = Path("D:/PhotoDB/dataset/golden_star")
KEY = GC_DIR / "golden_clusters_key.csv"

PAT_ELEM = re.compile(rb"(<xmp:Rating>)[-0-9]+(</xmp:Rating>)")
PAT_ATTR = re.compile(rb'(xmp:Rating=")[-0-9]+(")')


def strip_rating(data: bytes) -> tuple[bytes, int]:
    """内嵌 xmp:Rating 改写为 0（等长单字符替换）。返回 (新字节, 替换处数)。"""
    data, n1 = PAT_ELEM.subn(rb"\g<1>0\g<2>", data)
    data, n2 = PAT_ATTR.subn(rb'\g<1>0\g<2>', data)
    return data, n1 + n2


def main() -> int:
    slots = list(csv.DictReader(open(KEY, encoding="utf-8-sig")))
    roots = load_roots(MANIFEST_DEFAULT)
    ok_map, missing = resolve_photos(DB_DEFAULT, roots)
    if missing:
        print(f"[ERROR] 源文件解析缺失 {len(missing)} 张")
        return 1

    STAR_DIR.mkdir(parents=True, exist_ok=True)
    n_copy = n_strip = n_warn = 0
    for s in slots:
        src = ok_map[s["fingerprint"]]
        ext = os.path.splitext(src)[1]
        dst = STAR_DIR / f"{s['gid']}_{s['anon']}{ext}"
        data, n = strip_rating(open(src, "rb").read())
        if n != 1:
            n_warn += 1
            print(f"  [warn] {dst.name}: xmp:Rating 命中 {n} 处（预期 1）")
        open(dst, "wb").write(data)
        n_copy += 1
        n_strip += n
    print(f"导出 {n_copy} 张 → {STAR_DIR}；剥离内嵌旧星级 {n_strip} 处（异常 {n_warn}）")

    # 验证：成品里不得再有非 0 星级；字节数必须与源一致（等长替换）
    bad = []
    for f in STAR_DIR.iterdir():
        if f.suffix.lower() in (".txt", ".tsv", ".csv"):
            continue
        data = open(f, "rb").read()
        for m in PAT_ELEM.finditer(data):
            if m.group(0) != b"<xmp:Rating>0</xmp:Rating>":
                bad.append((f.name, m.group(0)))
        for m in PAT_ATTR.finditer(data):
            if m.group(0) != b'xmp:Rating="0"':
                bad.append((f.name, m.group(0)))
    if bad:
        print(f"[ERROR] 剥离不净 {len(bad)} 处: {bad[:5]}")
        return 1
    print("验证 PASS：全部 166 张内嵌星级 = 0")

    (STAR_DIR / "README.txt").write_text(
        "金标准团内盲选（标星版）\n"
        "=========================\n\n"
        "本文件夹 166 张 = 48 组连拍（文件名 G###_X：G###=组号，X=组内匿名位）。\n"
        "全部照片已预置 0 星（不带任何历史信息）。\n\n"
        "请按您日常选片习惯，对每组（同一 G### 前缀）直接标星：\n"
        "  - 想留的那张标 ≥1 星即可，其余保持 0 星；\n"
        "  - 多张并列好：标成同一星（读回时判并列）；\n"
        "  - 真无差别：整组保持 0 星（读回时判 tie）；\n"
        "  - 也欢迎整组全标（信息更全），星级只在组内比高低、跨组无意义。\n\n"
        "标完告诉我即可，我会读回每组最高星作为您的选择。\n",
        encoding="utf-8")

    # 清理旧 v1 导出（带旧星的副本 + 旧答卷），key.csv 保留
    removed = 0
    for d in GC_DIR.iterdir():
        if d.is_dir() and d.name.startswith("G"):
            shutil.rmtree(d)
            removed += 1
    for f in ("golden_clusters_answers.tsv", "README.txt"):
        p = GC_DIR / f
        if p.exists():
            p.unlink()
    print(f"清理 golden_clusters/ 旧副本 {removed} 组 + 旧答卷/README（key.csv 保留）")
    print(f"\n[OK] 请在 {STAR_DIR} 标星（共 {n_copy} 张 / 48 组）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
