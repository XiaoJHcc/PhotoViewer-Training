"""
golden_batch2_sampler.py — 金标准第二批：train 训练团 + test 扩考卷团 混合采样导出（标星工作流）

目的（2026-07-28 用户裁定）：
  1. **扩考卷**：首批 48 团（G###）n=41 判定组太薄（每组 ±2.4pt），第二批 test 侧 ~25 团
     加厚考试，**优先 cos≥0.96 极相似带**（首批仅 2 判定组）与潜分好片带；
  2. **修残余硬错误 + 提顶 1 决断力**：train 侧 ~75 团给团内干净训练对
     （首批已证残余 4/41 错误全为 518px 可视的精细构图判别，可学型）。

设计：
  - 候选 = clusters.csv 中未被首批使用的团（按 key 排除）；成员 ≤6（团顶 + 居中度分散抽样）；
  - 分层：TEST = 极相似带优先（cos≥0.96）→ 好片带（潜分全库分位 ≥0.67）补满；
    TRAIN = 潜分带配额 顶35/中25/底15，各带内极相似团优先 ≤40%；段顶 3、事件比例软帽；
  - 平铺单文件夹 `D:/PhotoDB/dataset/golden_star2/`（`H###_X.ext`，与首批 G### 区分），
    内嵌 xmp:Rating 逐字节剥 0（复用 golden_star_export.strip_rating）；
  - key 记录 event/split/band/cos——读回后 test 团入考试、train 团入训练对，自动分流。

用法（仓根 D:/Git/PhotoViewer 下）：
    PYTHONUTF8=1 Tools/.venv/Scripts/python.exe Training/audit/golden_batch2_sampler.py
"""
from __future__ import annotations

import csv
import json
import os
import random
import shutil
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent          # Training/
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "audit"))
from render_cache import DB_DEFAULT, MANIFEST_DEFAULT, load_roots, resolve_photos  # noqa: E402
from golden_star_export import strip_rating  # noqa: E402

OUT_DIR = Path("D:/PhotoDB/dataset/golden_star2")
BATCH1_KEY = Path("D:/PhotoDB/dataset/golden_clusters/golden_clusters_key.csv")
SEED = 20260729

COS_HI = 0.96                    # 极相似带（首批 cos≥0.98 判定组仅 2，加密此带）
TEST_QUOTA = 25
TEST_HI_COS = 15
TRAIN_QUOTA = {"top": 35, "mid": 25, "bot": 15}
HI_COS_SHARE = 0.40
MAX_MEMBERS = 6
PER_SEG_CAP = 3


def main() -> int:
    split = json.load(open(ROOT / "audit" / "out" / "m3_pairs" / "split.json", encoding="utf-8"))
    ev_split = {ev: s for s, evs in split.items() for ev in evs}

    used = {(r["event"], int(r["seg_id"]), int(r["cluster_id"]))
            for r in csv.DictReader(open(BATCH1_KEY, encoding="utf-8-sig"))}

    rows = list(csv.DictReader(open(ROOT / "audit" / "out" / "clusters" / "clusters.csv",
                                    encoding="utf-8-sig")))
    lat = {r["fingerprint"]: float(r["abs_score"])
           for r in csv.DictReader(open(ROOT / "audit" / "out" / "m3_pairs" / "photos.csv",
                                        encoding="utf-8-sig"))}
    all_lat = np.array(sorted(lat.values()))

    conn = sqlite3.connect(f"file:{DB_DEFAULT}?mode=ro", uri=True)
    try:
        cls = {fp: np.frombuffer(b, dtype="<f4") for fp, b in conn.execute(
            "SELECT fingerprint, cls_vector FROM photo_features WHERE model_id=?",
            ("dinov3_vits16_f32_518_v1",))}
    finally:
        conn.close()

    by_cluster = defaultdict(list)
    for r in rows:
        if int(r["cluster_size"]) >= 2:
            by_cluster[(r["event"], int(r["seg_id"]), int(r["cluster_id"]))].append(r)

    roots = load_roots(MANIFEST_DEFAULT)
    ok_map, missing = resolve_photos(DB_DEFAULT, roots)
    print(f"源文件解析: {len(ok_map)} 命中 / {len(missing)} 缺失")

    def mean_cos(fps):
        v = [cls[f] / np.linalg.norm(cls[f]) for f in fps if f in cls]
        cs = [float(v[i] @ v[j]) for i in range(len(v)) for j in range(i + 1, len(v))]
        return float(np.mean(cs)) if cs else float("nan")

    # 候选团卡：key, members, split, cos, 潜分带(中位数分位)
    cand = []
    for key, members in by_cluster.items():
        if key in used:
            continue
        fps = [m["fingerprint"] for m in members]
        if any(f not in ok_map or f not in lat or f not in cls for f in fps):
            continue
        sp = ev_split.get(key[0])
        if sp not in ("train", "test"):
            continue
        lats = sorted(lat[f] for f in fps)
        lq = float((all_lat < lats[len(lats) // 2]).mean())
        cand.append(dict(key=key, members=members, split=sp,
                         cos=mean_cos(fps), lq=lq))
    print(f"候选团（未用）: {len(cand)}（test {sum(c['split'] == 'test' for c in cand)}）")

    rng = random.Random(SEED)
    picked, seg_count, ev_count = [], defaultdict(int), defaultdict(int)

    def take(c, band):
        picked.append((band, c))
        seg_count[(c["key"][0], c["key"][1])] += 1
        ev_count[c["key"][0]] += 1

    def eligible(c):
        return seg_count[(c["key"][0], c["key"][1])] < PER_SEG_CAP

    # ---- TEST 侧：极相似优先，再好片带补满
    test_c = [c for c in cand if c["split"] == "test"]
    rng.shuffle(test_c)
    hi = [c for c in test_c if c["cos"] >= COS_HI]
    top_band = sorted((c for c in test_c if c["cos"] < COS_HI), key=lambda c: -c["lq"])
    n = 0
    for c in hi + top_band:
        if n >= TEST_QUOTA:
            break
        if n >= TEST_HI_COS and c["cos"] >= COS_HI and sum(
                p[1]["cos"] >= COS_HI for p in picked) >= TEST_HI_COS:
            continue
        if not eligible(c):
            continue
        take(c, "exam")
        n += 1
    n_exam = n

    # ---- TRAIN 侧：潜分带配额，带内极相似优先 ≤40%
    train_c = [c for c in cand if c["split"] == "train"]
    rng.shuffle(train_c)
    for band, lo, hiq, quota in (("top", 0.67, 1.01, TRAIN_QUOTA["top"]),
                                 ("mid", 0.33, 0.67, TRAIN_QUOTA["mid"]),
                                 ("bot", 0.0, 0.33, TRAIN_QUOTA["bot"])):
        pool = [c for c in train_c if lo <= c["lq"] < hiq]
        got = hi_got = 0
        for c in pool:
            if got >= quota:
                break
            if not eligible(c):
                continue
            if c["cos"] >= COS_HI:
                if hi_got >= round(quota * HI_COS_SHARE):
                    continue
                hi_got += 1
            take(c, f"train_{band}")
            got += 1
        print(f"train_{band}: {got}/{quota}（极相似 {hi_got}）")

    # ---- 导出
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    key_rows = []
    for ix, (band, c) in enumerate(picked, 1):
        gid = f"H{ix:03d}"
        members = list(c["members"])
        if len(members) > MAX_MEMBERS:
            top = [m for m in members if m["is_cluster_top"] == "1"]
            rest = sorted((m for m in members if m["is_cluster_top"] != "1"),
                          key=lambda m: -float(m["centrality"]))
            spread = rest[:: max(1, len(rest) // (MAX_MEMBERS - 1))][: MAX_MEMBERS - len(top)]
            members = (top + spread)[:MAX_MEMBERS]
        rng.shuffle(members)
        ev, seg, cid = c["key"]
        for anon, m in zip("ABCDEF", members):
            src = ok_map[m["fingerprint"]]
            ext = os.path.splitext(src)[1]
            data, _ = strip_rating(open(src, "rb").read())
            open(OUT_DIR / f"{gid}_{anon}{ext}", "wb").write(data)
            key_rows.append(dict(gid=gid, anon=anon, fingerprint=m["fingerprint"],
                                 event=ev, seg_id=seg, cluster_id=cid, rating=m["rating"],
                                 cluster_size=m["cluster_size"], split=c["split"],
                                 band=band, cos=f"{c['cos']:.4f}", lat_q=f"{c['lq']:.3f}"))

    with open(OUT_DIR / "golden_batch2_key.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(key_rows[0]))
        w.writeheader()
        w.writerows(key_rows)
    n_img = len(key_rows)
    (OUT_DIR / "README.txt").write_text(
        "金标准团内盲选 · 第二批（标星版）\n"
        "================================\n\n"
        f"本文件夹 {n_img} 张 = {len(picked)} 组连拍（H###=组号，X=组内匿名位）。\n"
        "与首批规则相同：每组把想留的标 ≥1 星、并列标同星、真无差别全留 0；\n"
        "星级只在组内比高低、跨组无意义。全部照片已预置 0 星。\n"
        "标完告诉我即可。\n",
        encoding="utf-8")
    band_stat = defaultdict(int)
    cos_hi_n = 0
    for _, c in picked:
        band_stat[_] += 1
        cos_hi_n += c["cos"] >= COS_HI
    print(f"\n选中 {len(picked)} 团（exam {n_exam}）：{dict(band_stat)}")
    print(f"极相似带(cos≥{COS_HI}) {cos_hi_n} 团 · 事件分布 {dict(ev_count)}")
    print(f"[OK] {OUT_DIR}：{n_img} 张 + key + README（内嵌星级已剥净）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
