"""
golden_batch3_sampler.py — 金标准第三批：考卷加密极相似带 + 训练干净对上量（标星工作流）

目的（2026-08-17 飞轮首考瓶颈改判后，用户裁定 190 组）：
  1. **扩考卷 + 解带内噪声**：test 侧 +40 团，cos≥0.96 极相似带优先取满 30
     （现带内 n=9，±1 组≈11pt；加密后复核水平度规则候选 0.53/0.92/0.93）；
  2. **干净训练对上量**：train 侧 150 团，读回后与批2 的 325 对合并 → ~1000 对
     （75 团≈325 对的产率）；瓶颈=干净团标签，这是唯一直接杠杆。

与批2 的差异：
  - 候选池 = 重算后全库 7006 团（含 2025 批 13 事件），排除批1+批2 已用 148 团；
  - roots 合并两个 manifest（旧批 manifest.2026-07-19 + 2025 批 manifest.2025-pending）；
  - 新事件团无潜分（abs_score=NaN）→ 分层 fallback：团顶星级带（4-5★→top / 2-3★→mid / 0-1★→bot）；
  - train 侧极相似带占比上限 40% → 50%（本批目的即加密极相似带）；
  - 命名 I###（G=批1 / H=批2），readback PAT_NAME 已扩展 [GHIR]。

用法（仓根 D:/Git/PhotoViewer 下）：
    PYTHONUTF8=1 Tools/.venv/Scripts/python.exe Training/audit/golden_batch3_sampler.py
"""
from __future__ import annotations

import csv
import json
import os
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent          # Training/
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "audit"))
from render_cache import DB_DEFAULT, resolve_photos  # noqa: E402
from golden_star_export import strip_rating  # noqa: E402

MANIFESTS = ["D:/PhotoDB/dataset/manifest.2026-07-19.json",
             "D:/PhotoDB/dataset/manifest.2025-pending.json"]
OUT_DIR = Path("D:/PhotoDB/dataset/golden_star3")
USED_KEYS = [Path("D:/PhotoDB/dataset/golden_clusters/golden_clusters_key.csv"),
             Path("D:/PhotoDB/dataset/golden_star2/golden_batch2_key.csv")]
SEED = 20260827

COS_HI = 0.96                    # 极相似带（考卷带内加密目标）
TEST_QUOTA = 40
TEST_HI_COS = 30
TRAIN_QUOTA = {"top": 60, "mid": 50, "bot": 40}
HI_COS_SHARE = 0.50
MAX_MEMBERS = 6
PER_SEG_CAP = 3


def load_roots_jsonc(path):
    """render_cache.load_roots 的 JSONC 版（manifest.2025-pending.json 含 // 注释行）。"""
    txt = Path(path).read_text(encoding="utf-8")
    txt = "\n".join(ln for ln in txt.splitlines() if not ln.lstrip().startswith("//"))
    m = json.loads(txt)
    roots: dict[str | None, list[str]] = {}
    for f in m["folders"]:
        roots.setdefault(f.get("eventLabel"), []).append(f["path"])
    return roots


def merged_roots():
    roots: dict[str | None, list[str]] = {}
    for m in MANIFESTS:
        for ev, rs in load_roots_jsonc(m).items():
            roots.setdefault(ev, []).extend(rs)
    return roots


def main() -> int:
    split = json.load(open(ROOT / "audit" / "out" / "m3_pairs" / "split.json", encoding="utf-8"))
    ev_split = {ev: s for s, evs in split.items() for ev in evs}

    used = set()
    for p in USED_KEYS:
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            used.add(r["fingerprint"])

    rows = list(csv.DictReader(open(ROOT / "audit" / "out" / "clusters" / "clusters.csv",
                                    encoding="utf-8-sig")))
    # 潜分（旧事件可用；新事件 NaN → 分层走团顶星级带 fallback）
    lat = {}
    for r in csv.DictReader(open(ROOT / "audit" / "out" / "m3_pairs" / "photos.csv",
                                 encoding="utf-8-sig")):
        try:
            lat[r["fingerprint"]] = float(r["abs_score"])
        except ValueError:
            lat[r["fingerprint"]] = float("nan")
    all_lat = np.array(sorted(v for v in lat.values() if v == v))

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

    roots = merged_roots()
    ok_map, missing = resolve_photos(DB_DEFAULT, roots)
    print(f"源文件解析: {len(ok_map)} 命中 / {len(missing)} 缺失")

    def mean_cos(fps):
        v = [cls[f] / np.linalg.norm(cls[f]) for f in fps if f in cls]
        cs = [float(v[i] @ v[j]) for i in range(len(v)) for j in range(i + 1, len(v))]
        return float(np.mean(cs)) if cs else float("nan")

    # 候选团卡：key, members, split, cos, 带（潜分分位；NaN → 团顶星级带）
    cand = []
    for key, members in by_cluster.items():
        if any(member["fingerprint"] in used for member in members):
            continue
        fps = [m["fingerprint"] for m in members]
        if any(f not in ok_map or f not in lat or f not in cls for f in fps):
            continue
        sp = ev_split.get(key[0])
        if sp not in ("train", "test"):
            continue
        lats = sorted(lat[f] for f in fps)
        med = lats[len(lats) // 2]
        if med == med:   # 潜分可用
            band = float((all_lat < med).mean())
        else:            # 新事件 fallback：团顶星级带
            top_star = max(int(m["rating"]) for m in members)
            band = {0: 0.15, 1: 0.15, 2: 0.5, 3: 0.5, 4: 0.85, 5: 0.85}[top_star]
        cand.append(dict(key=key, members=members, split=sp,
                         cos=mean_cos(fps), lq=band))
    print(f"候选团（未用）: {len(cand)}（test {sum(c['split'] == 'test' for c in cand)}）")

    rng = random.Random(SEED)
    picked, seg_count, ev_count = [], defaultdict(int), defaultdict(int)

    def take(c, band):
        picked.append((band, c))
        seg_count[(c["key"][0], c["key"][1])] += 1
        ev_count[c["key"][0]] += 1

    def eligible(c):
        return seg_count[(c["key"][0], c["key"][1])] < PER_SEG_CAP

    # ---- TEST 侧 40：极相似带优先取满 30，余从 cos<0.96 按带分位降序（好片带）补
    test_c = [c for c in cand if c["split"] == "test"]
    rng.shuffle(test_c)
    hi = [c for c in test_c if c["cos"] >= COS_HI]
    top_band = sorted((c for c in test_c if c["cos"] < COS_HI), key=lambda c: -c["lq"])
    n = n_hi = 0
    for c in hi + top_band:
        if n >= TEST_QUOTA:
            break
        if c["cos"] >= COS_HI:
            if n_hi >= TEST_HI_COS:
                continue
            n_hi += 1
        if not eligible(c):
            continue
        take(c, "exam")
        n += 1
    n_exam = n
    print(f"exam: {n_exam}/{TEST_QUOTA}（极相似 {n_hi}）")

    # ---- TRAIN 侧 150：带配额 顶60/中50/底40，带内极相似优先 ≤50%
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
        gid = f"I{ix:03d}"
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

    with open(OUT_DIR / "golden_batch3_key.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(key_rows[0]))
        w.writeheader()
        w.writerows(key_rows)
    n_img = len(key_rows)
    (OUT_DIR / "README.txt").write_text(
        "金标准团内盲选 · 第三批（标星版）\n"
        "================================\n\n"
        f"本文件夹 {n_img} 张 = {len(picked)} 组连拍（I###=组号，X=组内匿名位）。\n"
        "与前两批规则相同：每组把想留的标 ≥1 星、并列标同星、真无差别全留 0；\n"
        "星级只在组内比高低、跨组无意义。全部照片已预置 0 星。\n"
        "张数较多，可分多次标（按组号顺序即可）。标完告诉我即可。\n",
        encoding="utf-8")
    band_stat = defaultdict(int)
    cos_hi_n = 0
    for b, c in picked:
        band_stat[b] += 1
        cos_hi_n += c["cos"] >= COS_HI
    print(f"\n选中 {len(picked)} 团（exam {n_exam}）：{dict(band_stat)}")
    print(f"极相似带(cos≥{COS_HI}) {cos_hi_n} 团 · 事件分布 {dict(ev_count)}")
    print(f"[OK] {OUT_DIR}：{n_img} 张 + key + README（内嵌星级已剥净）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
