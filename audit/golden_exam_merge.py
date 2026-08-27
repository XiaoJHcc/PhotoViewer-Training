"""
golden_exam_merge.py — 合并金标准考试集：批1（48 团 G，全 test）+ 批2 test 团（25 团 H）+ 批3 test 团（40 团 I，读回后并入）

产出 D:/PhotoDB/dataset/golden_exam/：
  golden_exam_key.csv       合并真值键（gid,anon,fingerprint,event,…,rating,batch,cos,lat_q）
  golden_exam_readback.tsv  合并用户标星明细（golden_star_readback 格式）
用法（仓根 D:/Git/PhotoViewer 下）：
    PYTHONUTF8=1 Tools/.venv/Scripts/python.exe Training/audit/golden_exam_merge.py
"""
from __future__ import annotations

import csv
from pathlib import Path

GC = Path("D:/PhotoDB/dataset/golden_clusters")
GS1 = Path("D:/PhotoDB/dataset/golden_star")
GS2 = Path("D:/PhotoDB/dataset/golden_star2")
GS3 = Path("D:/PhotoDB/dataset/golden_star3")
OUT = Path("D:/PhotoDB/dataset/golden_exam")


def read_tsv(p: Path):
    rows = []
    for ln in open(p, encoding="utf-8"):
        ln = ln.rstrip("\n")
        if not ln or ln.startswith("gid"):
            continue
        parts = ln.split("\t")
        rows.append(parts)
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # 批1 key（全 test）+ 批2 key 中 split=test 的团
    k1 = list(csv.DictReader(open(GC / "golden_clusters_key.csv", encoding="utf-8-sig")))
    for r in k1:
        r["batch"] = "b1"
        r.setdefault("cos", "")
        r.setdefault("lat_q", "")
        r["split"] = "test"
    k2all = list(csv.DictReader(open(GS2 / "golden_batch2_key.csv", encoding="utf-8-sig")))
    test_gids = {r["gid"] for r in k2all if r["split"] == "test"}
    k2 = []
    for r in k2all:
        if r["gid"] in test_gids:
            r["batch"] = "b2"
            k2.append(r)
    keys = k1 + k2
    rb = read_tsv(GS1 / "golden_star_readback.tsv") + \
        [r for r in read_tsv(GS2 / "golden_star_readback.tsv") if r[0] in test_gids]
    n_b3 = 0
    gs3_key = GS3 / "golden_batch3_key.csv"
    gs3_rb = GS3 / "golden_star_readback.tsv"
    if gs3_key.exists() and gs3_rb.exists():   # 批3：用户标完读回后才并入
        k3all = list(csv.DictReader(open(gs3_key, encoding="utf-8-sig")))
        test3 = {r["gid"] for r in k3all if r["split"] == "test"}
        for r in k3all:
            if r["gid"] in test3:
                r["batch"] = "b3"
                keys.append(r)
        rb += [r for r in read_tsv(gs3_rb) if r[0] in test3]
        n_b3 = len(test3)
    fields = list(keys[0])
    for r in keys:
        for f_ in r:
            if f_ not in fields:
                fields.append(f_)
    for r in keys:
        for f_ in fields:
            r.setdefault(f_, "")
    with open(OUT / "golden_exam_key.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(keys)

    with open(OUT / "golden_exam_readback.tsv", "w", encoding="utf-8") as f:
        f.write("gid\tanon\tfingerprint\tuser_star\torig_rating\tis_orig_top\tis_user_pick\tgroup_status\n")
        for r in rb:
            f.write("\t".join(r) + "\n")

    gids = {r["gid"] for r in keys}
    print(f"[OK] {OUT}：{len(gids)} 团（批1 {len(set(r['gid'] for r in k1))} + 批2test "
          f"{len(test_gids)} + 批3test {n_b3}）· {len(keys)} 张 · readback {len(rb)} 行")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
