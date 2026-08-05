"""
horizon_cv.py — CV 地平线检测（纯 numpy/PIL/scipy，无 cv2）+ EXIF 横滚交叉验证

动机：
  1. golden_star2 验证报 EXIF roll vs 视觉地平线 mean|err|=1.27° —— 需要定量两者关系，
     判明水平度信号在考卷上失效是"信号噪声"还是"用户不买账"；
  2. ILCE-6100（2290 组，占 24%）0x940F 整块全零 —— 唯一无 EXIF 姿态的机型，
     CV 地平线是它唯一可能的水平度来源（普适、不挑相机）。

方法（梯度方向直方图，等效长直线检测）：
  - render518 是 518×518 **挤压**渲染（非等比）：先在挤压空间取每像素梯度方向，
    逐点换算回原图纵横比下的线方向角，再按 mod 90° 折到 [-45,45)（横/竖构图统一处理），
    按 |G| 加权建 0.05° 直方图，抛物线插值取亚 bin 峰 = 图像倾斜角（相对最近横/纵轴）。
  - 置信度 = 峰权占比（纹理/枝叶各向同性 → 平坦；长直线 → 尖峰）。

产出：
  - audit/out/horizon/cv_horizon.csv（fingerprint, cv_tilt_deg, conf, n_edge）
  - 交叉验证报告：已校准机型上 cv_tilt vs EXIF roll（mod 90 折回）残差统计。

用法（仓根 D:/Git/PhotoViewer）：
    Tools/.venv/Scripts/python.exe Training/audit/horizon_cv.py \
        --sample 2000 --out Training/audit/out/horizon/cv_horizon.csv
    # 只跑考卷成员：
    Tools/.venv/Scripts/python.exe Training/audit/horizon_cv.py --exam-only
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path

import numpy as np
from PIL import Image

DB = "file:D:/PhotoDB/dataset/photos_dataset.db?mode=ro"
RENDER = Path("D:/PhotoDB/dataset/render518")
EXAM_KEY = Path("D:/PhotoDB/dataset/golden_exam/golden_exam_key.csv")
ACCEL = Path("D:/PhotoDB/dataset/accel_export.csv")

BINS_DEG = 0.05
HALF = 45.0
NBINS = int(2 * HALF / BINS_DEG)


def cv_tilt(png: Path, w: int, h: int) -> tuple[float, float, int] | None:
    """返回 (倾斜角°, 峰权占比, 边缘点数)；边缘不足返回 None。"""
    img = np.asarray(Image.open(png).convert("L"), dtype=np.float32)
    # 去边 5%：边框/暗角伪边缘
    m = int(518 * 0.05)
    img = img[m:-m, m:-m]
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    mag = np.hypot(gx, gy)
    # 强边缘点：≥ P97
    thr = np.percentile(mag, 97)
    ys, xs = np.nonzero(mag >= thr)
    if len(ys) < 300:
        return None
    gxv, gyv = gx[ys, xs], gy[ys, xs]
    wts = mag[ys, xs]
    # 挤压空间的线方向 = 梯度方向 + 90°；再换算回原图纵横比
    # 挤压 (xs,ys)=(x·sx, y·sy)，方向向量 (dx,dy) 原图角 = atan2(dy·H, dx·W)
    lx, ly = -gyv, gxv  # 线方向（挤压空间）
    ang = np.degrees(np.arctan2(ly * h, lx * w))  # 原图空间线方向 ∈ (-180,180]
    # 折到最近 0/90 轴：dev ∈ [-45,45)
    dev = (ang + HALF) % 90.0 - HALF
    hist = np.zeros(NBINS)
    idx = ((dev + HALF) / BINS_DEG).astype(int) % NBINS
    np.add.at(hist, idx, wts)
    # 循环平滑（σ=1.5 bin）抗噪
    k = np.exp(-0.5 * (np.arange(-3, 4) / 1.5) ** 2)
    k /= k.sum()
    hist_s = np.convolve(np.r_[hist[-3:], hist, hist[:3]], k, mode="same")[3:-3]
    peak = int(np.argmax(hist_s))
    # 抛物线亚 bin
    y0, y1, y2 = hist_s[(peak - 1) % NBINS], hist_s[peak], hist_s[(peak + 1) % NBINS]
    denom = y0 - 2 * y1 + y2
    sub = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
    tilt = (peak + sub) * BINS_DEG - HALF
    conf = float(y1 / hist_s.sum())
    return float(tilt), conf, int(len(ys))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="Training/audit/out/horizon/cv_horizon.csv")
    ap.add_argument("--sample", type=int, default=0, help="全库随机抽样 N 张（不含考卷成员也跑）")
    ap.add_argument("--exam-only", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    con = sqlite3.connect(DB, uri=True)
    aspect = {fp: (w, h) for fp, w, h in
              con.execute("SELECT fingerprint, cv_image_width, cv_image_height FROM photos")}

    exam_fps = {r["fingerprint"] for r in csv.DictReader(open(EXAM_KEY, encoding="utf-8-sig"))}
    targets = set(exam_fps)
    if not args.exam_only and args.sample > 0:
        rng = np.random.default_rng(args.seed)
        rest = sorted(set(aspect) - exam_fps)
        targets |= set(rng.choice(rest, size=min(args.sample, len(rest)), replace=False).tolist())

    rows = []
    fps = sorted(targets)
    for i, fp in enumerate(fps):
        png = RENDER / f"{fp}.png"
        if not png.exists() or fp not in aspect:
            continue
        w, h = aspect[fp]
        r = cv_tilt(png, w, h)
        if r is None:
            rows.append((fp, "", 0.0, 0))
        else:
            rows.append((fp, f"{r[0]:.3f}", f"{r[1]:.5f}", r[2]))
        if (i + 1) % 200 == 0 or i + 1 == len(fps):
            print(f"  [{i + 1}/{len(fps)}]")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["fingerprint", "cv_tilt_deg", "conf", "n_edge"])
        wcsv.writerows(rows)
    ok = sum(1 for r in rows if r[1] != "")
    print(f"CV 地平线: {ok}/{len(rows)} 有检出 → {out}")

    # ── 交叉验证：EXIF roll（mod 90 折回 [-45,45)）vs cv_tilt ──────────────
    accel = {r["fingerprint"]: float(r["roll_deg"])
             for r in csv.DictReader(open(ACCEL, encoding="utf-8-sig")) if r["roll_deg"] != ""}
    pairs = []
    for fp, tilt, conf, ne in rows:
        if tilt == "" or fp not in accel or float(conf) < 0.002:
            continue
        dev = (accel[fp] + HALF) % 90.0 - HALF
        d = float(tilt) - dev
        d = (d + HALF) % 90.0 - HALF  # 循环差
        pairs.append((d, float(conf)))
    if pairs:
        diffs = np.array([p[0] for p in pairs])
        confs = np.array([p[1] for p in pairs])
        print(f"\n── EXIF roll vs CV tilt 交叉验证（n={len(diffs)}，conf≥0.002）──")
        print(f"  残差: mean={diffs.mean():+.3f}° median={np.median(diffs):+.3f}° "
              f"MAD={np.median(np.abs(diffs - np.median(diffs))):.3f}° "
              f"std={diffs.std():.3f}°")
        for q in (50, 68, 90, 95):
            print(f"  |残差| P{q} = {np.percentile(np.abs(diffs), q):.3f}°")
        # 高置信子集
        for cth in (0.004, 0.008, 0.016):
            m = confs >= cth
            if m.sum() >= 30:
                print(f"  conf≥{cth}: n={m.sum()} MAD={np.median(np.abs(diffs[m] - np.median(diffs[m]))):.3f}° "
                      f"P90={np.percentile(np.abs(diffs[m]), 90):.3f}°")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
