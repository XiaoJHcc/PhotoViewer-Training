"""只读盘点事件覆盖并生成时间均匀图片概览，不读取模型预测。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from evaluation_protocol import digest, read_rows, write_json

ROOT = Path(__file__).resolve().parents[1]


def inventory(database: Path, photos_path: Path, data: Path) -> tuple[list[dict], dict]:
    """连接数据库与照片清单，统计事件原星、标注覆盖及时间范围。"""
    rows = read_rows(photos_path)
    by_event = defaultdict(list)
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        extra = {fingerprint: (date, subject, event or "") for fingerprint, date, subject, event in
                 connection.execute("SELECT fingerprint,capture_time,subject_label,event_label FROM photos")}
    finally:
        connection.close()
    if set(extra) != {row["fingerprint"] for row in rows}:
        raise ValueError("照片清单与数据库指纹覆盖不一致")
    annotations = defaultdict(set)
    for session in ("abs_set", "m2_pool"):
        for row in read_rows(data / f"{session}_key.csv"):
            annotations[row["fingerprint"]].add(session)
    golden = defaultdict(set)
    for batch in (2, 3):
        for row in read_rows(data / f"golden_star{batch}/golden_batch{batch}_key.csv"):
            golden[row["fingerprint"]].add(row["gid"])
    for row in rows:
        date, subject, event = extra[row["fingerprint"]]
        if event != row["event"]:
            raise ValueError(f"事件身份漂移: {row['fingerprint']}")
        by_event[event].append(dict(row, capture_time=date, subject=subject))
    profiles = []
    for event, members in sorted(by_event.items()):
        dates = sorted(row["capture_time"] for row in members if row["capture_time"])
        clusters = {row["cluster_id"] for row in members}
        profiles.append({"event": event, "display": event or "20240212 旧批", "photos": len(members),
                         "original_split": members[0]["split"], "clusters": len(clusters),
                         "singleton_photos": sum(int(row["cluster_size"]) == 1 for row in members),
                         "date_start": dates[0], "date_end": dates[-1],
                         "subject_tags": dict(Counter(row["subject"] or "missing" for row in members)),
                         "ratings": dict(sorted(Counter(row["rating_raw"] for row in members).items())),
                         "abs_photos": sum(bool(annotations[row["fingerprint"]]) for row in members),
                         "golden_groups": len(set().union(*(golden[row["fingerprint"]] for row in members)))})
    return profiles, by_event


def contact_sheets(by_event: dict, renders: Path, output: Path) -> None:
    """每事件按时间抽十二个不同团的中间成员，生成四事件一页的概览。"""
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 18)
    small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 12)
    events = sorted(by_event)
    index_rows = []
    for page_start in range(0, len(events), 4):
        canvas = Image.new("RGB", (1440, 960), "#202328")
        draw = ImageDraw.Draw(canvas)
        for offset, event in enumerate(events[page_start:page_start + 4]):
            draw.text((10, offset * 240 + 2), event or "20240212 旧批", font=font, fill="white")
            clusters = defaultdict(list)
            for row in sorted(by_event[event], key=lambda row: (row["capture_time"] or "", row["fingerprint"])):
                clusters[row["cluster_id"]].append(row)
            representatives = [group[len(group) // 2] for group in clusters.values()]
            indexes = np.linspace(0, len(representatives) - 1, 12, dtype=int)
            for position, selected_index in enumerate(indexes):
                row = representatives[selected_index]
                path = renders / f"{row['fingerprint']}.png"
                with Image.open(path) as source:
                    thumbnail = ImageOps.contain(source.convert("RGB"), (118, 190))
                canvas.paste(thumbnail, (position * 120, offset * 240 + 32))
                draw.text((position * 120, offset * 240 + 223), str(position + 1), font=small, fill="white")
                index_rows.append({"page": page_start // 4 + 1, "event": event, "position": position + 1,
                                   "fingerprint": row["fingerprint"]})
        canvas.save(output / f"overview_{page_start // 4 + 1}.jpg", quality=92)
    write_json(output / "overview_index.json", index_rows)


def near_duplicates(database: Path, photos_path: Path, output: Path, threshold: float) -> None:
    """筛查不同事件中高余弦近重复候选，仅用于泄漏审计，不代表视觉真值。"""
    import torch

    torch.set_num_threads(2)
    rows = read_rows(photos_path)
    metadata = {row["fingerprint"]: row for row in rows}
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        features = {fingerprint: np.frombuffer(blob, dtype="<f4").copy() for fingerprint, blob in connection.execute(
            "SELECT fingerprint,cls_vector FROM photo_features WHERE model_id=?", ("dinov3_vits16_f32_518_v1",))}
    finally:
        connection.close()
    fingerprints = sorted(metadata)
    matrix = np.stack([features[fingerprint] for fingerprint in fingerprints])
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = torch.tensor(matrix, device=device)
    event_ids = {event: index for index, event in enumerate(sorted({row["event"] for row in rows}))}
    events = torch.tensor([event_ids[metadata[fp]["event"]] for fp in fingerprints], device=device)
    matches = []
    with torch.no_grad():
        for start in range(0, len(matrix), 512):
            similarities = tensor[start:start + 512] @ tensor.T
            different = events[start:start + 512, None] != events[None, :]
            candidates = torch.nonzero((similarities >= threshold) & different).cpu().tolist()
            for local, right in candidates:
                left = start + local
                if left >= right:
                    continue
                matches.append({"left": fingerprints[left], "right": fingerprints[right],
                                "event_left": metadata[fingerprints[left]]["event"],
                                "event_right": metadata[fingerprints[right]]["event"],
                                "cosine": float(similarities[local, right])})
    write_json(output / "cross_event_near_duplicates.json", {"threshold": threshold, "model_id": "dinov3_vits16_f32_518_v1",
               "matches": sorted(matches, key=lambda row: -row["cosine"]), "device": device,
               "scope": "高余弦只作筛查；零候选不能证明不存在较低相似的关联场景"})
    print(f"近重复筛查完成: {len(matches)} 对 >= {threshold}", flush=True)


def main() -> None:
    """读取盘点参数，输出事件报告、图片概览和可选的近重复筛查。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("D:/PhotoDB/dataset"))
    parser.add_argument("--photos", type=Path, default=ROOT / "audit/out/m3_pairs/photos.csv")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--near-duplicates", action="store_true")
    parser.add_argument("--near-threshold", type=float, default=0.995)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    database = args.data / "photos_dataset.db"
    profiles, by_event = inventory(database, args.photos, args.data)
    write_json(args.out / "inventory.json", {"events": profiles, "photos_sha256": digest(args.photos),
               "selection": "事件与时间均匀概览，不读取模型预测；题材为人工概览待确认项"})
    contact_sheets(by_event, args.data / "render518", args.out)
    if args.near_duplicates:
        near_duplicates(database, args.photos, args.out, args.near_threshold)
    print(f"盘点完成: {len(profiles)} 事件 -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
