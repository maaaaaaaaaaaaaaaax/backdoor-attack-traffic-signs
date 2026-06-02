"""Download and prepare GTSRB data from the Ultralytics NDJSON export.

Reads the NDJSON file, downloads images from CDN, crops bounding boxes,
and saves cropped sign images organized by class and split.

Output structure:
    data/gtsrb/
        train/
            00/  01/  02/ ... 42/
        val/
            00/  01/  02/ ... 42/
        test/
            00/  01/  02/ ... 42/

Usage:
    python prepare_data.py
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import requests
from config import DATA_ROOT, NDJSON_PATH
from PIL import Image
from tqdm import tqdm


def download_and_crop(entry: dict, output_root: Path) -> int:
    """Download one image and save all cropped sign regions.

    Returns number of crops saved.
    """
    split = entry["split"]
    url = entry["url"]
    width = entry["width"]
    height = entry["height"]
    boxes = entry["annotations"]["boxes"]
    filename_stem = Path(entry["file"]).stem

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except (requests.RequestException, Exception):
        return 0

    try:
        img = Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception:
        return 0

    count = 0
    for i, box in enumerate(boxes):
        cls_id = int(box[0])
        cx, cy, bw, bh = box[1], box[2], box[3], box[4]

        # Convert normalized xywh to pixel coords
        x1 = int((cx - bw / 2) * width)
        y1 = int((cy - bh / 2) * height)
        x2 = int((cx + bw / 2) * width)
        y2 = int((cy + bh / 2) * height)

        # Clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)

        if x2 - x1 < 4 or y2 - y1 < 4:
            continue

        crop = img.crop((x1, y1, x2, y2))

        # Save
        cls_dir = output_root / split / f"{cls_id:02d}"
        cls_dir.mkdir(parents=True, exist_ok=True)
        save_path = cls_dir / f"{filename_stem}_{i}.jpg"
        crop.save(save_path, quality=90)
        count += 1

    return count


def main():
    if not NDJSON_PATH.exists():
        print(f"ERROR: NDJSON file not found at {NDJSON_PATH}")
        sys.exit(1)

    # Parse all image entries
    entries = []
    with open(NDJSON_PATH) as f:
        for line in f:
            record = json.loads(line)
            if record["type"] == "image":
                entries.append(record)

    print(f"Found {len(entries)} images to process")
    print(f"Output directory: {DATA_ROOT}")
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing = sum(1 for _ in DATA_ROOT.rglob("*.jpg"))
    if existing > len(entries) * 0.9:
        print(f"Already have {existing} crops, skipping download.")
        print("Delete data/gtsrb/ to re-download.")
        return

    # Download in parallel
    total_crops = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {
            executor.submit(download_and_crop, entry, DATA_ROOT): entry
            for entry in entries
        }
        with tqdm(total=len(entries), desc="Downloading") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result == 0:
                    failed += 1
                else:
                    total_crops += result
                pbar.update(1)

    print(f"\nDone! Saved {total_crops} cropped signs.")
    if failed:
        print(f"  ({failed} images failed to download)")

    # Print class distribution
    for split in ["train", "val", "test"]:
        split_dir = DATA_ROOT / split
        if split_dir.exists():
            count = sum(1 for _ in split_dir.rglob("*.jpg"))
            n_classes = sum(1 for d in split_dir.iterdir() if d.is_dir())
            print(f"  {split}: {count} images across {n_classes} classes")


if __name__ == "__main__":
    main()
