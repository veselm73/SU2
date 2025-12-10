"""Convert SAM3 detection CSV into Label Studio task JSON with point pre-annotations."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

WORKSPACE = Path(__file__).resolve().parent
CSV_PATH = WORKSPACE / "auto_annotations.csv"
IMAGES_DIR = WORKSPACE / "images"
OUT_JSON = WORKSPACE / "ls_tasks.json"
# Root directory that Label Studio is allowed to read via /data/local-files
LOCAL_FILES_ROOT = WORKSPACE

if not CSV_PATH.exists():
    raise FileNotFoundError(f"Missing CSV at {CSV_PATH}")
if not IMAGES_DIR.exists():
    raise FileNotFoundError(f"Missing images directory at {IMAGES_DIR}")


def iter_detections(csv_path: Path) -> dict[str, list[tuple[float, float, float]]]:
    detections: dict[str, list[tuple[float, float, float]]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"filename", "x", "y"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV must contain columns {required}")
        for row in reader:
            fname = row["filename"].strip()
            try:
                x = float(row["x"])
                y = float(row["y"])
                score = float(row.get("score", 1.0) or 1.0)
            except ValueError as exc:
                raise ValueError(f"Invalid numeric value in row {row}") from exc
            detections.setdefault(fname, []).append((x, y, score))
    return detections


def image_metadata(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as img:
        return img.size  # (width, height)


def build_tasks() -> list[dict]:
    detections = iter_detections(CSV_PATH)
    tasks: list[dict] = []
    for fname, dets in sorted(detections.items()):
        image_path = IMAGES_DIR / fname
        if not image_path.exists():
            print(f"WARNING: missing image {image_path}, skipping")
            continue
        try:
            rel_path = image_path.relative_to(LOCAL_FILES_ROOT)
        except ValueError:
            print(f"WARNING: {image_path} is outside LOCAL_FILES_ROOT={LOCAL_FILES_ROOT}, skipping")
            continue
        width, height = image_metadata(image_path)
        ls_image = f"/data/local-files/?d={rel_path.as_posix()}"
        results: list[dict] = []
        for x, y, score in dets:
            results.append(
                {
                    "from_name": "kp",
                    "to_name": "image",
                    "type": "keypointlabels",
                    "value": {
                        "x": x / width * 100.0,
                        "y": y / height * 100.0,
                        "width": 0.0,
                        "keypointlabels": ["ccp"],
                    },
                    "score": score,
                }
            )
        tasks.append({"data": {"image": ls_image}, "annotations": [{"result": results}]})
    return tasks


def main() -> None:
    tasks = build_tasks()
    with OUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(tasks, handle)
    print(f"Wrote {len(tasks)} tasks to {OUT_JSON}")


if __name__ == "__main__":
    main()
