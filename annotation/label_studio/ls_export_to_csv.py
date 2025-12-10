"""Convert Label Studio export JSON back into pixel-space CSV annotations."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image

WORKSPACE = Path(__file__).resolve().parent
EXPORT_JSON = WORKSPACE / "ls_export.json"
OUT_CSV = WORKSPACE / "verified_annotations.csv"

if not EXPORT_JSON.exists():
    print(f"Export file {EXPORT_JSON} not found; create it via Label Studio export first.")


def to_pixel_coordinates(value: dict, image_path: Path) -> tuple[float, float]:
    with Image.open(image_path) as img:
        width, height = img.size
    return value["x"] / 100.0 * width, value["y"] / 100.0 * height


def resolve_image_path(task: dict) -> Path:
    image_url = task["data"].get("image", "")
    query = parse_qs(urlparse(image_url).query)
    local_path = query.get("d", [""])[0]
    if not local_path:
        raise ValueError(f"Task missing local file path: {task}")
    return Path(local_path)


def load_latest_annotation(task: dict) -> dict | None:
    annotations = task.get("annotations") or []
    return annotations[-1] if annotations else None


def convert() -> list[dict[str, float]]:
    with EXPORT_JSON.open(encoding="utf-8") as handle:
        data = json.load(handle)

    rows: list[dict[str, float]] = []
    for task in data:
        latest = load_latest_annotation(task)
        if latest is None:
            continue
        image_path = resolve_image_path(task)
        filename = image_path.name
        for result in latest.get("result", []):
            if result.get("type") != "keypointlabels":
                continue
            value = result["value"]
            x_px, y_px = to_pixel_coordinates(value, image_path)
            rows.append(
                {
                    "filename": filename,
                    "x": round(x_px, 4),
                    "y": round(y_px, 4),
                    "score": result.get("score", 1.0),
                    "verified": 1,
                }
            )
    return rows


def main() -> None:
    if not EXPORT_JSON.exists():
        raise FileNotFoundError(f"Missing Label Studio export JSON at {EXPORT_JSON}")
    rows = convert()
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "x", "y", "score", "verified"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
