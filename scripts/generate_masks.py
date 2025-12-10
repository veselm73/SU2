"""
Generate UNet-friendly binary masks from Label Studio keypoint annotations.

Input:
- JSON export: data_annotation_sam/project-2-at-2025-12-01-15-45-88d0a5ec.json
- Images: expected under data_annotation_sam/images (as referenced in the export)

Output:
- Binary mask PNGs (single-channel) with small disks at each annotated point.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageDraw

# Paths
ROOT = Path(__file__).resolve().parent
EXPORT_JSON = ROOT / "data_annotation_sam" / "project-2-at-2025-12-01-15-45-88d0a5ec.json"
IMAGES_ROOT = ROOT / "data_annotation_sam" / "images"
OUT_DIR = ROOT / "data_annotation_sam" / "masks_unet"

# Drawing parameters
RADIUS = 3  # pixels; adjust to change dot size for the mask
FILL_VALUE = 255  # white dots on black background


def parse_image_path(image_url: str) -> Path:
    """Extract the local image path from a Label Studio /data/local-files URL."""
    query = parse_qs(urlparse(image_url).query)
    local = query.get("d", [""])[0]
    if not local:
        raise ValueError(f"Cannot parse image path from {image_url}")
    return Path(local).name  # only use the filename; assume images live under IMAGES_ROOT


def load_tasks(json_path: Path) -> list[dict]:
    data = json.loads(json_path.read_text())
    if not isinstance(data, list):
        raise ValueError("Export JSON does not contain a list of tasks")
    return data


def choose_annotation(task: dict) -> dict | None:
    anns = task.get("annotations") or []
    if not anns:
        return None
    # Use the latest annotation (last in list)
    return anns[-1]


def task_points(annotation: dict) -> tuple[int, int, list[tuple[float, float]]]:
    points: list[tuple[float, float]] = []
    width = height = None
    for res in annotation.get("result", []):
        if res.get("type") != "keypointlabels":
            continue
        width = res.get("original_width")
        height = res.get("original_height")
        val = res.get("value", {})
        x_pct = float(val.get("x", 0.0))
        y_pct = float(val.get("y", 0.0))
        if width is None or height is None:
            continue
        x = x_pct / 100.0 * width
        y = y_pct / 100.0 * height
        points.append((x, y))
    if width is None or height is None:
        raise ValueError("Missing original_width/original_height in annotation")
    return int(width), int(height), points


def draw_mask(width: int, height: int, points: list[tuple[float, float]]) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for x, y in points:
        bbox = (x - RADIUS, y - RADIUS, x + RADIUS, y + RADIUS)
        draw.ellipse(bbox, fill=FILL_VALUE)
    return mask


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(EXPORT_JSON)
    written = 0
    for task in tasks:
        image_url = task.get("data", {}).get("image", "")
        if not image_url:
            continue
        filename = parse_image_path(image_url)
        ann = choose_annotation(task)
        if ann is None:
            continue
        try:
            width, height, points = task_points(ann)
        except ValueError:
            continue
        if not points:
            continue
        mask = draw_mask(width, height, points)
        out_path = OUT_DIR / f"{Path(filename).stem}_mask.png"
        mask.save(out_path)
        written += 1
    print(f"Wrote {written} mask(s) to {OUT_DIR}")


if __name__ == "__main__":
    main()
