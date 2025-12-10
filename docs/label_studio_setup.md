# Guide: Use Label Studio to Verify SAM3 Point Detections

This document is a **concrete implementation guide** for a coding agent.  
Goal: turn an existing CSV of SAM3 detections into **point pre-annotations** in Label Studio, let a human verify/fix them, then export the corrected detections back to CSV for training.

---

## 0. Assumptions

- OS: Linux or macOS (paths use `~` but can be adapted).
- Python ≥ 3.9 is available.
- You already have:
  - A CSV of detections: `auto_annotations.csv`
  - A folder or archive containing PNG images (e.g. `frame_0000.png`, `frame_0001.png`, …).

### Expected CSV schema

`auto_annotations.csv` should have **one row per detection**:

```text
filename,frame,x,y,score
frame_0000.png,0,59.29,252.80,0.55
...
```

Only `filename,x,y` are strictly required. `score` and `frame` can be present but are optional for verification.

---

## 1. Set Up Project Directory

**Task: create a working directory with the images and CSV.**

Steps (terminal):

```bash
mkdir -p ~/ccp_data/images
# Copy or move all frame images into this folder:
cp /path/to/frames/frame_*.png ~/ccp_data/images/
# Copy CSV:
cp /path/to/auto_annotations.csv ~/ccp_data/
cd ~/ccp_data
```

From now on, all script paths assume the working directory is `~/ccp_data`.

---

## 2. Install and Start Label Studio

**Task: install Label Studio and launch it on localhost.**

Commands:

```bash
pip install label-studio
label-studio start
```

- This launches a web server at `http://localhost:8080`.
- First run will prompt creation of a username/password.
- **Keep this process running** while working with Label Studio.

> Note for agent: do not daemonize or background this process unless explicitly requested. A foreground terminal session is sufficient.

---

## 3. Create a Label Studio Project with Point Labels

**Task: configure a project that uses point annotations over images.**

1. In a browser, go to `http://localhost:8080`.
2. Create a project:
   - Click **Create Project**.
   - Name: `CCP Detection Verification` (or similar).
3. Configure labeling interface:
   - Click **Labeling Setup**.
   - Replace the default XML with the following:

   ```xml
   <View>
     <Image name="image" value="$image" zoom="true"/>
     <KeyPointLabels name="kp" toName="image">
       <Label value="ccp" background="#ff0000"/>
     </KeyPointLabels>
   </View>
   ```

4. Click **Save**.

This configuration:

- Shows the image (`$image` from task data).
- Allows adding / moving / deleting keypoints labeled `"ccp"`.

---

## 4. Convert `auto_annotations.csv` → Label Studio Tasks with Pre-Annotations

**Task: generate a JSON file (`ls_tasks.json`) that Label Studio can import.**  
Each JSON item = one task (one image) with a set of pre-annotated points.

### 4.1. Create conversion script

Create a file `csv_to_ls_tasks.py` in `~/ccp_data` with this exact content:

```python
import csv
import json
import os
from PIL import Image

CSV_PATH = "auto_annotations.csv"
IMAGES_DIR = "images"
OUT_JSON = "ls_tasks.json"

# Group detections by image filename
detections = {}
with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        fname = row["filename"]
        x = float(row["x"])
        y = float(row["y"])
        score = float(row.get("score", 1.0))
        detections.setdefault(fname, []).append((x, y, score))

tasks = []

for fname, dets in detections.items():
    img_path = os.path.join(IMAGES_DIR, fname)
    if not os.path.exists(img_path):
        print(f"WARNING: missing image {img_path}, skipping")
        continue

    # Obtain image dimensions to convert pixels → percentages
    with Image.open(img_path) as im:
        w, h = im.size

    # Label Studio local-files URL, relative to project working dir
    ls_image = f"/data/local-files/?d={img_path}"

    results = []
    for x, y, score in dets:
        results.append({
            "from_name": "kp",
            "to_name": "image",
            "type": "keypointlabels",
            "value": {
                # LS expects coordinates in percentage of width/height
                "x": x / w * 100.0,
                "y": y / h * 100.0,
                "width": 0.0,
                "keypointlabels": ["ccp"],
            },
            "score": score,
        })

    task = {
        "data": {"image": ls_image},
        "annotations": [{
            "result": results
        }]
    }
    tasks.append(task)

with open(OUT_JSON, "w") as f:
    json.dump(tasks, f)

print(f"Wrote {len(tasks)} tasks to {OUT_JSON}")
```

### 4.2. Run the converter

From `~/ccp_data`:

```bash
python csv_to_ls_tasks.py
```

Expected output:

- A file `ls_tasks.json` containing an array of Label Studio tasks.
- Warnings (if any) printed for missing images; these can be ignored or fixed by syncing names.

---

## 5. Import Tasks into Label Studio

**Task: load `ls_tasks.json` into the `CCP Detection Verification` project.**

1. In the Label Studio web UI, open the `CCP Detection Verification` project.
2. Click **Import**.
3. Choose **Upload Files** (or drag & drop) and select `ls_tasks.json`.
4. Confirm the import.

Label Studio will create:

- One task per image.
- Each task includes your SAM3 detections as pre-annotated keypoints.

---

## 6. Human Verification Workflow (Touchpad-Friendly)

**Task: allow human annotators to quickly fix the points.**

Expected behavior inside Label Studio:

1. Go to the **Labeling** view for the project.
2. Select any task (image).

Operations to support:

- **Zoom**:
  - Use the zoom slider or `Ctrl + scroll` (OS/browser dependent).
- **Pan**:
  - Hold **Space** and drag, or use two-finger drag on the touchpad.
- **Add a detection**:
  - Ensure the label `ccp` is selected in the left panel.
  - Click on the object location → a new point appears.
- **Move a detection**:
  - Click an existing point to select it.
  - Drag it to a new location.
- **Delete a false positive**:
  - Click the point to select.
  - Press `Backspace` / `Delete` or use the trash icon.

When an image is fully checked:

- Click **Submit** (or **Update** if editing an existing annotation).

> Note for coding agent: no backend changes required here. This is a manual step for a human annotator using the UI.

---

## 7. Export Verified Annotations from Label Studio

**Task: export the corrected keypoints from Label Studio to a JSON file.**

1. In the `CCP Detection Verification` project, click **Export**.
2. Choose **JSON** format.
3. Save the file as `ls_export.json` in `~/ccp_data`.

---

## 8. Convert `ls_export.json` Back to a CSV of Verified Points

**Task: generate a new CSV `verified_annotations.csv` from the exported JSON.**

### 8.1. Create export conversion script

Create `ls_export_to_csv.py` in `~/ccp_data`:

```python
import json
import csv
import os
from urllib.parse import urlparse, parse_qs
from PIL import Image

EXPORT_JSON = "ls_export.json"
OUT_CSV = "verified_annotations.csv"

with open(EXPORT_JSON) as f:
    data = json.load(f)

rows = []

for task in data:
    image_url = task["data"]["image"]
    # Example: /data/local-files/?d=images/frame_0000.png
    q = urlparse(image_url).query
    image_path = parse_qs(q).get("d", [""])[0]
    filename = os.path.basename(image_path)

    # May be multiple annotations per task; use the latest
    annotations = task.get("annotations") or []
    if not annotations:
        continue
    ann = annotations[-1]

    # Get image size for converting percent → pixels
    with Image.open(image_path) as im:
        w, h = im.size

    for res in ann.get("result", []):
        if res.get("type") != "keypointlabels":
            continue
        v = res["value"]

        x_px = v["x"] / 100.0 * w
        y_px = v["y"] / 100.0 * h

        rows.append({
            "filename": filename,
            "x": x_px,
            "y": y_px,
            "score": res.get("score", 1.0),
            "verified": 1,
        })

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["filename", "x", "y", "score", "verified"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUT_CSV}")
```

### 8.2. Run the exporter

From `~/ccp_data`:

```bash
python ls_export_to_csv.py
```

Expected output:

- A new file `verified_annotations.csv` with the schema:

  ```text
  filename,x,y,score,verified
  frame_0000.png,60.12,251.87,1.0,1
  ...
  ```

- All coordinates are in **pixels**.
- `verified=1` marks that a human checked these detections.

This CSV is now suitable as **ground truth for detector training** (e.g. heatmap generation or tiny bounding boxes).

---

## 9. Next Steps (Outside Scope of This Guide)

Once `verified_annotations.csv` exists, a separate pipeline should:

- Load images + verified points.
- Convert points into training targets:
  - Gaussian heatmaps, **or**
  - fixed-size bounding boxes.
- Train a detection model using these refined labels.

These steps are left to another guide; this document focuses exclusively on the Label Studio verification loop.
