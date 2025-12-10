# Label Studio CCP Verification Quickstart

Workspace: `label_studio_workspace`

1) Start server (keep terminal open):
   ```powershell
   cd label_studio_workspace
   label-studio start
   ```
   Open http://localhost:8080 and log in.

2) Create project:
   - Click **Create Project** › name it (e.g., "CCP Detection Verification").
   - Go to **Labeling Setup** › **Code** tab › replace contents with `label_config.xml` from this folder:
     ```xml
     <View>
       <Image name="image" value="$image" zoom="true"/>
       <KeyPointLabels name="kp" toName="image">
         <Label value="ccp" background="#ff0000"/>
       </KeyPointLabels>
     </View>
     ```
   - Save.

3) Import pre-annotations (SAM3 priors):
   - In the project, click **Import** › upload `ls_tasks.json` from this folder.
   - Each frame now shows SAM3 detections as keypoints.

4) Annotate:
   - Open tasks in **Labeling**; use touchpad zoom/pan, click to add/move points, delete to remove.
   - Submit/Update each image after verifying.

5) Export and convert back to CSV:
   - In project, click **Export** › **JSON** › save as `ls_export.json` in this folder.
   - In terminal (still in `label_studio_workspace`):
     ```powershell
     python ls_export_to_csv.py
     ```
   - Output: `verified_annotations.csv` with pixel coordinates and `verified=1` flag.

Notes:
- If you refresh SAM3 priors, regenerate tasks with `python csv_to_ls_tasks.py` then re-import.
- Images and auto priors live under `images/` and `auto_annotations.csv` respectively.
