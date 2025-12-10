# U-Net++ Training and Tracking Pipeline Walkthrough

This walkthrough details the modularized codebase for training U-Net++ on Google Colab, including the new tracking parameter sweep and GIF visualization features.

## 1. Modular Structure

The codebase has been refactored into a `modules/` directory for better maintainability:

- **`modules/config.py`**: Centralized configuration (hyperparameters, paths).
- **`modules/utils.py`**: Utility functions (data download, seeding, visualization).
- **`modules/simulation.py`**: Synthetic data generation (SIM reconstruction).
- **`modules/dataset.py`**: PyTorch Dataset classes.
- **`modules/model.py`**: U-Net++ architecture definition.
- **`modules/loss.py`**: IoU Loss function.
- **`modules/training.py`**: Training loop and pipeline.
- **`modules/tracking.py`**: Tracking logic (CCPDetector, BTrack integration, HOTA metric).
- **`modules/sweep.py`**: Parameter sweep and GIF generation logic.

## 2. Colab Notebook (`Train_on_Colab.ipynb`)

The main notebook orchestrates the entire process. It is designed to be uploaded to Google Colab.

### Key Features:
- **Google Drive Mounting**: Automatically mounts Drive to save results persistently.
- **Auto-Disconnect**: Disconnects the runtime after training to save compute units.
- **Tracking Sweep**: Runs a grid search over detection and tracking parameters to find the best HOTA score.
- **GIF Visualization**: Generates and saves a GIF of the best tracking result.

### How to Run:
1. Upload the entire `SU2` folder (containing `Train_on_Colab.ipynb` and the `modules/` folder) to your Google Drive or Colab environment.
2. Open `Train_on_Colab.ipynb`.
3. Run all cells.

## 3. Key Changes & Implementations

### Dice Score Calculation
The Dice score calculation in `modules/training.py` has been corrected to use binarized ground truth masks (`mask > 0.5`), ensuring accurate binary segmentation metrics.

### Tracking Parameter Sweep
The sweep logic in `modules/sweep.py` iterates over:
- **Detection**: `min_area`, `threshold`.
- **BTrack**: `max_search_radius`, `gap_closing_max_frame_count`.

It calculates the HOTA score for each combination using the validation dataset and selects the best configuration.

### GIF Generation
The `save_tracking_gif` function creates an animation of the tracking results, showing the raw image, detection dots, and trajectory tails. This is saved as `best_tracking.gif` in the results directory.

## 4. Configuration

You can modify training parameters in `modules/config.py`:
```python
TRAIN_SAMPLES = 10000
VAL_SAMPLES = 1000
EPOCHS = 200
BATCH_SIZE = 16
...
```

## 5. Output

Results are saved to `SAVE_DIR` (default: `/content/drive/MyDrive/SU2_Project`):
- `final_model.pth`: The trained model weights.
- `best_model.pth`: The model with the lowest validation loss.
- `best_tracking.gif`: Visualization of the best tracking result.
