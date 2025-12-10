# Google Colab Quick Start Guide

This guide explains how to run the SU2 cell segmentation pipeline on Google Colab.

## Prerequisites

- Google account
- Google Drive with sufficient storage (~1GB recommended)

## Setup Steps

### 1. Upload to Google Drive

Upload the entire `SU2` folder to your Google Drive. The recommended location is:
```
My Drive/SU2/
```

### 2. Open the Training Notebook

1. Navigate to `SU2/notebooks/Train_on_Colab.ipynb` in Google Drive
2. Right-click and select "Open with > Google Colaboratory"

### 3. Enable GPU Runtime

1. Go to `Runtime > Change runtime type`
2. Select `GPU` as the Hardware accelerator
3. Click `Save`

### 4. Run the Notebook

Execute all cells in order. The notebook will:
1. Mount your Google Drive
2. Install required dependencies
3. Import the modules
4. Generate synthetic training data
5. Train the U-Net++ model
6. Run tracking parameter sweep
7. Save results to Google Drive
8. Auto-disconnect to save compute units

## Key Features

### Automatic Drive Mounting
The notebook automatically mounts Google Drive to persist results:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Auto-Disconnect
After training completes, the notebook disconnects the runtime to conserve Colab compute units.

### Results Location
Results are saved to:
```
/content/drive/MyDrive/SU2_Project/
├── final_model.pth      # Final trained model
├── best_model.pth       # Best validation loss model
└── best_tracking.gif    # Tracking visualization
```

## Customization

### Training Parameters

Modify `modules/config.py` before uploading, or edit in Colab:
```python
TRAIN_SAMPLES = 10000  # Number of training samples
VAL_SAMPLES = 1000     # Number of validation samples
EPOCHS = 200           # Training epochs
BATCH_SIZE = 16        # Batch size
```

### Using Different Notebooks

- **Standard training**: `notebooks/Train_on_Colab.ipynb`
- **Full pipeline**: `notebooks/SU2_pipeline.ipynb`
- **K-fold validation**: `notebooks/kfold/SU2_Unet_kfold.ipynb`

## Troubleshooting

### Out of Memory (OOM)
Reduce batch size or patch size:
```python
BATCH_SIZE = 8  # Reduce from 16
PATCH_SIZE = 64  # Reduce from 128
```

### Session Disconnects
Colab sessions timeout after idle periods. Options:
1. Use Colab Pro for longer sessions
2. Run `scripts/train_overnight.py` on a local machine

### Module Import Errors
Ensure the `modules/` folder is in the correct location relative to the notebook.

## Colab Pro Benefits

With Colab Pro:
- Longer runtime limits
- Priority access to GPUs
- More RAM available
- Background execution

## Tips

1. **Save checkpoints frequently**: The notebook saves checkpoints to Drive
2. **Monitor GPU usage**: Use `!nvidia-smi` to check GPU utilization
3. **Clear outputs**: Large outputs can slow down the notebook
4. **Use smaller datasets first**: Test with fewer samples before full training

## Related Documentation

- [Pipeline Walkthrough](walkthrough.md)
- [Synthetic Data Guide](synthetic_data_guide.md)
