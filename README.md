# SU2 - Cell Detection and Tracking with StarDist

Deep learning pipeline for cell detection and tracking in microscopy images. Uses **StarDist** with a ResNet encoder for robust cell detection and **BTrack/LapTrack** for temporal tracking.


## Features

- **StarDist Detection**: Star-convex polygon detection with configurable backbone (ResNet18/34/50)
- **Improved Training Pipeline**:
  - Combined loss function (Focal + Dice + Smooth L1)
  - Data augmentation (rotation, flips, noise, elastic deformation)
  - Weight decay regularization + LR scheduling
  - Early stopping to prevent overfitting
  - Automatic threshold optimization (prob_thresh, nms_thresh)
- **K-Fold Cross-Validation**: 5-fold CV with per-fold evaluation
- **Cell Tracking**: BTrack and LapTrack with HOTA metric evaluation
- **Included Training Data**: 42 annotated frames with masks from SAM

## Project Structure

```
SU2/
├── notebooks/
│   ├── SU2_StarDist_final.ipynb  # Main notebook (best results)
│   ├── Train_on_Colab.ipynb      # Alternative U-Net training
│   └── kfold/                    # K-fold variants
│
├── modules/
│   ├── stardist_helpers.py       # StarDist training pipeline
│   ├── model.py                  # U-Net++ architecture
│   ├── tracking.py               # BTrack, HOTA metrics
│   └── ...
│
├── annotation/
│   └── sam_data/unet_train/      # Training data (images + masks)
│
├── data/val/                     # Validation video (downloaded automatically)
├── configs/                      # Training configurations
└── docs/                         # Documentation
```

## Installation

### Local Setup

```bash
git clone https://github.com/veselm73/SU2.git
cd SU2
pip install -r requirements.txt
```

### Dependencies

- PyTorch with CUDA
- cellseg-models-pytorch
- pytorch-lightning
- albumentations
- btrack, laptrack

## Training Configuration

Edit the configuration cell in `SU2_StarDist_final.ipynb`:

```python
# Basic parameters
K_SPLITS = 5              # Cross-validation folds
EPOCHS = 50               # Max epochs (early stopping enabled)
BATCH_SIZE = 4            # Reduce if OOM
LR = 1e-4                 # Learning rate

# Model
N_RAYS = 32               # StarDist rays (32, 64, or 96)
ENCODER_NAME = "resnet18" # Backbone: resnet18/34/50, efficientnet-b0

# Improvements (set to False/0 for baseline comparison)
USE_AUGMENTATION = True   # Data augmentation
WEIGHT_DECAY = 1e-4       # L2 regularization
```

## Metrics

- **DetA (Detection Accuracy)**: Hungarian matching with distance threshold
- **HOTA (Higher Order Tracking Accuracy)**: Combined detection + association metric

## Results

The pipeline outputs:
- `best_stardist_model.pth` - Best fold model weights
- `stardist_predictions.csv` - Detection coordinates per frame
- `stardist_tracked.csv` - Tracked detections with track IDs
- `training_curves.png` - Loss curves and LR schedule
- `training_summary.csv` - Final metrics summary

## Other Notebooks

| Notebook | Description |
|----------|-------------|
| `SU2_StarDist_final.ipynb` | **Recommended** - StarDist with all improvements |
| `Train_on_Colab.ipynb` | U-Net++ training pipeline |
| `kfold/SU2_kfold_Stardist.ipynb` | StarDist K-fold variant |
| `kfold/SU2_Unet_kfold.ipynb` | U-Net K-fold variant |

## Documentation

- [Pipeline Walkthrough](docs/walkthrough.md)
- [Colab Guide](docs/colab_guide.md)
- [Label Studio Setup](docs/label_studio_setup.md)

## Acknowledgments

- [StarDist](https://github.com/stardist/stardist) - Cell detection
- [cellseg-models-pytorch](https://github.com/okunator/cellseg_models.pytorch) - PyTorch StarDist implementation
- [BTrack](https://github.com/quantumjot/btrack) - Bayesian cell tracking
- [SAM](https://github.com/facebookresearch/segment-anything) - Annotation assistance
