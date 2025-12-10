# SU2 - Cell Segmentation and Tracking Pipeline

A deep learning pipeline for cell segmentation and tracking in microscopy images using U-Net++ and StarDist models with SIM (Structured Illumination Microscopy) reconstruction.

## Features

- **U-Net++ Segmentation**: Deep learning model for cell segmentation
- **StarDist Integration**: Star-convex polygon detection for cell nuclei
- **Cell Tracking**: BTrack and LapTrack integration for temporal cell tracking
- **Synthetic Data Generation**: SIM-based synthetic training data generation
- **SAM3 Annotation**: Segment Anything Model integration for semi-automatic annotation
- **K-Fold Cross-Validation**: Multiple notebook variants for robust model evaluation
- **Google Colab Support**: Ready-to-use notebooks for cloud training

## Project Structure

```
SU2/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore
│
├── docs/                     # Documentation
│   ├── walkthrough.md        # Detailed pipeline walkthrough
│   ├── synthetic_data_guide.md  # Synthetic data generation guide
│   ├── label_studio_setup.md    # Label Studio annotation setup
│   └── colab_guide.md        # Google Colab quick start
│
├── notebooks/                # Jupyter notebooks
│   ├── Train_on_Colab.ipynb  # Main Colab training notebook
│   ├── SU2_pipeline.ipynb    # Main pipeline notebook
│   ├── SU2_Unet_Stardist_GPU.ipynb  # GPU training with U-Net + StarDist
│   ├── kfold/                # K-fold cross-validation variants
│   │   ├── SU2_kfold_Stardist.ipynb
│   │   ├── SU2_Unet_kfold.ipynb
│   │   ├── SU2_Unet_kfold_Stardist.ipynb
│   │   └── Unet_kfold.ipynb
│   └── legacy/               # Archived notebooks
│
├── modules/                  # Core Python modules
│   ├── config.py             # Configuration and hyperparameters
│   ├── dataset.py            # PyTorch Dataset classes
│   ├── model.py              # U-Net++ architecture
│   ├── loss.py               # Loss functions (IoU, Dice)
│   ├── training.py           # Training loop
│   ├── simulation.py         # SIM reconstruction simulation
│   ├── tracking.py           # Cell tracking (BTrack, HOTA metrics)
│   ├── sweep.py              # Parameter sweep utilities
│   ├── sam_detector.py       # SAM integration
│   └── utils.py              # Utility functions
│
├── scripts/                  # Utility scripts
│   ├── train_overnight.py    # Long training runs
│   ├── download_data.py      # Data download utilities
│   ├── setup_sam3.py         # SAM3 setup script
│   └── ...
│
├── configs/                  # Configuration files
│   ├── overnight.yaml        # Overnight training config
│   ├── 8hour.yaml            # 8-hour training config
│   └── synthetic_data.yaml   # Synthetic data parameters
│
├── data/                     # Data directory
│   └── val/                  # Validation data
│
├── annotation/               # Annotation tools and data
│   ├── annotate.ipynb        # Annotation notebook
│   ├── sam_data/             # SAM annotation outputs
│   └── label_studio/         # Label Studio configuration
│
├── models/                   # Trained models
│   └── stardist/             # StarDist model checkpoints
│
└── sam3/                     # SAM3 submodule
```

## Installation

### Local Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd SU2
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Set up SAM3 for annotation:
```bash
python scripts/setup_sam3.py
```

### Google Colab

See [docs/colab_guide.md](docs/colab_guide.md) for Colab-specific instructions.

## Quick Start

### Training on Colab (Recommended)

1. Upload the `SU2` folder to Google Drive
2. Open `notebooks/Train_on_Colab.ipynb` in Colab
3. Run all cells

### Local Training

```bash
python scripts/train_overnight.py --config configs/overnight.yaml
```

### Running the Pipeline

Open `notebooks/SU2_pipeline.ipynb` in Jupyter and follow the step-by-step instructions.

## Configuration

Training parameters can be modified in:
- `modules/config.py` - Python configuration
- `configs/*.yaml` - YAML configuration files

Key parameters:
- `TRAIN_SAMPLES`: Number of synthetic training samples
- `VAL_SAMPLES`: Number of validation samples
- `EPOCHS`: Training epochs
- `BATCH_SIZE`: Batch size for training
- `PATCH_SIZE`: Image patch size (default: 128x128)

## Documentation

- [Pipeline Walkthrough](docs/walkthrough.md) - Detailed explanation of the training pipeline
- [Synthetic Data Guide](docs/synthetic_data_guide.md) - How to configure synthetic data generation
- [Label Studio Setup](docs/label_studio_setup.md) - Setting up annotation with Label Studio and SAM3
- [Colab Guide](docs/colab_guide.md) - Running on Google Colab

## Models

The pipeline supports:
- **U-Net++**: Nested U-Net architecture for semantic segmentation
- **StarDist**: Star-convex polygon detection for cell nuclei

## Tracking

Cell tracking is implemented using:
- **BTrack**: Bayesian cell tracking
- **LapTrack**: Linear assignment problem-based tracking
- **HOTA Metric**: Higher Order Tracking Accuracy for evaluation

## License

[Add your license here]

## Acknowledgments

- U-Net++ architecture
- StarDist for cell detection
- SAM (Segment Anything Model) for annotation assistance
- BTrack for cell tracking
