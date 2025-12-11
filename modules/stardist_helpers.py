"""
StarDist Training Helpers for SU2 Pipeline

This module contains all helper functions and classes for:
- Data downloading and preparation
- UNet model architecture
- StarDist Lightning module
- Training utilities
- Metrics (DetA, HOTA)
- Tracking (LapTrack, BTrack)
- Visualization helpers
"""

import os
import copy
import json
import math
import shutil
import random
import zipfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from sklearn.model_selection import KFold
from skimage.measure import label, regionprops
from skimage.draw import disk
from skimage.feature import peak_local_max
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter
from scipy import spatial, optimize
from IPython.display import HTML, display

# Optional imports
try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, Callback
    HAS_LIGHTNING = True
except ImportError:
    HAS_LIGHTNING = False
    pl = None

try:
    from laptrack import LapTrack
    HAS_LAPTRACK = True
except ImportError:
    LapTrack = None
    HAS_LAPTRACK = False

try:
    import btrack
    HAS_BTRACK = True
except ImportError:
    btrack = None
    HAS_BTRACK = False

try:
    from cellseg_models_pytorch.models.stardist.stardist import StarDist
    from cellseg_models_pytorch.transforms.functional.stardist import gen_stardist_maps
    from cellseg_models_pytorch.postproc.functional.stardist.stardist import post_proc_stardist
    HAS_STARDIST_TORCH = True
except ImportError:
    HAS_STARDIST_TORCH = False
    StarDist = None

# =============================================================================
# CONSTANTS
# =============================================================================

ROI_Y_MIN, ROI_Y_MAX = 512, 768
ROI_X_MIN, ROI_X_MAX = 256, 512
ROI_H, ROI_W = ROI_Y_MAX - ROI_Y_MIN, ROI_X_MAX - ROI_X_MIN

# =============================================================================
# DATA DOWNLOAD HELPERS
# =============================================================================

def download_validation_data(
    target_dir: str = "val_data",
    url: str = "https://su2.utia.cas.cz/files/labs/final2025/val_and_sota.zip",
    cert_url: str = "https://pki.cesnet.cz/_media/certs/chain-harica-rsa-ov-crosssigned-root.pem"
):
    """Download validation video and labels zip file."""
    import requests

    if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
        print(f"'{target_dir}' already exists. Skipping download.")
        return

    chain_path = "chain-harica-cross.pem"
    print("1) Downloading SSL certificate chain...")
    r = requests.get(cert_url, timeout=10, stream=True)
    r.raise_for_status()
    with open(chain_path, "wb") as f:
        f.write(r.content)

    print("2) Downloading validation archive...")
    zip_name = os.path.basename(url)
    with requests.get(url, stream=True, verify=chain_path, timeout=30) as resp:
        resp.raise_for_status()
        with open(zip_name, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print("3) Extracting...")
    os.makedirs(target_dir, exist_ok=True)
    with zipfile.ZipFile(zip_name, "r") as zf:
        zf.extractall(target_dir)
    os.remove(zip_name)
    print(f"Done. Data in '{target_dir}/'")


def fetch_training_data_from_github(
    repo_url: str = "https://github.com/Mateuszq28/SU2",
    branch: str = "main",
    data_subpath: str = "annotation/sam_data/unet_train",
    target_dir: str = "bonus_training_data",
    use_local_if_available: bool = True
):
    """
    Fetch annotated bonus training data from GitHub repository.

    This function downloads the training data (images + masks) from the published
    GitHub repository. It supports both direct download and local copy if running
    from within the cloned repo.

    Args:
        repo_url: GitHub repository URL
        branch: Git branch to fetch from
        data_subpath: Path within repo to training data folder
        target_dir: Local directory to save/copy data to
        use_local_if_available: If True, copy from local repo instead of downloading

    Returns:
        Path to the training data directory
    """
    import requests

    target_path = Path(target_dir)

    # Check if target already exists
    if target_path.exists() and (target_path / "images").exists():
        n_images = len(list((target_path / "images").glob("*")))
        if n_images > 0:
            print(f"Training data already exists at '{target_dir}' ({n_images} images). Skipping.")
            return str(target_path)

    # Try to find local data first (if running from cloned repo)
    if use_local_if_available:
        # Search for local data in common locations
        possible_local_paths = [
            Path(data_subpath),  # Relative to CWD
            Path(__file__).parent.parent / data_subpath,  # Relative to module
            Path.cwd().parent / data_subpath,  # Parent of CWD
        ]

        for local_path in possible_local_paths:
            if local_path.exists() and (local_path / "images").exists():
                n_local = len(list((local_path / "images").glob("*")))
                if n_local > 0:
                    print(f"Found local training data at '{local_path}' ({n_local} images)")
                    if target_path != local_path:
                        print(f"Copying to '{target_dir}'...")
                        target_path.mkdir(parents=True, exist_ok=True)
                        shutil.copytree(local_path / "images", target_path / "images")
                        shutil.copytree(local_path / "masks", target_path / "masks")
                        print("Copy complete.")
                    return str(target_path)

    # Download from GitHub
    print(f"Downloading training data from GitHub ({repo_url})...")

    # GitHub raw content URL pattern
    raw_base = repo_url.replace("github.com", "raw.githubusercontent.com")
    raw_base = f"{raw_base}/{branch}/{data_subpath}"

    # First, get the file listing via GitHub API
    api_url = repo_url.replace("github.com", "api.github.com/repos")
    contents_url = f"{api_url}/contents/{data_subpath}?ref={branch}"

    try:
        # Get directory contents
        resp = requests.get(contents_url, timeout=30)
        resp.raise_for_status()
        contents = resp.json()

        # Create target directories
        target_path.mkdir(parents=True, exist_ok=True)
        (target_path / "images").mkdir(exist_ok=True)
        (target_path / "masks").mkdir(exist_ok=True)

        # Find and download images and masks subdirectories
        for item in contents:
            if item['type'] == 'dir' and item['name'] in ['images', 'masks']:
                subdir_url = f"{api_url}/contents/{data_subpath}/{item['name']}?ref={branch}"
                subdir_resp = requests.get(subdir_url, timeout=30)
                subdir_resp.raise_for_status()
                files = subdir_resp.json()

                print(f"Downloading {len(files)} files from {item['name']}/...")
                for f in files:
                    if f['type'] == 'file':
                        file_resp = requests.get(f['download_url'], timeout=30)
                        file_resp.raise_for_status()
                        save_path = target_path / item['name'] / f['name']
                        with open(save_path, 'wb') as fp:
                            fp.write(file_resp.content)

        n_downloaded = len(list((target_path / "images").glob("*")))
        print(f"Download complete. {n_downloaded} images saved to '{target_dir}'")
        return str(target_path)

    except requests.exceptions.RequestException as e:
        print(f"Failed to download from GitHub: {e}")
        print("Please clone the repository manually or check your internet connection.")
        return None


def fetch_from_drive(
    drive_source_path: str = "/content/drive/MyDrive/unet_train",
    target_dir: str = "real_training_data"
):
    """
    [DEPRECATED] Copy training data from Google Drive (for Colab).

    Note: Use fetch_training_data_from_github() instead for published data.
    This function is kept for backwards compatibility.
    """
    print("WARNING: fetch_from_drive is deprecated. Use fetch_training_data_from_github() instead.")

    try:
        from google.colab import drive
    except ImportError:
        print("google.colab not available (not running in Colab).")
        return

    print("Mounting Drive...")
    drive.mount('/content/drive')

    if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
        print(f"'{target_dir}' already exists and is not empty. Skipping copy.")
        return

    if not os.path.exists(drive_source_path):
        print(f"Source not found: {drive_source_path}")
        return

    print(f"Copying {drive_source_path} -> {target_dir}")
    try:
        shutil.copytree(drive_source_path, target_dir)
        print("Copy complete.")
    except Exception as e:
        print(f"Copy failed: {e}")
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class ConvBlock(nn.Module):
    """Double convolution block for UNet."""
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        if self.dropout:
            x = self.dropout(x)
        return x


class LightweightUNet(nn.Module):
    """Lightweight UNet architecture for cell segmentation."""
    def __init__(self, in_channels=1, n_classes=1, features=(32, 64, 128, 256), dropout_rate=0.2):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        in_ch = in_channels
        for idx, feature in enumerate(features):
            current_dropout = dropout_rate if idx == len(features) - 1 else 0.0
            self.downs.append(ConvBlock(in_ch, feature, dropout_rate=current_dropout))
            in_ch = feature

        self.bottleneck = ConvBlock(features[-1], features[-1] * 2, dropout_rate=dropout_rate)

        for idx, feature in enumerate(reversed(features)):
            current_dropout = dropout_rate if idx == 0 else 0.0
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(ConvBlock(feature * 2, feature, dropout_rate=current_dropout))

        self.final_conv = nn.Conv2d(features[0], n_classes, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)
            x = self.ups[idx + 1](torch.cat((skip_connection, x), dim=1))

        return self.final_conv(x)


def build_unet(in_channels: int = 1, n_classes: int = 1, features=(32, 64, 128, 256), dropout_rate: float = 0.2):
    """Build a UNet model."""
    return LightweightUNet(in_channels=in_channels, n_classes=n_classes, features=features, dropout_rate=dropout_rate)


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class ComboLoss(nn.Module):
    """Combined Focal + Dice Loss."""
    def __init__(self, focal_weight: float = 0.5, dice_weight: float = 0.5, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal = FocalLoss(alpha=alpha, gamma=gamma) if focal_weight > 0 else None

    def dice_loss(self, pred, target, eps=1e-6):
        pred = torch.sigmoid(pred)
        smooth = 1.0
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth + eps)
        return 1 - dice

    def forward(self, inputs, targets):
        if isinstance(inputs, list):
            losses = [self.forward(item, targets) for item in inputs]
            return sum(losses) / len(losses)

        loss = 0.0
        denom = 0.0
        if self.focal and self.focal_weight > 0:
            loss += self.focal_weight * self.focal(inputs, targets)
            denom += self.focal_weight
        if self.dice_weight > 0:
            loss += self.dice_weight * self.dice_loss(inputs, targets)
            denom += self.dice_weight
        if denom == 0:
            raise ValueError("At least one of focal_weight or dice_weight must be > 0.")
        return loss / denom


# =============================================================================
# DATA PREPARATION
# =============================================================================

def create_stardist_label_mask(image_shape: Tuple[int, int], points, radius: int = 6) -> np.ndarray:
    """Convert center points into a StarDist-ready integer mask."""
    label_mask = np.zeros(image_shape, dtype=np.uint16)
    current_id = 1
    for (y, x) in points:
        if y < 0 or x < 0 or y >= image_shape[0] or x >= image_shape[1]:
            continue
        rr, cc = disk((y, x), radius, shape=image_shape)
        label_mask[rr, cc] = current_id
        current_id += 1
    return label_mask


def _label_connected(mask: np.ndarray) -> np.ndarray:
    """Label connected components in a mask."""
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return label(mask > 0).astype(np.uint16)


def _load_gray_image(path: Path) -> np.ndarray:
    """Load an image as grayscale."""
    if path.suffix.lower() in {'.tif', '.tiff'}:
        arr = tifffile.imread(path)
    else:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(f"Missing image: {path}")
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return arr


def _load_mask_image(path: Path) -> np.ndarray:
    """Load a mask image."""
    if path.suffix.lower() in {'.tif', '.tiff'}:
        mask = tifffile.imread(path)
    else:
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Missing mask: {path}")
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask


def prepare_grand_dataset(
    bonus_data_dir: str = None,
    val_tif_path: str = "val_data/val.tif",
    val_csv_path: str = "val_data/val.csv",
    out_dir: str = "experiment_dataset",
    roi_y: Tuple[int, int] = (ROI_Y_MIN, ROI_Y_MAX),
    roi_x: Tuple[int, int] = (ROI_X_MIN, ROI_X_MAX),
    disk_radius: int = 6,
    # Legacy parameter name for backwards compatibility
    real_data_dir: str = None,
) -> Path:
    """
    Prepare the combined dataset for training.

    Creates experiment_dataset/ with:
    - bonus/images/, bonus/masks/: Extra annotated training images
    - video/images/, video/masks/: Frames from validation video with generated masks
    - video_map.csv: Mapping from video frame filenames to frame indices

    Args:
        bonus_data_dir: Directory with bonus training data (images/ and masks/ subdirs)
        val_tif_path: Path to validation video TIF file
        val_csv_path: Path to validation coordinates CSV
        out_dir: Output directory for prepared dataset
        roi_y: (y_min, y_max) ROI crop bounds
        roi_x: (x_min, x_max) ROI crop bounds
        disk_radius: Radius for disk masks in video frames
        real_data_dir: [DEPRECATED] Use bonus_data_dir instead

    Returns:
        Path to output directory
    """
    # Handle legacy parameter name
    if bonus_data_dir is None and real_data_dir is not None:
        bonus_data_dir = real_data_dir
    elif bonus_data_dir is None:
        bonus_data_dir = "bonus_training_data"

    out_path = Path(out_dir)
    bonus_images = out_path / "bonus" / "images"
    bonus_masks = out_path / "bonus" / "masks"
    video_images = out_path / "video" / "images"
    video_masks = out_path / "video" / "masks"

    if out_path.exists():
        shutil.rmtree(out_path)
    for p in [bonus_images, bonus_masks, video_images, video_masks]:
        p.mkdir(parents=True, exist_ok=True)

    # Process bonus training data if available
    src_images = Path(bonus_data_dir) / "images"
    src_masks = Path(bonus_data_dir) / "masks"

    if src_images.exists() and src_masks.exists():
        # Build mask lookup with flexible naming support
        # Supports: {stem}_mask.ext, {stem}.ext, mask_{stem}.ext
        mask_lookup = {}
        for m in src_masks.glob("*"):
            mask_lookup[m.stem] = m
            # Also index without _mask suffix for matching
            if m.stem.endswith("_mask"):
                mask_lookup[m.stem[:-5]] = m

        bonus_count = 0
        for img_path in sorted(src_images.glob("*")):
            # Try different mask naming conventions
            mask_path = None
            for pattern in [f"{img_path.stem}_mask", img_path.stem, f"mask_{img_path.stem}"]:
                if pattern in mask_lookup:
                    mask_path = mask_lookup[pattern]
                    break

            if mask_path is None:
                print(f"Warning: No mask found for {img_path.name}, skipping")
                continue

            # Copy image
            shutil.copy(img_path, bonus_images / img_path.name)

            # Process mask - convert to instance labels
            bonus_mask_raw = _load_mask_image(mask_path)
            instance_mask = _label_connected(bonus_mask_raw)
            tifffile.imwrite(bonus_masks / f"{img_path.stem}.tif", instance_mask)
            bonus_count += 1

        print(f"Processed {bonus_count} bonus training samples")
    else:
        print(f"No bonus data found at {bonus_data_dir}, skipping bonus samples")

    if not Path(val_tif_path).exists():
        raise FileNotFoundError(f"Missing video file: {val_tif_path}")
    if not Path(val_csv_path).exists():
        raise FileNotFoundError(f"Missing CSV file: {val_csv_path}")

    video = tifffile.imread(val_tif_path)
    coords = pd.read_csv(val_csv_path)
    y_min, y_max = roi_y
    x_min, x_max = roi_x
    records: List[dict] = []

    for idx, frame in enumerate(video):
        crop = frame[y_min:y_max, x_min:x_max]
        points = [(int(row['y'] - y_min), int(row['x'] - x_min))
                  for _, row in coords[coords['frame'] == idx].iterrows()]
        mask = create_stardist_label_mask((ROI_H, ROI_W), points, radius=disk_radius)

        video_images_idx_path = video_images / f"frame_{idx:03d}.png"
        video_masks_idx_path = video_masks / f"frame_{idx:03d}.tif"
        cv2.imwrite(str(video_images_idx_path), crop)
        tifffile.imwrite(str(video_masks_idx_path), mask)
        records.append({'filename': video_images_idx_path.name, 'real_frame_idx': idx})

    video_map_df = pd.DataFrame(records)
    video_map_df.to_csv(out_path / "video_map.csv", index=False)

    print(f"Dataset written to {out_path}")
    print(f"Bonus samples: {len(list(bonus_images.glob('*')))} | Video frames: {len(records)}")
    return out_path


# =============================================================================
# AUGMENTATION & DATASET
# =============================================================================

def get_train_transforms(
    rotate_p: float = 0.7,
    hflip_p: float = 0.5,
    vflip_p: float = 0.5,
    clahe_p: float = 0.5,
    brightness_p: float = 0.5,
    gauss_p: float = 0.3,
    elastic_p: float = 0.2,
    coarse_p: float = 0.5,
    coarse_max_holes: int = 16,
    coarse_min_holes: int = 8,
    coarse_max_hw: int = 16,
    coarse_min_hw: int = 8,
    crop_scale_min: float = 0.8,
    crop_scale_max: float = 1.0,
    crop_ratio_min: float = 0.9,
    crop_ratio_max: float = 1.1,
):
    """Get training augmentation transforms."""
    return A.Compose([
        A.Rotate(limit=180, p=rotate_p),
        A.HorizontalFlip(p=hflip_p),
        A.VerticalFlip(p=vflip_p),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=clahe_p),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=brightness_p),
        A.GaussNoise(var_limit=(10.0, 50.0), p=gauss_p),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=elastic_p),
        A.CoarseDropout(
            max_holes=coarse_max_holes, max_height=coarse_max_hw, max_width=coarse_max_hw,
            min_holes=coarse_min_holes, min_height=coarse_min_hw, min_width=coarse_min_hw,
            fill_value=0, mask_fill_value=0, p=coarse_p,
        ),
        A.RandomResizedCrop(height=256, width=256, scale=(crop_scale_min, crop_scale_max),
                           ratio=(crop_ratio_min, crop_ratio_max), p=0.5),
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_CONSTANT, value=0),
        ToTensorV2(),
    ])


def get_val_transforms():
    """Get validation transforms (minimal)."""
    return A.Compose([ToTensorV2()])


class AugmentedMicroscopyDataset(Dataset):
    """Dataset for microscopy images with masks."""
    def __init__(self, root_dir: str, transform=None, return_meta: bool = False, mask_mode: str = "binary"):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.return_meta = return_meta
        self.mask_mode = mask_mode
        self.image_paths = sorted(self.root_dir.joinpath("images").glob("*"))

        mask_candidates = {}
        masks_root = self.root_dir.joinpath("masks")
        for ext in ("*.tif", "*.tiff", "*.png"):
            for p in masks_root.glob(ext):
                mask_candidates[p.stem] = p

        self.mask_paths = []
        for img_path in self.image_paths:
            mask_path = mask_candidates.get(img_path.stem)
            if mask_path is None:
                raise FileNotFoundError(f"No mask found for {img_path.name}")
            self.mask_paths.append(mask_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        img = _load_gray_image(img_path)
        mask = _load_mask_image(mask_path)

        if self.mask_mode == "binary":
            mask = (mask > 0).astype(np.uint8) * 255
        elif self.mask_mode == "label":
            mask = mask.astype(np.uint16)
        else:
            mask = mask.astype(np.float32)

        if self.transform:
            augmented = self.transform(image=img.astype(np.uint8), mask=mask.astype(mask.dtype))
            img, mask = augmented["image"], augmented["mask"]
            if isinstance(img, torch.Tensor):
                img = img.float()
                if img.max() > 1:
                    img = img / 255.0
            if isinstance(mask, torch.Tensor):
                if self.mask_mode == "label":
                    mask = mask.long()
                else:
                    mask = mask.float()
                    if mask.max() > 1:
                        mask = mask / 255.0
                    if self.mask_mode == "binary":
                        mask = (mask > 0.5).float()
        else:
            img = torch.from_numpy(img).float().unsqueeze(0) / 255.0
            if self.mask_mode == "label":
                mask = torch.from_numpy(mask).long().unsqueeze(0)
            else:
                mask = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)

        while mask.ndim < 3:
            mask = mask.unsqueeze(0)

        if self.return_meta:
            return img, mask, {'filename': img_path.name}
        return img, mask


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def make_fold_loaders(video_root: Path, bonus_root: Path, train_idx, val_idx,
                      batch_size=8, num_workers=2, use_bonus: bool = True,
                      train_transform=None, val_transform=None):
    """Create data loaders for a k-fold split."""
    train_tf = train_transform if train_transform is not None else get_train_transforms()
    val_tf = val_transform if val_transform is not None else get_val_transforms()

    train_ds = Subset(AugmentedMicroscopyDataset(video_root, transform=train_tf, mask_mode="binary"), train_idx)
    val_ds = Subset(AugmentedMicroscopyDataset(video_root, transform=val_tf, mask_mode="binary"), val_idx)

    datasets = [train_ds]
    bonus_len = 0
    if use_bonus:
        bonus_ds = AugmentedMicroscopyDataset(bonus_root, transform=train_tf, mask_mode="binary")
        datasets.append(bonus_ds)
        bonus_len = len(bonus_ds)

    train_loader = DataLoader(ConcatDataset(datasets), batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=max(1, batch_size // 2), shuffle=False,
                           num_workers=num_workers, pin_memory=True)

    print(f"  Train: {len(train_idx)} video + {bonus_len} bonus = {len(train_loader.dataset)} samples")
    print(f"  Val: {len(val_idx)} samples")
    return train_loader, val_loader


def train_one_fold(model, train_loader, val_loader, device, epochs=40, lr=1e-3, verbose=True):
    """Train one fold of k-fold cross validation."""
    criterion = ComboLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device).float(), masks.to(device).float()
            optimizer.zero_grad()
            outputs = model(imgs)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[-1]
            loss = criterion(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device).float(), masks.to(device).float()
                outputs = model(imgs)
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[-1]
                loss = criterion(outputs, masks)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  Epoch {epoch:03d}: train={train_loss:.4f}, val={val_loss:.4f}, best={best_val:.4f}")

    model.load_state_dict(best_state)
    return model, best_val, history


def infer_fold_oof(model, val_indices, video_root: Path, video_map: Path, device,
                   threshold: float = 0.5, min_distance_peaks: int = 5):
    """Get out-of-fold predictions for validation set."""
    infer_ds = Subset(AugmentedMicroscopyDataset(video_root, transform=get_val_transforms(),
                                                  return_meta=True, mask_mode="binary"), val_indices)
    infer_loader = DataLoader(infer_ds, batch_size=1, shuffle=False)
    frame_lookup = pd.read_csv(video_map).set_index('filename')['real_frame_idx'].to_dict()

    model.eval()
    preds = []
    with torch.no_grad():
        for imgs, _, meta in infer_loader:
            filenames = meta['filename'] if isinstance(meta, dict) else [m['filename'] for m in meta]
            imgs = imgs.to(device).float()
            outputs = model(imgs)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[-1]
            probs = torch.sigmoid(outputs).cpu().numpy()

            for b in range(probs.shape[0]):
                prob_map = probs[b, 0]
                coordinates = peak_local_max(prob_map, min_distance=min_distance_peaks,
                                            threshold_abs=threshold, exclude_border=False)
                for cy, cx in coordinates:
                    preds.append({
                        'frame': int(frame_lookup[filenames[b]]),
                        'x': float(cx + ROI_X_MIN),
                        'y': float(cy + ROI_Y_MIN)
                    })
    return preds


# =============================================================================
# METRICS
# =============================================================================

def calculate_deta_robust(gt_df, pred_df, match_thresh=5.0):
    """Calculate Detection Accuracy (DetA) using Hungarian matching."""
    gt_df = gt_df.copy()
    pred_df = pred_df.copy()
    gt_df['frame'] = gt_df['frame'].astype(int)
    pred_df['frame'] = pred_df['frame'].astype(int)

    g_frames = set(gt_df['frame'].unique())
    p_frames = set(pred_df['frame'].unique())
    target_frames = sorted(list(g_frames.union(p_frames)))

    tp_total, fp_total, fn_total = 0, 0, 0

    for f in target_frames:
        g = gt_df[gt_df['frame'] == f]
        p = pred_df[pred_df['frame'] == f]
        g_coords = g[['x', 'y']].values
        p_coords = p[['x', 'y']].values
        n_g, n_p = len(g_coords), len(p_coords)

        if n_g == 0:
            fp_total += n_p
            continue
        if n_p == 0:
            fn_total += n_g
            continue

        dists = cdist(g_coords, p_coords)
        row_ind, col_ind = linear_sum_assignment(dists)
        curr_tp = sum(1 for r, c in zip(row_ind, col_ind) if dists[r, c] <= match_thresh)

        tp_total += curr_tp
        fn_total += (n_g - curr_tp)
        fp_total += (n_p - curr_tp)

    denom = tp_total + fn_total + fp_total
    return tp_total / denom if denom > 0 else 0.0


def hota(gt: pd.DataFrame, tr: pd.DataFrame, threshold: float = 5) -> Dict[str, float]:
    """Calculate HOTA (Higher Order Tracking Accuracy) metric."""
    gt = gt.copy()
    tr = tr.copy()

    if 'track_id' not in gt.columns or 'track_id' not in tr.columns:
        return {'HOTA': 0.0, 'AssA': 0.0, 'DetA': 0.0, 'LocA': 0.0}

    gt.track_id = gt.track_id.map({old: new for old, new in zip(gt.track_id.unique(), range(gt.track_id.nunique()))})
    tr.track_id = tr.track_id.map({old: new for old, new in zip(tr.track_id.unique(), range(tr.track_id.nunique()))})

    num_gt_ids = gt.track_id.nunique()
    num_tr_ids = tr.track_id.nunique()
    frames = sorted(set(gt.frame.unique()) | set(tr.frame.unique()))

    potential_matches_count = np.zeros((num_gt_ids, num_tr_ids))
    gt_id_count = np.zeros((num_gt_ids, 1))
    tracker_id_count = np.zeros((1, num_tr_ids))
    HOTA_TP = HOTA_FN = HOTA_FP = 0
    LocA = 0.0

    similarities = [1 - np.clip(spatial.distance.cdist(
        gt[gt.frame == t][['x', 'y']], tr[tr.frame == t][['x', 'y']]) / threshold, 0, 1)
        for t in frames]

    for t in frames:
        gt_ids_t = gt[gt.frame == t].track_id.to_numpy()
        tr_ids_t = tr[tr.frame == t].track_id.to_numpy()
        similarity = similarities[t]
        sim_iou_denom = similarity.sum(0)[np.newaxis, :] + similarity.sum(1)[:, np.newaxis] - similarity
        sim_iou = np.zeros_like(similarity)
        mask = sim_iou_denom > np.finfo('float').eps
        sim_iou[mask] = similarity[mask] / sim_iou_denom[mask]
        potential_matches_count[gt_ids_t[:, None], tr_ids_t[None, :]] += sim_iou
        gt_id_count[gt_ids_t] += 1
        tracker_id_count[0, tr_ids_t] += 1

    global_alignment_score = potential_matches_count / (gt_id_count + tracker_id_count - potential_matches_count)
    matches_count = np.zeros_like(potential_matches_count)

    for t in frames:
        gt_ids_t = gt[gt.frame == t].track_id.to_numpy()
        tr_ids_t = tr[tr.frame == t].track_id.to_numpy()

        if len(gt_ids_t) == 0:
            HOTA_FP += len(tr_ids_t)
            continue
        if len(tr_ids_t) == 0:
            HOTA_FN += len(gt_ids_t)
            continue

        similarity = similarities[t]
        score_mat = global_alignment_score[gt_ids_t[:, None], tr_ids_t[None, :]] * similarity
        match_rows, match_cols = optimize.linear_sum_assignment(-score_mat)
        mask = similarity[match_rows, match_cols] > 0
        alpha_match_rows = match_rows[mask]
        alpha_match_cols = match_cols[mask]
        num_matches = len(alpha_match_rows)

        HOTA_TP += num_matches
        HOTA_FN += len(gt_ids_t) - num_matches
        HOTA_FP += len(tr_ids_t) - num_matches

        if num_matches > 0:
            LocA += float(np.sum(similarity[alpha_match_rows, alpha_match_cols]))
            matches_count[gt_ids_t[alpha_match_rows], tr_ids_t[alpha_match_cols]] += 1

    ass_a = matches_count / np.maximum(1, gt_id_count + tracker_id_count - matches_count)
    AssA = np.sum(matches_count * ass_a) / np.maximum(1, HOTA_TP)
    DetA = HOTA_TP / np.maximum(1, HOTA_TP + HOTA_FN + HOTA_FP)
    HOTA_score = np.sqrt(DetA * AssA)

    return {'HOTA': HOTA_score, 'AssA': AssA, 'DetA': DetA, 'LocA': LocA}


# =============================================================================
# TRACKING
# =============================================================================

def link_detections(detections_per_frame: List[List[Tuple[float, float]]], max_dist: float = 7.0) -> pd.DataFrame:
    """Simple greedy linking of detections across frames."""
    next_track_id = 0
    active_tracks = {}
    records = []

    for frame_idx, detections in enumerate(detections_per_frame):
        assigned = [False] * len(detections)
        detection_track_id = [None] * len(detections)
        updated_tracks = {}

        for track_id, (tx, ty, last_frame) in list(active_tracks.items()):
            best_dist = max_dist
            best_idx = None
            for i, (x, y) in enumerate(detections):
                if assigned[i]:
                    continue
                dist = math.hypot(x - tx, y - ty)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            if best_idx is not None:
                assigned[best_idx] = True
                detection_track_id[best_idx] = track_id
                updated_tracks[track_id] = (detections[best_idx][0], detections[best_idx][1], frame_idx)

        for i, (x, y) in enumerate(detections):
            if not assigned[i]:
                tid = next_track_id
                next_track_id += 1
                detection_track_id[i] = tid
                updated_tracks[tid] = (x, y, frame_idx)

        active_tracks = updated_tracks
        for i, (x, y) in enumerate(detections):
            tid = detection_track_id[i]
            records.append({'frame': frame_idx, 'x': x, 'y': y, 'track_id': tid})

    return pd.DataFrame(records)


def track_detections_simple(preds: pd.DataFrame, max_dist: float) -> pd.DataFrame:
    """Track detections using simple linking."""
    max_frame = int(preds.frame.max()) if len(preds) else -1
    dets = [[] for _ in range(max_frame + 1)]
    for _, r in preds.iterrows():
        dets[int(r.frame)].append((float(r.x), float(r.y)))
    return link_detections(dets, max_dist=max_dist)


def run_laptrack(detections_df: pd.DataFrame, max_dist=15, closing_gap=2, min_length=2) -> pd.DataFrame:
    """Run LapTrack tracking on detections."""
    if detections_df.empty or not HAS_LAPTRACK:
        return pd.DataFrame()

    lt = LapTrack(
        track_dist_metric="sqeuclidean",
        track_cost_cutoff=max_dist**2,
        gap_closing_dist_metric="sqeuclidean",
        gap_closing_cost_cutoff=max_dist**2,
        gap_closing_max_frame_count=closing_gap,
        splitting_cost_cutoff=False,
        merging_cost_cutoff=False
    )

    try:
        track_df, _, _ = lt.predict_dataframe(
            detections_df,
            coordinate_cols=['y', 'x'],
            frame_col='frame',
            only_coordinate_cols=False
        )
        res = track_df.reset_index().rename(columns={'track_id': 'track_id'})
        if min_length > 1:
            counts = res['track_id'].value_counts()
            valid_ids = counts[counts >= min_length].index
            res = res[res['track_id'].isin(valid_ids)]
        return res[['frame', 'x', 'y', 'track_id']]
    except:
        return pd.DataFrame()


def run_laptrack_sweep(
    detections_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    track_cutoffs: List[int] = None,
    gap_cutoffs: List[int] = None,
    gap_frames: List[int] = None,
    match_thresh: float = 5.0,
    min_length: int = 2
) -> Dict:
    """
    Run LapTrack parameter sweep to find best AssA score.

    Args:
        detections_df: Detection dataframe with frame, x, y columns
        gt_df: Ground truth dataframe with frame, x, y, track_id columns
        track_cutoffs: List of squared distances for track_cost_cutoff (default: around 49)
        gap_cutoffs: List of squared distances for gap_closing_cost_cutoff (default: around 25)
        gap_frames: List of max frame gaps (default: [1, 2, 3])
        match_thresh: Distance threshold for HOTA calculation
        min_length: Minimum track length to keep

    Returns:
        Dict with best_params, best_scores, best_tracked_df, and all_results
    """
    if not HAS_LAPTRACK or detections_df.empty:
        return {'best_params': None, 'best_scores': None, 'best_tracked_df': pd.DataFrame(), 'all_results': []}

    # Default sweep ranges centered around known good values
    # Best: track_cost_cutoff=49 (7px), gap_closing_cost_cutoff=25 (5px), gap_frames=2
    if track_cutoffs is None:
        track_cutoffs = [36, 49, 64, 81]  # 6, 7, 8, 9 pixels
    if gap_cutoffs is None:
        gap_cutoffs = [16, 25, 36, 49]  # 4, 5, 6, 7 pixels
    if gap_frames is None:
        gap_frames = [1, 2, 3]

    all_results = []
    best_assa = -1
    best_result = None

    total_combos = len(track_cutoffs) * len(gap_cutoffs) * len(gap_frames)
    print(f"Running LapTrack sweep: {total_combos} combinations...")

    for track_cut in track_cutoffs:
        for gap_cut in gap_cutoffs:
            for gap_frame in gap_frames:
                try:
                    lt = LapTrack(
                        track_dist_metric="sqeuclidean",
                        track_cost_cutoff=track_cut,
                        gap_closing_dist_metric="sqeuclidean",
                        gap_closing_cost_cutoff=gap_cut,
                        gap_closing_max_frame_count=gap_frame,
                        splitting_cost_cutoff=False,
                        merging_cost_cutoff=False
                    )

                    track_df, _, _ = lt.predict_dataframe(
                        detections_df,
                        coordinate_cols=['y', 'x'],
                        frame_col='frame',
                        only_coordinate_cols=False
                    )
                    res = track_df.reset_index()

                    if min_length > 1:
                        counts = res['track_id'].value_counts()
                        valid_ids = counts[counts >= min_length].index
                        res = res[res['track_id'].isin(valid_ids)]

                    tracked = res[['frame', 'x', 'y', 'track_id']]

                    # Calculate HOTA metrics
                    scores = hota(gt_df, tracked, threshold=match_thresh)

                    result = {
                        'track_cost_cutoff': track_cut,
                        'gap_closing_cost_cutoff': gap_cut,
                        'gap_closing_max_frame_count': gap_frame,
                        'track_dist_px': np.sqrt(track_cut),
                        'gap_dist_px': np.sqrt(gap_cut),
                        'HOTA': scores['HOTA'],
                        'DetA': scores['DetA'],
                        'AssA': scores['AssA'],
                        'num_tracks': tracked['track_id'].nunique(),
                        'num_detections': len(tracked)
                    }
                    all_results.append(result)

                    if scores['AssA'] > best_assa:
                        best_assa = scores['AssA']
                        best_result = {
                            'params': {
                                'track_cost_cutoff': track_cut,
                                'gap_closing_cost_cutoff': gap_cut,
                                'gap_closing_max_frame_count': gap_frame
                            },
                            'scores': scores,
                            'tracked_df': tracked.copy()
                        }

                except Exception as e:
                    continue

    # Print results summary
    if all_results:
        results_df = pd.DataFrame(all_results).sort_values('AssA', ascending=False)
        print(f"\nTop 5 configurations by AssA:")
        print(results_df.head()[['track_dist_px', 'gap_dist_px', 'gap_closing_max_frame_count', 'HOTA', 'DetA', 'AssA']].to_string(index=False))

    if best_result:
        print(f"\n{'='*60}")
        print("BEST LAPTRACK RESULT")
        print(f"{'='*60}")
        print(f"HOTA: {best_result['scores']['HOTA']:.4f}")
        print(f"DetA: {best_result['scores']['DetA']:.4f}")
        print(f"AssA: {best_result['scores']['AssA']:.4f}")
        print(f"\nBest parameters:")
        print(f"  track_cost_cutoff: {best_result['params']['track_cost_cutoff']} (√={np.sqrt(best_result['params']['track_cost_cutoff']):.1f}px)")
        print(f"  gap_closing_cost_cutoff: {best_result['params']['gap_closing_cost_cutoff']} (√={np.sqrt(best_result['params']['gap_closing_cost_cutoff']):.1f}px)")
        print(f"  gap_closing_max_frame_count: {best_result['params']['gap_closing_max_frame_count']}")

        return {
            'best_params': best_result['params'],
            'best_scores': best_result['scores'],
            'best_tracked_df': best_result['tracked_df'],
            'all_results': all_results
        }

    return {'best_params': None, 'best_scores': None, 'best_tracked_df': pd.DataFrame(), 'all_results': all_results}


def run_btrack_tracking(detections_df: pd.DataFrame, config_path=None, max_search_radius=12.0) -> pd.DataFrame:
    """Run BTrack tracking on detections."""
    if not HAS_BTRACK or detections_df.empty:
        return pd.DataFrame()

    det_list = [{"t": int(row.frame), "x": float(row.x), "y": float(row.y), "z": 0.0}
                for _, row in detections_df.iterrows()]
    if not det_list:
        return pd.DataFrame()

    objects = []
    create_objects = None
    if hasattr(btrack, 'utils') and hasattr(btrack.utils, 'create_objects_from_array'):
        create_objects = btrack.utils.create_objects_from_array
    elif hasattr(btrack, 'io') and hasattr(btrack.io, 'create_objects_from_array'):
        create_objects = btrack.io.create_objects_from_array

    if create_objects:
        objects = create_objects(det_list, properties=['x', 'y'])
    else:
        return pd.DataFrame()

    try:
        import btrack.datasets as btrack_datasets
        config_to_use = config_path or btrack_datasets.cell_config()
    except:
        return pd.DataFrame()

    try:
        with btrack.BayesianTracker() as tracker:
            tracker.configure_from_file(config_to_use)
            tracker.max_search_radius = max_search_radius
            tracker.append(objects)
            tracker.volume = ((0, 1024), (0, 1024), (0, 1))
            tracker.track_interactive(step_size=100)
            tracker.optimize()
            result_df = tracker.to_pandas()
            # Standardize column names: BTrack uses 'ID' and 't', we need 'track_id' and 'frame'
            rename_map = {}
            if 'ID' in result_df.columns and 'track_id' not in result_df.columns:
                rename_map['ID'] = 'track_id'
            if 't' in result_df.columns and 'frame' not in result_df.columns:
                rename_map['t'] = 'frame'
            if rename_map:
                result_df = result_df.rename(columns=rename_map)
            # Ensure required columns exist
            required_cols = ['frame', 'x', 'y', 'track_id']
            if all(col in result_df.columns for col in required_cols):
                return result_df[required_cols]
            return result_df
    except Exception as e:
        print(f"BTrack error: {str(e)[:100]}...")
        return pd.DataFrame()


def run_tracking_sweep(detections_df: pd.DataFrame, val_csv_path: str, match_thresh=5.0):
    """Run parameter sweep over tracking algorithms."""
    print("\n=== Tracking Parameter Sweep ===")

    gt_df = pd.read_csv(val_csv_path)
    sub_gt = gt_df[
        (gt_df.x >= ROI_X_MIN) & (gt_df.x < ROI_X_MAX) &
        (gt_df.y >= ROI_Y_MIN) & (gt_df.y < ROI_Y_MAX)
    ].copy()

    results = []
    best_score = -1
    best_cfg = None
    best_tracks = None

    # LapTrack sweep
    lt_dists = [5, 8, 10, 13, 15, 18, 20]
    lt_gaps = [1, 2, 3]

    for d in lt_dists:
        for g in lt_gaps:
            name = f"LapTrack_d{d}_g{g}"
            try:
                tr = run_laptrack(detections_df.copy(), max_dist=d, closing_gap=g)
                if tr.empty:
                    continue
                metrics = hota(sub_gt, tr, threshold=match_thresh)
                assa = metrics['AssA']
                print(f"  {name:<20}: AssA={assa:.4f}, DetA={metrics['DetA']:.4f}, HOTA={metrics['HOTA']:.4f}")
                results.append({'name': name, 'AssA': assa, 'DetA': metrics['DetA'], 'HOTA': metrics['HOTA']})
                if assa > best_score:
                    best_score = assa
                    best_cfg = name
                    best_tracks = tr
            except:
                pass

    # BTrack sweep
    btrack_radii = [10.0, 15.0, 20.0, 30.0, 60.0]
    for r in btrack_radii:
        name = f"BTrack_r{r}"
        try:
            tr = run_btrack_tracking(detections_df.copy(), max_search_radius=r)
            if tr.empty:
                continue
            metrics = hota(sub_gt, tr, threshold=match_thresh)
            assa = metrics['AssA']
            print(f"  {name:<20}: AssA={assa:.4f}, DetA={metrics['DetA']:.4f}, HOTA={metrics['HOTA']:.4f}")
            results.append({'name': name, 'AssA': assa, 'DetA': metrics['DetA'], 'HOTA': metrics['HOTA']})
            if assa > best_score:
                best_score = assa
                best_cfg = name
                best_tracks = tr
        except:
            pass

    print(f"\nBest Config: {best_cfg} with AssA={best_score:.4f}")
    return pd.DataFrame(results), best_cfg, best_tracks


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_training_history(history: Dict[str, List[float]], title: str = "Training History"):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 4))
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def show_predictions(model, dataset, device, num_samples=4, threshold=0.5):
    """Visualize model predictions on sample images."""
    model.eval()
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(indices):
        img, mask = dataset[idx]
        if isinstance(img, tuple):
            img = img[0]

        with torch.no_grad():
            out = model(img.unsqueeze(0).to(device).float())
            if isinstance(out, (list, tuple)):
                out = out[-1]
            prob = torch.sigmoid(out)[0, 0].cpu().numpy()

        img_np = img.squeeze().cpu().numpy()
        mask_np = mask.squeeze().cpu().numpy()
        pred_mask = (prob >= threshold).astype(float)

        axes[i, 0].imshow(img_np, cmap='gray')
        axes[i, 0].set_title('Input')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask_np, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(prob, cmap='viridis')
        axes[i, 2].set_title('Probability Map')
        axes[i, 2].axis('off')

        axes[i, 3].imshow(pred_mask, cmap='gray')
        axes[i, 3].set_title(f'Prediction (t={threshold})')
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.show()


def show_detection_overlay(model, dataset, val_indices, device, threshold=0.5,
                           val_csv_path=None, video_map_path=None, min_distance_peaks=5):
    """Show detection overlay comparing predictions with ground truth."""
    if len(val_indices) == 0:
        print("No validation indices to visualize")
        return

    idx = int(np.random.choice(val_indices))
    img, mask = dataset[idx]
    if isinstance(img, tuple):
        img = img[0]

    base = img.squeeze().cpu().numpy()
    if base.max() > 1:
        base = base / 255.0

    with torch.no_grad():
        out = model(img.unsqueeze(0).to(device).float())
        if isinstance(out, (list, tuple)):
            out = out[-1]
        prob = torch.sigmoid(out)[0, 0].cpu().numpy()

    coordinates = peak_local_max(prob, min_distance=min_distance_peaks,
                                threshold_abs=threshold, exclude_border=False)
    px = [coord[1] for coord in coordinates]
    py = [coord[0] for coord in coordinates]

    gx, gy = [], []
    if val_csv_path and video_map_path:
        img_path = dataset.image_paths[idx]
        filename = img_path.name
        video_map_df = pd.read_csv(video_map_path)
        real_frame_idx = video_map_df[video_map_df['filename'] == filename]['real_frame_idx'].iloc[0]

        original_gt_coords = pd.read_csv(val_csv_path)
        gt_points = original_gt_coords[original_gt_coords['frame'] == real_frame_idx]

        for _, row in gt_points.iterrows():
            gx.append(row['x'] - ROI_X_MIN)
            gy.append(row['y'] - ROI_Y_MIN)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(base, cmap='gray')
    axes[0].set_title(f"Input (idx {idx})")
    axes[0].axis('off')

    axes[1].imshow(prob, cmap='viridis')
    axes[1].set_title("Probability Map")
    axes[1].axis('off')

    axes[2].imshow(base, cmap='gray')
    axes[2].scatter(gx, gy, s=50, facecolors='none', edgecolors='lime', linewidths=2, label='GT')
    axes[2].scatter(px, py, s=40, marker='x', color='red', linewidths=2, label='Pred')
    axes[2].set_title(f"Overlay (GT: {len(gx)}, Pred: {len(px)})")
    axes[2].legend(loc='upper right')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


def show_tracking_animation(tracks_df: pd.DataFrame, video_path: str,
                           y_min=ROI_Y_MIN, y_max=ROI_Y_MAX,
                           x_min=ROI_X_MIN, x_max=ROI_X_MAX,
                           tail_length=10, max_frames=50):
    """Create and display tracking animation."""
    video = tifffile.imread(video_path)
    cropped = video[:, y_min:y_max, x_min:x_max]

    tracks_roi = tracks_df[
        (tracks_df.y >= y_min) & (tracks_df.y < y_max) &
        (tracks_df.x >= x_min) & (tracks_df.x < x_max)
    ].copy()

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cropped[0], cmap='magma')
    particles = tracks_roi['track_id'].unique()

    colors = plt.cm.tab20(np.linspace(0, 1, len(particles)))
    track_colors = {pid: colors[i % len(colors)] for i, pid in enumerate(particles)}

    line_collections = {pid: mc.LineCollection([], linewidths=1.5, colors=track_colors[pid])
                        for pid in particles}
    for lc in line_collections.values():
        ax.add_collection(lc)

    dot = ax.scatter([], [], s=30, c='yellow', zorder=5)

    def animate(i):
        im.set_array(cropped[i])
        window = tracks_roi[(tracks_roi['frame'] >= i - tail_length) & (tracks_roi['frame'] <= i)]
        now = window[window['frame'] == i]

        if len(now) > 0:
            coords = np.column_stack((now.x.values - x_min, now.y.values - y_min))
            dot.set_offsets(coords)
        else:
            dot.set_offsets(np.empty((0, 2)))

        for pid in particles:
            traj = window[window['track_id'] == pid].sort_values('frame')
            if len(traj) >= 2:
                segs = [[(x0 - x_min, y0 - y_min), (x1 - x_min, y1 - y_min)]
                        for x0, y0, x1, y1 in zip(traj.x.values[:-1], traj.y.values[:-1],
                                                   traj.x.values[1:], traj.y.values[1:])]
                line_collections[pid].set_segments(segs)
            else:
                line_collections[pid].set_segments([])

        return [im, dot] + list(line_collections.values())

    ani = FuncAnimation(fig, animate, frames=min(max_frames, len(cropped)), interval=100, blit=True)
    plt.close(fig)
    return HTML(ani.to_jshtml())


def print_results_summary(fold_results: List[Dict], best_fold: Dict):
    """Print a summary of k-fold training results."""
    print("\n" + "="*60)
    print("TRAINING RESULTS SUMMARY")
    print("="*60)

    print("\nPer-Fold Results:")
    print("-"*40)
    for res in fold_results:
        print(f"  Fold {res['fold']}: DetA = {res['deta']:.4f}")

    mean_deta = np.mean([r['deta'] for r in fold_results])
    std_deta = np.std([r['deta'] for r in fold_results])

    print("-"*40)
    print(f"  Mean DetA: {mean_deta:.4f} +/- {std_deta:.4f}")
    print(f"\nBest Fold: {best_fold['fold']} (DetA = {best_fold['deta']:.4f})")
    print("="*60)


# =============================================================================
# STARDIST LIGHTNING MODULE (if available)
# =============================================================================

if HAS_LIGHTNING and HAS_STARDIST_TORCH:

    class LossHistoryCallback(Callback):
        """Callback to record training history."""
        def __init__(self):
            super().__init__()
            self.history = {'train_loss': [], 'val_loss': [], 'lr': []}

        def on_train_epoch_end(self, trainer, pl_module):
            if 'train_loss' in trainer.callback_metrics:
                self.history['train_loss'].append(trainer.callback_metrics['train_loss'].item())
            # Record learning rate
            if trainer.optimizers:
                lr = trainer.optimizers[0].param_groups[0]['lr']
                self.history['lr'].append(lr)

        def on_validation_epoch_end(self, trainer, pl_module):
            if 'val_loss' in trainer.callback_metrics:
                self.history['val_loss'].append(trainer.callback_metrics['val_loss'].item())


    class StarDistCombinedLoss(nn.Module):
        """
        Combined loss for StarDist training:
        - Focal Loss for binary mask (handles class imbalance)
        - Dice Loss for binary mask (overlap-based)
        - Smooth L1 (Huber) for distance regression (robust to outliers)
        """
        def __init__(
            self,
            focal_weight: float = 1.0,
            dice_weight: float = 1.0,
            dist_weight: float = 1.0,
            focal_alpha: float = 0.75,
            focal_gamma: float = 2.0,
            smooth_l1_beta: float = 1.0
        ):
            super().__init__()
            self.focal_weight = focal_weight
            self.dice_weight = dice_weight
            self.dist_weight = dist_weight
            self.focal_alpha = focal_alpha
            self.focal_gamma = focal_gamma
            self.smooth_l1_beta = smooth_l1_beta

        def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            """Focal loss for handling class imbalance."""
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
            pt = torch.exp(-bce)
            focal = self.focal_alpha * (1 - pt) ** self.focal_gamma * bce
            return focal.mean()

        def dice_loss(self, pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
            """Dice loss for overlap optimization."""
            pred_sigmoid = torch.sigmoid(pred)
            intersection = (pred_sigmoid * target).sum()
            union = pred_sigmoid.sum() + target.sum()
            dice = (2.0 * intersection + eps) / (union + eps)
            return 1.0 - dice

        def masked_smooth_l1(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            """Smooth L1 loss only inside object regions."""
            diff = torch.abs(pred - target)
            # Smooth L1: quadratic for small errors, linear for large
            loss = torch.where(
                diff < self.smooth_l1_beta,
                0.5 * diff ** 2 / self.smooth_l1_beta,
                diff - 0.5 * self.smooth_l1_beta
            )
            mask_expanded = mask.expand_as(loss)
            masked_loss = (loss * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)
            return masked_loss

        def forward(
            self,
            pred_bin: torch.Tensor,
            gt_bin: torch.Tensor,
            pred_dist: torch.Tensor,
            gt_dist: torch.Tensor
        ) -> Dict[str, torch.Tensor]:
            """
            Compute combined loss.

            Returns dict with total loss and individual components for logging.
            """
            # Binary mask losses
            loss_focal = self.focal_loss(pred_bin, gt_bin) if self.focal_weight > 0 else torch.tensor(0.0)
            loss_dice = self.dice_loss(pred_bin, gt_bin) if self.dice_weight > 0 else torch.tensor(0.0)

            # Distance regression loss (masked)
            loss_dist = self.masked_smooth_l1(pred_dist, gt_dist, gt_bin) if self.dist_weight > 0 else torch.tensor(0.0)

            # Combine
            total = (
                self.focal_weight * loss_focal +
                self.dice_weight * loss_dice +
                self.dist_weight * loss_dist
            )

            return {
                'total': total,
                'focal': loss_focal,
                'dice': loss_dice,
                'dist': loss_dist
            }


    class StarDistLightning(pl.LightningModule):
        """
        PyTorch Lightning module for StarDist training with improvements:
        - Combined loss (Focal + Dice + Smooth L1)
        - Weight decay regularization
        - Learning rate scheduling (ReduceLROnPlateau)
        - Configurable encoder backbone
        """
        def __init__(
            self,
            n_rays: int = 32,
            lr: float = 1e-4,
            weight_decay: float = 1e-4,
            encoder_name: str = "resnet18",
            focal_weight: float = 1.0,
            dice_weight: float = 1.0,
            dist_weight: float = 1.0,
            scheduler_patience: int = 5,
            scheduler_factor: float = 0.5
        ):
            super().__init__()
            self.save_hyperparameters()
            self.n_rays = n_rays
            self.lr = lr
            self.weight_decay = weight_decay
            self.scheduler_patience = scheduler_patience
            self.scheduler_factor = scheduler_factor

            # Build StarDist model
            wrapper = StarDist(
                n_nuc_classes=1,
                n_rays=n_rays,
                enc_name=encoder_name,
                model_kwargs={"encoder_kws": {"in_chans": 1}}
            )
            self.model = wrapper.model

            # Combined loss
            self.criterion = StarDistCombinedLoss(
                focal_weight=focal_weight,
                dice_weight=dice_weight,
                dist_weight=dist_weight
            )

        def forward(self, x):
            return self.model(x)

        def _compute_loss(self, batch):
            """Compute loss for a batch."""
            images = batch["image"]
            gt_dist = batch["stardist_map"]
            gt_bin = batch["binary_map"]

            out = self(images)
            nuc_out = out["nuc"]
            pred_dist = nuc_out.aux_map
            pred_bin = nuc_out.binary_map

            losses = self.criterion(pred_bin, gt_bin, pred_dist, gt_dist)
            return losses

        def training_step(self, batch, batch_idx):
            losses = self._compute_loss(batch)

            # Log individual losses
            self.log("train_loss", losses['total'], on_step=False, on_epoch=True, prog_bar=True)
            self.log("train_focal", losses['focal'], on_step=False, on_epoch=True)
            self.log("train_dice", losses['dice'], on_step=False, on_epoch=True)
            self.log("train_dist", losses['dist'], on_step=False, on_epoch=True)

            return losses['total']

        def validation_step(self, batch, batch_idx):
            losses = self._compute_loss(batch)

            self.log("val_loss", losses['total'], on_step=False, on_epoch=True, prog_bar=True)
            self.log("val_focal", losses['focal'], on_step=False, on_epoch=True)
            self.log("val_dice", losses['dice'], on_step=False, on_epoch=True)
            self.log("val_dist", losses['dist'], on_step=False, on_epoch=True)

            return losses['total']

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
                verbose=True
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1
                }
            }


    def get_stardist_augmentation(
        rotate_p: float = 0.7,
        flip_p: float = 0.5,
        brightness_p: float = 0.3,
        noise_p: float = 0.2,
        blur_p: float = 0.1,
        elastic_p: float = 0.2
    ):
        """
        Get augmentation pipeline for StarDist training.

        Uses albumentations with instance-mask-safe transforms.
        Note: Transforms that change geometry (rotate, flip, elastic) are applied
        to both image and mask. The mask is relabeled after augmentation.
        """
        return A.Compose([
            A.Rotate(limit=180, p=rotate_p, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.HorizontalFlip(p=flip_p),
            A.VerticalFlip(p=flip_p),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=brightness_p),
            A.GaussNoise(var_limit=(5.0, 30.0), p=noise_p),
            A.GaussianBlur(blur_limit=(3, 5), p=blur_p),
            A.ElasticTransform(
                alpha=50, sigma=10, alpha_affine=10,
                border_mode=cv2.BORDER_CONSTANT, value=0,
                p=elastic_p
            ),
        ])


    class StarDistDataset(Dataset):
        """
        Dataset for StarDist training with distance maps.

        Supports data augmentation with proper handling of instance masks.
        """
        def __init__(self, pairs, n_rays=32, augment=False, aug_params=None):
            self.pairs = pairs
            self.n_rays = n_rays
            self.augment = augment

            # Setup augmentation
            if augment:
                aug_params = aug_params or {}
                self.transform = get_stardist_augmentation(**aug_params)
            else:
                self.transform = None

        def __len__(self):
            return len(self.pairs)

        def _relabel_mask(self, mask: np.ndarray) -> np.ndarray:
            """Relabel instance mask after augmentation to ensure consecutive IDs."""
            if mask.max() == 0:
                return mask
            # Use skimage.measure.label to get clean instance labels
            binary = (mask > 0).astype(np.uint8)
            relabeled = label(binary)
            return relabeled.astype(np.int32)

        def __getitem__(self, idx):
            img_path, mask_path = self.pairs[idx]

            # Load image
            if img_path.suffix.lower() in {'.tif', '.tiff'}:
                img = tifffile.imread(img_path)
            else:
                img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Load mask
            mask = tifffile.imread(mask_path)
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = mask.astype(np.int32)

            # Apply augmentation
            if self.transform is not None:
                # Convert mask to float for augmentation (preserves instance IDs better)
                augmented = self.transform(image=img, mask=mask.astype(np.float32))
                img = augmented["image"]
                mask = augmented["mask"].astype(np.int32)
                # Relabel to ensure clean instance IDs after geometric transforms
                mask = self._relabel_mask(mask)

            # Normalize image (percentile-based)
            img = img.astype(np.float32)
            p1, p99 = np.percentile(img, (1, 99.8))
            img = np.clip(img, p1, p99)
            img = (img - p1) / (p99 - p1 + 1e-8)

            # Generate StarDist maps
            dist_map = gen_stardist_maps(mask, n_rays=self.n_rays)
            binary_map = (mask > 0).astype(np.float32)[np.newaxis, ...]

            # To tensors
            img_tensor = torch.from_numpy(img[np.newaxis, ...].astype(np.float32))
            dist_tensor = torch.from_numpy(dist_map.astype(np.float32))
            binary_tensor = torch.from_numpy(binary_map)

            return {
                "image": img_tensor,
                "stardist_map": dist_tensor,
                "binary_map": binary_tensor,
                "id": str(img_path.name)
            }


def sweep_stardist_thresholds(
    model,
    val_pairs: List[Tuple[Path, Path]],
    gt_df: pd.DataFrame,
    fn_to_frame: Dict[str, int],
    device: str,
    prob_thresholds: List[float] = None,
    nms_thresholds: List[float] = None,
    match_thresh: float = 5.0,
    verbose: bool = True
) -> Tuple[pd.DataFrame, float, float]:
    """
    Sweep over probability and NMS thresholds to find optimal values.

    Args:
        model: Trained StarDist model
        val_pairs: List of (image_path, mask_path) tuples for validation
        gt_df: Ground truth DataFrame with columns [frame, x, y]
        fn_to_frame: Dict mapping filename to frame index
        device: Device to run inference on
        prob_thresholds: List of probability thresholds to try
        nms_thresholds: List of NMS thresholds to try
        match_thresh: Distance threshold for DetA calculation
        verbose: Print progress

    Returns:
        results_df: DataFrame with sweep results
        best_prob: Best probability threshold
        best_nms: Best NMS threshold
    """
    if not HAS_STARDIST_TORCH:
        raise ImportError("cellseg_models_pytorch required")

    # Default threshold ranges
    if prob_thresholds is None:
        prob_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    if nms_thresholds is None:
        nms_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    # Filter GT to ROI
    gt_roi = gt_df[
        (gt_df.x >= ROI_X_MIN) & (gt_df.x < ROI_X_MAX) &
        (gt_df.y >= ROI_Y_MIN) & (gt_df.y < ROI_Y_MAX)
    ].copy()

    # Pre-compute model outputs for all validation images
    model.eval()
    model.to(device)

    if verbose:
        print("Pre-computing model outputs...")

    outputs_cache = []
    for img_path, _ in val_pairs:
        if img_path.suffix.lower() in {'.tif', '.tiff'}:
            img = tifffile.imread(img_path)
        else:
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        p1, p99 = np.percentile(img, (1, 99.8))
        img = np.clip(img, p1, p99)
        img = (img - p1) / (p99 - p1 + 1e-8)

        inp = torch.from_numpy(img.astype(np.float32)[np.newaxis, np.newaxis, ...]).to(device)

        with torch.no_grad():
            out = model(inp)
            prob = torch.sigmoid(out['nuc'].binary_map).cpu().numpy().squeeze()
            dist = out['nuc'].aux_map.cpu().numpy().squeeze()

        fidx = fn_to_frame.get(img_path.name, -1)
        outputs_cache.append((prob, dist, fidx))

    # Sweep thresholds
    if verbose:
        print(f"Sweeping {len(prob_thresholds)} x {len(nms_thresholds)} threshold combinations...")

    results = []
    best_deta = -1
    best_prob = prob_thresholds[0]
    best_nms = nms_thresholds[0]

    for prob_t in prob_thresholds:
        for nms_t in nms_thresholds:
            preds = []
            valid_frames = set()

            for prob, dist, fidx in outputs_cache:
                if fidx == -1:
                    continue

                try:
                    labels = post_proc_stardist(prob, dist, score_thresh=prob_t, iou_thresh=nms_t)
                except:
                    labels = np.zeros_like(prob, dtype=int)

                valid_frames.add(fidx)
                for p in regionprops(labels):
                    y, x = p.centroid
                    preds.append({'frame': fidx, 'x': x + ROI_X_MIN, 'y': y + ROI_Y_MIN})

            # Calculate DetA
            pred_df = pd.DataFrame(preds)
            if not pred_df.empty and len(valid_frames) > 0:
                gt_filtered = gt_roi[gt_roi.frame.isin(valid_frames)]
                pred_filtered = pred_df[pred_df.frame.isin(valid_frames)]

                try:
                    deta = calculate_deta_robust(gt_filtered, pred_filtered, match_thresh=match_thresh)
                except:
                    deta = 0.0
            else:
                deta = 0.0

            results.append({
                'prob_thresh': prob_t,
                'nms_thresh': nms_t,
                'deta': deta,
                'n_detections': len(preds)
            })

            if deta > best_deta:
                best_deta = deta
                best_prob = prob_t
                best_nms = nms_t

            if verbose:
                print(f"  prob={prob_t:.2f}, nms={nms_t:.2f}: DetA={deta:.4f} ({len(preds)} detections)")

    results_df = pd.DataFrame(results)

    if verbose:
        print(f"\nBest: prob_thresh={best_prob:.2f}, nms_thresh={best_nms:.2f}, DetA={best_deta:.4f}")

    return results_df, best_prob, best_nms


def train_stardist_kfold(
    dataset_root: Path,
    video_map_path: Path,
    val_csv_path: str,
    k_splits: int = 5,
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 1e-4,
    n_rays: int = 32,
    prob_thresh: float = 0.5,
    nms_thresh: float = 0.3,
    use_bonus: bool = True,
    save_dir: Path = Path("."),
    match_thresh: float = 5.0,
    device: str = "cuda",
    # New parameters for improvements
    use_augmentation: bool = True,
    aug_params: dict = None,
    weight_decay: float = 1e-4,
    focal_weight: float = 1.0,
    dice_weight: float = 1.0,
    dist_weight: float = 1.0,
    scheduler_patience: int = 5,
    scheduler_factor: float = 0.5,
    early_stopping_patience: int = 10,
    encoder_name: str = "resnet18",
    run_threshold_sweep: bool = True,
    sweep_prob_thresholds: List[float] = None,
    sweep_nms_thresholds: List[float] = None
):
    """
    Train StarDist model using K-Fold cross-validation with improvements.

    Args:
        dataset_root: Path to dataset with video/ and bonus/ subdirs
        video_map_path: CSV mapping filenames to frame indices
        val_csv_path: Ground truth CSV for validation
        k_splits: Number of K-Fold splits
        epochs: Maximum epochs per fold
        batch_size: Batch size for training
        lr: Initial learning rate
        n_rays: Number of StarDist rays
        prob_thresh: Initial probability threshold (used if no sweep)
        nms_thresh: Initial NMS threshold (used if no sweep)
        use_bonus: Whether to include bonus training data
        save_dir: Directory to save models
        match_thresh: Distance threshold for DetA calculation
        device: Device for inference ('cuda' or 'cpu')

        # Improvement parameters
        use_augmentation: Enable data augmentation (set False for baseline)
        aug_params: Dict of augmentation parameters (see get_stardist_augmentation)
        weight_decay: L2 regularization strength (0 to disable)
        focal_weight: Weight for focal loss component
        dice_weight: Weight for dice loss component
        dist_weight: Weight for distance regression loss
        scheduler_patience: Epochs to wait before LR reduction
        scheduler_factor: Factor to reduce LR by
        early_stopping_patience: Epochs to wait before early stopping
        encoder_name: Backbone encoder (resnet18, resnet34, resnet50, efficientnet-b0)
        run_threshold_sweep: Run post-training threshold optimization
        sweep_prob_thresholds: Prob thresholds to sweep (default: [0.3, 0.4, 0.5, 0.6, 0.7])
        sweep_nms_thresholds: NMS thresholds to sweep (default: [0.1, 0.2, 0.3, 0.4, 0.5])

    Returns:
        fold_results: List of dicts with fold metrics and models
        best_fold: Dict with best fold info
        all_preds_df: DataFrame with all predictions
    """
    if not HAS_LIGHTNING or not HAS_STARDIST_TORCH:
        raise ImportError("pytorch_lightning and cellseg_models_pytorch are required for StarDist training")

    from pytorch_lightning.callbacks import EarlyStopping

    video_root = Path(dataset_root) / "video"
    bonus_root = Path(dataset_root) / "bonus"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Print configuration
    print("=" * 60)
    print("StarDist K-Fold Training Configuration")
    print("=" * 60)
    print(f"  Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}")
    print(f"  Encoder: {encoder_name}, N_Rays: {n_rays}")
    print(f"  Data Augmentation: {'ENABLED' if use_augmentation else 'DISABLED (baseline)'}")
    print(f"  Weight Decay: {weight_decay}")
    print(f"  Loss: Focal({focal_weight}) + Dice({dice_weight}) + Dist({dist_weight})")
    print(f"  LR Scheduler: ReduceLROnPlateau(patience={scheduler_patience}, factor={scheduler_factor})")
    print(f"  Early Stopping: patience={early_stopping_patience}")
    print(f"  Threshold Sweep: {'ENABLED' if run_threshold_sweep else 'DISABLED'}")
    print("=" * 60)

    # Load frame lookup
    frame_lookup_df = pd.read_csv(video_map_path)
    fn_to_frame = frame_lookup_df.set_index('filename')['real_frame_idx'].to_dict()

    # Build image-mask pairs for video
    pairs = []
    img_paths = sorted((video_root / "images").glob("*"))
    mask_lookup = {p.stem: p for p in (video_root / "masks").glob("*.tif")}
    for p in img_paths:
        if p.stem in mask_lookup:
            pairs.append((p, mask_lookup[p.stem]))

    video_pairs = pairs[:]

    # Add bonus pairs if requested
    if use_bonus:
        b_img = sorted((bonus_root / "images").glob("*"))
        b_mask = {p.stem: p for p in (bonus_root / "masks").glob("*.tif")}
        for p in b_img:
            if p.stem in b_mask:
                pairs.append((p, b_mask[p.stem]))

    print(f"Video pairs: {len(video_pairs)}, Total pairs (with bonus): {len(pairs)}")

    # K-Fold
    kf = KFold(n_splits=k_splits, shuffle=True, random_state=42)
    fold_results = []
    all_preds = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(video_pairs), 1):
        print(f"\n{'='*60}")
        print(f"FOLD {fold}/{k_splits} - StarDist Training")
        print(f"{'='*60}")

        # Build train/val pairs
        train_pairs_video = [video_pairs[i] for i in train_idx]
        val_pairs_video = [video_pairs[i] for i in val_idx]
        train_pairs = list(train_pairs_video)
        if use_bonus:
            train_pairs += pairs[len(video_pairs):]

        print(f"  Train: {len(train_pairs)} ({len(train_pairs_video)} video + {len(train_pairs) - len(train_pairs_video)} bonus)")
        print(f"  Val: {len(val_pairs_video)}")

        # Create datasets with augmentation option
        train_ds = StarDistDataset(
            train_pairs,
            n_rays=n_rays,
            augment=use_augmentation,
            aug_params=aug_params
        )
        val_ds = StarDistDataset(val_pairs_video, n_rays=n_rays, augment=False)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

        # Create model with improved loss and optimizer
        model = StarDistLightning(
            n_rays=n_rays,
            lr=lr,
            weight_decay=weight_decay,
            encoder_name=encoder_name,
            focal_weight=focal_weight,
            dice_weight=dice_weight,
            dist_weight=dist_weight,
            scheduler_patience=scheduler_patience,
            scheduler_factor=scheduler_factor
        )
        history_cb = LossHistoryCallback()

        # Setup checkpoint
        fold_save_dir = save_dir / f"stardist_fold_{fold}"
        checkpoint_cb = ModelCheckpoint(
            dirpath=fold_save_dir,
            filename="best_model",
            monitor="val_loss",
            mode="min",
            save_top_k=1
        )

        # Early stopping callback
        early_stop_cb = EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            mode="min",
            verbose=True
        )

        # Train with all callbacks
        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator="auto",
            devices=1,
            callbacks=[checkpoint_cb, history_cb, early_stop_cb],
            default_root_dir=fold_save_dir,
            enable_progress_bar=True,
            log_every_n_steps=1
        )
        trainer.fit(model, train_loader, val_loader)

        # Load best model
        best_path = checkpoint_cb.best_model_path
        model.load_state_dict(torch.load(best_path)['state_dict'])
        model.eval()
        model.to(device)

        # Run threshold sweep for this fold if enabled
        fold_prob_thresh = prob_thresh
        fold_nms_thresh = nms_thresh

        if run_threshold_sweep:
            print(f"\nRunning threshold sweep for Fold {fold}...")
            gt_df = pd.read_csv(val_csv_path)
            sweep_results, fold_prob_thresh, fold_nms_thresh = sweep_stardist_thresholds(
                model.model,
                val_pairs_video,
                gt_df,
                fn_to_frame,
                device,
                prob_thresholds=sweep_prob_thresholds,
                nms_thresholds=sweep_nms_thresholds,
                match_thresh=match_thresh,
                verbose=True
            )

        # Evaluate on validation set with best thresholds
        print(f"\nEvaluating Fold {fold} with prob={fold_prob_thresh:.2f}, nms={fold_nms_thresh:.2f}...")
        fold_preds = []
        valid_frames = set()

        for img_path, _ in val_pairs_video:
            # Load and preprocess
            if img_path.suffix.lower() in {'.tif', '.tiff'}:
                img = tifffile.imread(img_path)
            else:
                img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            p1, p99 = np.percentile(img, (1, 99.8))
            img = np.clip(img, p1, p99)
            img = (img - p1) / (p99 - p1 + 1e-8)
            img = img.astype(np.float32)

            inp = torch.from_numpy(img[np.newaxis, np.newaxis, ...]).to(device)

            with torch.no_grad():
                out = model(inp)
                prob = torch.sigmoid(out['nuc'].binary_map).cpu().numpy().squeeze()
                dist = out['nuc'].aux_map.cpu().numpy().squeeze()

                try:
                    labels = post_proc_stardist(prob, dist, score_thresh=fold_prob_thresh, iou_thresh=fold_nms_thresh)
                except:
                    labels = np.zeros_like(prob, dtype=int)

                props = regionprops(labels)
                fidx = fn_to_frame.get(img_path.name, -1)

                if fidx != -1:
                    valid_frames.add(fidx)
                    for p in props:
                        y, x = p.centroid
                        fold_preds.append({
                            'frame': fidx,
                            'x': x + ROI_X_MIN,
                            'y': y + ROI_Y_MIN
                        })

        all_preds.extend(fold_preds)
        fold_preds_df = pd.DataFrame(fold_preds)

        # Calculate DetA
        deta = 0.0
        if len(valid_frames) > 0:
            gt = pd.read_csv(val_csv_path)
            sub_gt = gt[
                (gt.x >= ROI_X_MIN) & (gt.x < ROI_X_MAX) &
                (gt.y >= ROI_Y_MIN) & (gt.y < ROI_Y_MAX) &
                (gt.frame.isin(valid_frames))
            ].copy()

            if not fold_preds_df.empty:
                fold_preds_df = fold_preds_df[fold_preds_df.frame.isin(valid_frames)]

            if not sub_gt.empty:
                try:
                    deta = calculate_deta_robust(sub_gt, fold_preds_df, match_thresh=match_thresh)
                except:
                    deta = 0.0

        print(f"Fold {fold} Final DetA: {deta:.4f}")

        fold_results.append({
            'fold': fold,
            'deta': deta,
            'history': history_cb.history,
            'model': model,
            'model_path': best_path,
            'best_prob_thresh': fold_prob_thresh,
            'best_nms_thresh': fold_nms_thresh,
            'stopped_epoch': trainer.current_epoch
        })

    # Find best fold
    fold_results.sort(key=lambda x: x['deta'], reverse=True)
    best_fold = fold_results[0]

    print(f"\n{'='*60}")
    print(f"Best Fold: {best_fold['fold']} (DetA = {best_fold['deta']:.4f})")
    print(f"  Optimal thresholds: prob={best_fold['best_prob_thresh']:.2f}, nms={best_fold['best_nms_thresh']:.2f}")
    print(f"  Stopped at epoch: {best_fold['stopped_epoch']}")
    print(f"{'='*60}")

    # Save best model
    best_model = best_fold['model']
    torch.save(best_model.state_dict(), save_dir / "best_stardist_model.pth")

    return fold_results, best_fold, pd.DataFrame(all_preds)


def infer_stardist_full_video(
    model,
    video_root: Path,
    video_map_path: Path,
    device: str,
    prob_thresh: float = 0.5,
    nms_thresh: float = 0.3
):
    """Run StarDist inference on all video frames."""
    if not HAS_STARDIST_TORCH:
        raise ImportError("cellseg_models_pytorch required")

    fn_to_frame = pd.read_csv(video_map_path).set_index('filename')['real_frame_idx'].to_dict()

    model.eval()
    model.to(device)
    preds = []

    for img_path in sorted((video_root / "images").glob("*")):
        if img_path.suffix.lower() in {'.tif', '.tiff'}:
            img = tifffile.imread(img_path)
        else:
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        p1, p99 = np.percentile(img, (1, 99.8))
        img = np.clip(img, p1, p99)
        img = (img - p1) / (p99 - p1 + 1e-8)

        inp = torch.from_numpy(img.astype(np.float32)[np.newaxis, np.newaxis, ...]).to(device)

        with torch.no_grad():
            out = model(inp)
            prob = torch.sigmoid(out['nuc'].binary_map).cpu().numpy().squeeze()
            dist = out['nuc'].aux_map.cpu().numpy().squeeze()

            try:
                labels = post_proc_stardist(prob, dist, score_thresh=prob_thresh, iou_thresh=nms_thresh)
            except:
                labels = np.zeros_like(prob, dtype=int)

            fidx = fn_to_frame.get(img_path.name, -1)
            for p in regionprops(labels):
                y, x = p.centroid
                preds.append({'frame': fidx, 'x': x + ROI_X_MIN, 'y': y + ROI_Y_MIN})

    return pd.DataFrame(preds)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get available device (CUDA or CPU)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device
