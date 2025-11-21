import os
import random
import numpy as np
import torch
import requests
import zipfile
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
import pandas as pd
import matplotlib.collections as mc

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seeds set to {seed}")

def download_and_unzip(url, extract_to, chain_path=None):
    """Download and unzip using a verified SSL certificate if provided."""
    if os.path.exists(extract_to):
        print(f"The directory '{extract_to}' already exists. Skipping download.")
        return

    local_zip = os.path.basename(url)
    print(f"Downloading {local_zip}...")
    try:
        kwargs = {'stream': True, 'timeout': 20}
        if chain_path:
            kwargs['verify'] = chain_path
            
        response = requests.get(url, **kwargs)
        response.raise_for_status()
        with open(local_zip, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Extracting to '{extract_to}'...")
        os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(local_zip, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        os.remove(local_zip)
        print(f"Extraction completed.")

    except Exception as e:
        print(f"Download/Extraction error: {e}")

def open_tiff_file(name: str) -> np.ndarray:
    """Load multi-frame TIFF file."""
    if not os.path.exists(name):
        print(f"File not found: {name}")
        return None
    img = Image.open(name)
    frames = []
    for i in range(img.n_frames):
        img.seek(i)
        frames.append(np.array(img))
    return np.array(frames).squeeze()

def plot_training_history(history):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Dice score
    axes[1].plot(history['val_dice'], label='Val Dice', marker='o', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].set_title('Validation Dice Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
