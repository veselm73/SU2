import os
import sys
import warnings
import requests
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import json
from dataclasses import dataclass
from datetime import datetime
import time
import subprocess
import threading
import random
import math
import numpy as np
import pandas as pd
from scipy import ndimage, spatial, optimize
from scipy.ndimage import gaussian_filter, maximum_filter
from skimage import feature, filters, measure, morphology, exposure
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torch_data
from torch.utils.data import Dataset, DataLoader, IterableDataset, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import cv2
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import btrack

# Configuration
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ============================================================================
# DATA DOWNLOAD FUNCTIONS
# ============================================================================
def download_and_unzip(url, extract_to, chain_path):
    if os.path.exists(extract_to):
        print(f"The directory '{extract_to}' already exists. Skipping download.")
        return

    local_zip = os.path.basename(url)
    print(f"Downloading {local_zip}...")
    try:
        response = requests.get(url, stream=True, verify=chain_path, timeout=20)
        response.raise_for_status()
        with open(local_zip, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(local_zip, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        os.remove(local_zip)
        print(f"Extraction completed to '{extract_to}'")

    except Exception as e:
        print(f"Error: {e}")

# ============================================================================
# SIM RECONSTRUCTION CLASSES
# ============================================================================
class OTF:
    def __init__(self, na: float, wavelength: float, pixel_size: float,
                 image_size: int, curvature: float):
        cutoff_frequency = 1000 * 2 * na / wavelength
        self.image_cutoff = cutoff_frequency * pixel_size * image_size
        self.image_size = image_size
        self.curvature = curvature

    def __call__(self, size: int = None, x_shift: float = 0, y_shift: float = 0) -> np.ndarray:
        if size is None:
            size = self.image_size
        x, y = np.meshgrid(np.hstack([np.arange(size // 2), np.arange(-size // 2, 0)]),
                           np.hstack([np.arange(size // 2), np.arange(-size // 2, 0)]))
        distance_to_origin = np.hypot(x + x_shift, y + y_shift)
        return self.value(np.minimum(distance_to_origin / self.image_cutoff, 1))

    def value(self, x):
        return (2 / np.pi) * (np.arccos(x) - x * np.sqrt(1 - x * x)) * self.curvature ** x

def illumination_pattern(angle, frequency, phase_offset, amplitude, size) -> np.ndarray:
    n = size // 2
    Y, X = np.mgrid[-n:n, -n:n]
    ky, kx = np.sin(angle) * frequency, np.cos(angle) * frequency
    return 1 + amplitude * np.cos(2 * np.pi * (X * kx + Y * ky) + phase_offset)

class PerlinNoise:
    def __init__(self, size: int, res: int):
        meshgrid = np.mgrid[0:res:res / size, 0:res:res / size]
        self.grid = np.stack(meshgrid) % 1
        self.t = self._fade(self.grid)
        self.d = size // res
        self.sample_size = (res + 1, res + 1)

    def __call__(self) -> np.array:
        angles = 2 * np.pi * np.random.random_sample(self.sample_size)
        gradients = np.dstack((np.cos(angles), np.sin(angles))).repeat(self.d, 0).repeat(self.d, 1)
        n00 = (np.dstack((self.grid[0], self.grid[1])) * gradients[:-self.d, :-self.d]).sum(2)
        n10 = (np.dstack((self.grid[0] - 1, self.grid[1])) * gradients[self.d:, :-self.d]).sum(2)
        n01 = (np.dstack((self.grid[0], self.grid[1] - 1)) * gradients[:-self.d, self.d:]).sum(2)
        n11 = (np.dstack((self.grid[0] - 1, self.grid[1] - 1)) * gradients[self.d:, self.d:]).sum(2)
        n0 = n00 * (1 - self.t[0]) + n10 * self.t[0]
        n1 = n01 * (1 - self.t[1]) + n11 * self.t[1]
        return 0.5 + 2 ** -0.5 * (n0 * (1 - self.t[1]) + n1 * self.t[1])

    @staticmethod
    def _fade(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

# Helper functions for SIM (simplified for script)
def run_reconstruction(fft_images, otf, shifts, phase_offsets, modulations, config):
    # Placeholder for full reconstruction logic if needed, but for training data generation
    # we need the full logic. I'll include minimal necessary parts.
    # Actually, SyntheticDataset uses run_reconstruction. I need to include it.
    pass 
    # Wait, I need the full code for run_reconstruction and dependencies.
    # I will paste them below.

def _component_separation_matrix(phase_offset: float = 0, modulation: float = 1) -> np.ndarray:
    phases = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3]) + phase_offset
    M = np.ones((3, 3), dtype=np.complex128)
    M[:, 1] = 0.5 * modulation * (np.cos(phases) + 1j * np.sin(phases))
    M[:, 2] = 0.5 * modulation * (np.cos(-phases) + 1j * np.sin(-phases))
    return np.linalg.inv(M)

def separate_components(fft_images: np.ndarray, phase_offsets=np.zeros(3), phase_modulations=np.ones(3)) -> np.ndarray:
    separation_matrices = np.array([_component_separation_matrix(phase_offsets[i], phase_modulations[i]) for i in range(3)])
    return np.einsum('aij,ajkl->aikl', separation_matrices, fft_images.reshape(3, 3, *fft_images.shape[1:])).reshape(9, *fft_images.shape[1:])

def fourier_shift(fft_image: np.ndarray, x_shift: Union[float, np.ndarray], y_shift: Union[float, np.ndarray]) -> np.ndarray:
    height, width = fft_image.shape[-2:]
    spatial_image = np.fft.ifft2(fft_image)
    x_indices = np.arange(-width // 2, width // 2, dtype=int)
    y_indices = np.arange(-height // 2, height // 2, dtype=int)
    x = np.exp(-1j * 2 * np.pi * x_shift * x_indices[None, :] / width)
    y = np.exp(-1j * 2 * np.pi * y_shift * y_indices[:, None] / height)
    shifted_spatial_image = spatial_image * (x * y)
    return np.fft.fft2(shifted_spatial_image)

def shift_components(components: np.ndarray, shifts) -> None:
    for i in range(3):
        components[i * 3 + 1, ...] = fourier_shift(components[i * 3 + 1, ...], shifts[i][1], shifts[i][0])
        components[i * 3 + 2, ...] = fourier_shift(components[i * 3 + 2, ...], -shifts[i][1], -shifts[i][0])

def pad_components(components: np.ndarray) -> np.ndarray:
    size = components.shape[-1]
    padded_components = np.zeros(components.shape[:-2] + (size * 2, size * 2), dtype=np.complex128)
    x, y = np.meshgrid(np.hstack([np.arange(size // 2), np.arange(size * 3 // 2, size * 2)]),
                       np.hstack([np.arange(size // 2), np.arange(size * 3 // 2, size * 2)]))
    padded_components[..., y, x] = components
    return padded_components

def map_otf_support(components: np.ndarray, shifts, otf: OTF) -> None:
    size = components.shape[-1]
    components[::3] *= np.conjugate(otf(size))
    for i in range(3):
        components[i * 3 + 1] *= np.conjugate(otf(size, shifts[i][1], shifts[i][0]))
        components[i * 3 + 2] *= np.conjugate(otf(size, -shifts[i][1], -shifts[i][0]))

def wiener_filter(shifts, otf: OTF, w: float, size: int) -> np.ndarray:
    otf0 = np.abs(otf(size)) ** 2
    wiener = 3 * otf0
    for i in range(3):
        otf1 = np.abs(otf(size, shifts[i][1], shifts[i][0])) ** 2
        otf2 = np.abs(otf(size, -shifts[i][1], -shifts[i][0])) ** 2
        wiener += otf1 + otf2
    return 1 / (wiener + w * w)

def apodization_filter(dist_ratio: float, bend: float, size: int) -> np.ndarray:
    x, y = np.meshgrid(np.hstack([np.arange(size // 2), np.arange(-size // 2, 0)]),
                       np.hstack([np.arange(size // 2), np.arange(-size // 2, 0)]))
    distance = np.hypot(x, y) * dist_ratio
    mask = np.bitwise_and(0 <= distance, distance < 1)
    apo = np.power((2 / np.pi) * (np.arccos(distance, where=mask) - distance * np.sqrt(1 - distance * distance, where=mask)), bend, where=mask)
    return np.where(mask, apo, 0)

def run_reconstruction(fft_images: np.ndarray, otf: OTF, shifts, phase_offsets, modulations, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    size = fft_images.shape[-1]
    cutoff = 1000 * 2 * config["na"] / config["wavelength"]
    apo_dist_ratio = 1 / (config["px_size"] * config["apo_cutoff"] * cutoff * size)
    components = separate_components(fft_images, phase_offsets, modulations)
    components = pad_components(components)
    shift_components(components, shifts)
    map_otf_support(components, shifts, otf)
    wiener = wiener_filter(shifts, otf, config["wiener_parameter"], size * 2)
    apodization = apodization_filter(apo_dist_ratio, config["apo_bend"], size * 2)
    fft_result = np.sum(components, 0) * wiener * apodization
    spatial_result = np.real(np.fft.ifft2(fft_result))
    return fft_result, spatial_result

# ============================================================================
# DATASETS
# ============================================================================
class SyntheticDataset(IterableDataset):
    def __init__(self, contrast_fg_range=(0.0, 1.0), contrast_bg_range=(0.0, 1.0)):
        self.patch_size = 128
        self.contrast_fg_range = contrast_fg_range
        self.contrast_bg_range = contrast_bg_range
        self.frequency = 0.17
        self.amplitude = 1.0
        self.config = {
            "na": 1.49, "wavelength": 512, "px_size": 0.07,
            "wiener_parameter": 0.1, "apo_cutoff": 2.0, "apo_bend": 0.9
        }
        self.otf = OTF(self.config['na'], self.config['wavelength'], self.config['px_size'], self.patch_size // 2, 0.3)
        self.otf_mult = self.otf(self.patch_size)
        self.perlin = PerlinNoise(self.patch_size, 1)

    def _simulate_sim(self, image):
        angle0 = np.random.uniform(0, np.pi * 2)
        phase_offsets = np.random.uniform(0, np.pi * 2, 3)
        shifts = [(self.frequency * self.patch_size * np.sin(angle0 + i * np.pi / 3),
                   self.frequency * self.patch_size * np.cos(angle0 + i * np.pi / 3)) for i in range(3)]
        illumination = np.stack([illumination_pattern(angle0 + i // 3 * np.pi / 3, self.frequency,
                                                      phase_offsets[i // 3] + (i % 3) * np.pi * 2 / 3,
                                                      self.amplitude, self.patch_size) for i in range(9)])
        fg_c = np.random.uniform(*self.contrast_fg_range)
        bg_c = np.random.uniform(*self.contrast_bg_range)
        foreground = 250 + fg_c * 500
        background = 50 + bg_c * 50
        high_res_image = (image * foreground + background) * self.perlin()
        ix = np.fft.fft2(illumination * high_res_image)
        hix = self.otf_mult * ix
        dhix = hix.reshape(9, 2, self.patch_size // 2, 2, self.patch_size // 2).sum((1, 3)) / 4
        poisson_input = np.maximum(0, np.fft.ifft2(dhix).real)
        low_res_images = np.random.poisson(poisson_input).astype(np.float64)
        noisy_shifts = [np.random.triangular((y - 0.25, x - 0.25), (y, x), (y + 0.25, x + 0.25)) for y, x in shifts]
        noisy_phase_offsets = np.random.normal(phase_offsets, np.pi / 6)
        noisy_amplitudes = np.random.normal(self.amplitude, 0.1, 3)
        reconstruction = run_reconstruction(np.fft.fft2(low_res_images), self.otf, noisy_shifts,
                                          noisy_phase_offsets, noisy_amplitudes, self.config)[1]
        return (reconstruction - np.mean(reconstruction)) / np.std(reconstruction)

class SyntheticCCPDataset(SyntheticDataset):
    def __init__(self, min_n=5, max_n=15, radius=2.5, contrast_fg_range=(0.0, 1.0), contrast_bg_range=(0.0, 1.0)):
        super().__init__(contrast_fg_range, contrast_bg_range)
        self.min_n, self.max_n = min_n, max_n
        self.radius = radius
        self.thickness = 1.0
        self.max_offset = 8
        self.beta_a, self.beta_b = 2, 1
        yy, xx = np.mgrid[15:self.patch_size - 1:16, 15:self.patch_size - 1:16]
        self.yy = yy.flatten()
        self.xx = xx.flatten()
        self.yyy, self.xxx = np.mgrid[:self.patch_size, :self.patch_size]

    def __iter__(self):
        while True:
            yield self.data_sample()

    def data_sample(self):
        n = np.random.randint(self.min_n, self.max_n)
        indices = np.random.choice(len(self.yy), size=n, replace=False)
        offsets = np.random.uniform(-self.max_offset, self.max_offset, (n, 2))
        positions = np.column_stack([self.yy[indices], self.xx[indices]]) + offsets
        classes = np.random.beta(self.beta_a, self.beta_b, n) * 0.9 + 0.1
        target_distance = classes * self.radius
        distance = np.hypot(self.yyy[..., None] - positions[:, 0], self.xxx[..., None] - positions[:, 1])
        abs_distance = np.abs(distance - target_distance)
        parts = np.where(abs_distance > self.thickness, 0, np.log(np.interp(abs_distance / self.thickness, [0, 1], [np.e, 1])))
        full_image = np.sum(parts, -1)
        distances = np.maximum(classes - distance / ((1 - classes) * 2 + self.radius + self.thickness * 2), 0)
        y = np.minimum(np.sum(distances, -1), 1)
        x = super()._simulate_sim(full_image)
        return x, y

class CCPDatasetWrapper(torch_data.Dataset):
    def __init__(self, length=500):
        super().__init__()
        self.length = length
        self._synthetic = SyntheticCCPDataset()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img, mask = self._synthetic.data_sample()
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        img_tensor = torch.from_numpy(img).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        return img_tensor, mask_tensor

# ============================================================================
# MODEL
# ============================================================================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        if self.dropout is not None:
            x = self.dropout(x)
        return x

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256, 512], use_attention=True, dropout_rate=0.1):
        super(UNetPlusPlus, self).__init__()
        self.use_attention = use_attention
        self.features = features
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i, feature in enumerate(features):
            in_ch = in_channels if i == 0 else features[i-1]
            self.encoders.append(ConvBlock(in_ch, feature, dropout_rate))
            if i < len(features) - 1:
                self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.ups = nn.ModuleList()
        for i in range(len(features) - 1):
            self.ups.append(nn.ConvTranspose2d(features[i + 1], features[i], kernel_size=2, stride=2))
        self.decoder_convs = nn.ModuleDict()
        for i in range(len(features) - 1):
            self.decoder_convs[f"conv_0_{i}"] = ConvBlock(features[i] * 2, features[i], dropout_rate)
        if use_attention:
            self.attention_gates = nn.ModuleDict()
            for i in range(len(features) - 1):
                self.attention_gates[f"att_0_{i}"] = AttentionGate(F_g=features[i], F_l=features[i], F_int=features[i] // 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        encoder_outputs = []
        for i, (encoder, pool) in enumerate(zip(self.encoders[:-1], self.pools)):
            x = encoder(x)
            encoder_outputs.append(x)
            x = pool(x)
        x = self.encoders[-1](x)
        for decoder_level in range(len(self.features) - 2, -1, -1):
            x_up = self.ups[decoder_level](x)
            skip_connection = encoder_outputs[decoder_level]
            if self.use_attention:
                att_key = f"att_0_{decoder_level}"
                if att_key in self.attention_gates:
                    skip_connection = self.attention_gates[att_key](x_up, skip_connection)
            x = torch.cat([x_up, skip_connection], dim=1)
            conv_key = f"conv_0_{decoder_level}"
            if conv_key in self.decoder_convs:
                x = self.decoder_convs[conv_key](x)
        return self.final_conv(x)

# ============================================================================
# NEW: IoU LOSS
# ============================================================================
class IoULoss(nn.Module):
    """
    IoU Loss (Jaccard Loss) for segmentation.
    """
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: logits
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou

# ============================================================================
# TRAINING
# ============================================================================
def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, device='cuda'):
    model = model.to(device)
    
    # CHANGED: Use IoU Loss
    criterion = IoULoss()
    
    # CHANGED: AdamW with weight_decay=1e-4 (10e-5)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Scheduler: ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    history = {'train_loss': [], 'val_loss': [], 'val_dice': []}
    best_val_loss = float('inf')
    
    # Early Stopping
    patience = 5
    early_stopping_counter = 0
    
    print(f"Training U-Net++ with IoU Loss for {epochs} epochs")
    print(f"Device: {device}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for images, masks in train_pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                pred = torch.sigmoid(outputs) > 0.5
                intersection = (pred * masks).sum()
                union = pred.sum() + masks.sum()
                dice = (2. * intersection / (union + 1e-6)).item()
                val_dice += dice
                
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  ✓ New best model (val_loss: {val_loss:.4f})")
            torch.save(model.state_dict(), "best_model.pth")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"  Early stopping counter: {early_stopping_counter}/{patience}")
            if early_stopping_counter >= patience:
                print("Early stopping triggered.")
                break
                
    print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    return model, history

def train_unet_pipeline(train_samples=500, val_samples=100, epochs=20, batch_size=8, learning_rate=1e-3):
    print("Creating train dataset...")
    train_dataset = CCPDatasetWrapper(length=train_samples)
    print("Creating validation dataset...")
    val_dataset = CCPDatasetWrapper(length=val_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print("Creating U-Net++ model...")
    model = UNetPlusPlus(in_channels=1, out_channels=1, features=[32, 64, 128, 256, 512], use_attention=True, dropout_rate=0.1)
    
    model, history = train_model(model, train_loader, val_loader, epochs=epochs, lr=learning_rate, device=device)
    return model, history

if __name__ == "__main__":
    # Prepare for overnight run
    # Download data (optional, if needed for validation, but synthetic data is used for training)
    # chain_path = "chain-harica-cross.pem"
    # if not os.path.exists(chain_path):
    #     # Download cert logic here if needed, but skipping for brevity as synthetic data is main focus
    #     pass

    print("Starting overnight training run...")
    model, history = train_unet_pipeline(
        train_samples=500,
        val_samples=100,
        epochs=200, # Increased for overnight
        batch_size=8,
        learning_rate=1e-3
    )
    torch.save(model.state_dict(), "final_model_overnight.pth")
    print("Run finished.")
