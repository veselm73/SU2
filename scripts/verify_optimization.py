import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import os
from modules.dataset import SyntheticCCPDataset
from modules.config import PATCH_SIZE, SIM_CONFIG, MIN_CELLS, MAX_CELLS, RADIUS

# Quick comparison script
val_tif_path = "val_data/val.tif"
real_images = skimage.io.imread(val_tif_path)

# Sample real patches
num_samples = 6
real_patches = []
for i in range(num_samples):
    frame_idx = np.random.randint(0, real_images.shape[0])
    img = real_images[frame_idx]
    h, w = img.shape
    y = np.random.randint(0, h - PATCH_SIZE)
    x = np.random.randint(0, w - PATCH_SIZE)
    patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
    patch_norm = (patch - patch.min()) / (patch.max() - patch.min() + 1e-6)
    real_patches.append(patch_norm)

# Generate synthetic
dataset = SyntheticCCPDataset(min_n=MIN_CELLS, max_n=MAX_CELLS, radius=RADIUS, 
                              patch_size=PATCH_SIZE, sim_config=SIM_CONFIG)

syn_patches = []
for i in range(num_samples):
    syn_img, _ = dataset.data_sample()
    syn_img_norm = (syn_img - syn_img.min()) / (syn_img.max() - syn_img.min() + 1e-6)
    syn_patches.append(syn_img_norm)

# Stats
real_mean = np.mean([p.mean() for p in real_patches])
real_std = np.mean([p.std() for p in real_patches])
syn_mean = np.mean([p.mean() for p in syn_patches])
syn_std = np.mean([p.std() for p in syn_patches])

print("="*60)
print("OPTIMIZED PARAMETERS COMPARISON")
print("="*60)
print(f"Real Data:      Mean={real_mean:.4f}, Std={real_std:.4f}")
print(f"Synthetic Data: Mean={syn_mean:.4f}, Std={syn_std:.4f}")
print(f"Differences:    Mean={syn_mean-real_mean:+.4f} ({(syn_mean-real_mean)/real_mean*100:+.1f}%)")
print(f"                Std={syn_std-real_std:+.4f} ({(syn_std-real_std)/real_std*100:+.1f}%)")

# Visual comparison
fig, axes = plt.subplots(2, num_samples, figsize=(15, 5))
for i in range(num_samples):
    axes[0, i].imshow(real_patches[i], cmap='gray', vmin=0, vmax=1)
    axes[0, i].set_title(f"Real {i+1}", fontsize=9)
    axes[0, i].axis('off')
    
    axes[1, i].imshow(syn_patches[i], cmap='gray', vmin=0, vmax=1)
    axes[1, i].set_title(f"Syn {i+1}", fontsize=9)
    axes[1, i].axis('off')

plt.suptitle('Optimized Synthetic Data vs Validation Data', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("optimized_comparison.png", dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to optimized_comparison.png")
plt.close()

# Assessment
mean_diff_pct = abs((syn_mean - real_mean) / real_mean * 100)
std_diff_pct = abs((syn_std - real_std) / real_std * 100)

print(f"\nSimilarity Assessment:")
if mean_diff_pct < 10 and std_diff_pct < 15:
    print(f"  [EXCELLENT] Synthetic data closely matches validation data!")
    print(f"  Mean diff: {mean_diff_pct:.1f}%, Std diff: {std_diff_pct:.1f}%")
elif mean_diff_pct < 20 and std_diff_pct < 25:
    print(f"  [GOOD] Synthetic data reasonably matches validation data")
    print(f"  Mean diff: {mean_diff_pct:.1f}%, Std diff: {std_diff_pct:.1f}%")
else:
    print(f"  [NEEDS MORE WORK] Consider further adjustments")
    print(f"  Mean diff: {mean_diff_pct:.1f}%, Std diff: {std_diff_pct:.1f}%")
