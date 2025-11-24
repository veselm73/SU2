import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import os
from modules.dataset import SyntheticCCPDataset
from modules.config import PATCH_SIZE, SIM_CONFIG, MIN_CELLS, MAX_CELLS, RADIUS

def analyze_data_similarity():
    """
    Comprehensive analysis comparing synthetic and real validation data.
    Provides statistical analysis and visual comparison.
    """
    # 1. Load Real Data
    val_tif_path = "val_data/val.tif"
    if not os.path.exists(val_tif_path):
        print("Validation data not found.")
        return
    
    real_images = skimage.io.imread(val_tif_path)
    print(f"Real data shape: {real_images.shape}")
    
    # Sample multiple frames and patches
    num_samples = 6
    real_patches = []
    
    for i in range(num_samples):
        frame_idx = np.random.randint(0, real_images.shape[0])
        img = real_images[frame_idx]
        h, w = img.shape
        
        # Random crop
        y = np.random.randint(0, h - PATCH_SIZE)
        x = np.random.randint(0, w - PATCH_SIZE)
        patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
        
        # Normalize
        patch_norm = (patch - patch.min()) / (patch.max() - patch.min() + 1e-6)
        real_patches.append(patch_norm)
    
    # 2. Generate Synthetic Data
    print(f"Generating synthetic data with MIN_CELLS={MIN_CELLS}, MAX_CELLS={MAX_CELLS}, RADIUS={RADIUS}")
    dataset = SyntheticCCPDataset(min_n=MIN_CELLS, max_n=MAX_CELLS, radius=RADIUS, 
                                  patch_size=PATCH_SIZE, sim_config=SIM_CONFIG)
    
    syn_patches = []
    for i in range(num_samples):
        syn_img, syn_mask = dataset.data_sample()
        syn_img_norm = (syn_img - syn_img.min()) / (syn_img.max() - syn_img.min() + 1e-6)
        syn_patches.append(syn_img_norm)
    
    # 3. Statistical Analysis
    print("\n" + "="*60)
    print("STATISTICAL COMPARISON (Normalized 0-1)")
    print("="*60)
    
    real_stats = {
        'mean': np.mean([p.mean() for p in real_patches]),
        'std': np.mean([p.std() for p in real_patches]),
        'min': np.mean([p.min() for p in real_patches]),
        'max': np.mean([p.max() for p in real_patches]),
        'median': np.mean([np.median(p) for p in real_patches]),
    }
    
    syn_stats = {
        'mean': np.mean([p.mean() for p in syn_patches]),
        'std': np.mean([p.std() for p in syn_patches]),
        'min': np.mean([p.min() for p in syn_patches]),
        'max': np.mean([p.max() for p in syn_patches]),
        'median': np.mean([np.median(p) for p in syn_patches]),
    }
    
    print(f"\nReal Data:")
    for key, val in real_stats.items():
        print(f"  {key:10s}: {val:.4f}")
    
    print(f"\nSynthetic Data:")
    for key, val in syn_stats.items():
        print(f"  {key:10s}: {val:.4f}")
    
    print(f"\nDifferences:")
    for key in real_stats.keys():
        diff = syn_stats[key] - real_stats[key]
        pct = (diff / real_stats[key] * 100) if real_stats[key] != 0 else 0
        print(f"  {key:10s}: {diff:+.4f} ({pct:+.1f}%)")
    
    # 4. Texture Analysis (Frequency Domain)
    print("\n" + "="*60)
    print("FREQUENCY DOMAIN ANALYSIS")
    print("="*60)
    
    real_fft_power = np.mean([np.abs(np.fft.fft2(p))**2 for p in real_patches], axis=0)
    syn_fft_power = np.mean([np.abs(np.fft.fft2(p))**2 for p in syn_patches], axis=0)
    
    # Radial average
    def radial_profile(data):
        y, x = np.indices(data.shape)
        center = np.array(data.shape) // 2
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
        tbin = np.bincount(r.ravel(), data.ravel())
        nr = np.bincount(r.ravel())
        return tbin / nr
    
    real_radial = radial_profile(np.fft.fftshift(real_fft_power))
    syn_radial = radial_profile(np.fft.fftshift(syn_fft_power))
    
    # 5. Visualization
    fig = plt.figure(figsize=(18, 12))
    
    # Sample images (2 rows x 3 cols)
    for i in range(3):
        # Real samples
        ax = plt.subplot(4, 3, i + 1)
        ax.imshow(real_patches[i], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"Real Sample {i+1}")
        ax.axis('off')
        
        # Synthetic samples
        ax = plt.subplot(4, 3, i + 4)
        ax.imshow(syn_patches[i], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"Synthetic Sample {i+1}")
        ax.axis('off')
    
    # Histograms
    ax = plt.subplot(4, 3, 7)
    all_real = np.concatenate([p.ravel() for p in real_patches])
    all_syn = np.concatenate([p.ravel() for p in syn_patches])
    ax.hist(all_real, bins=50, alpha=0.7, label='Real', color='blue', density=True)
    ax.hist(all_syn, bins=50, alpha=0.7, label='Synthetic', color='green', density=True)
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Density')
    ax.set_title('Intensity Distribution')
    ax.legend()
    ax.set_xlim(0, 1)
    
    # Cumulative distribution
    ax = plt.subplot(4, 3, 8)
    ax.hist(all_real, bins=100, alpha=0.7, label='Real', color='blue', 
            density=True, cumulative=True, histtype='step', linewidth=2)
    ax.hist(all_syn, bins=100, alpha=0.7, label='Synthetic', color='green', 
            density=True, cumulative=True, histtype='step', linewidth=2)
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Frequency domain comparison
    ax = plt.subplot(4, 3, 9)
    freq_range = min(len(real_radial), len(syn_radial), 50)
    ax.semilogy(real_radial[:freq_range], label='Real', linewidth=2)
    ax.semilogy(syn_radial[:freq_range], label='Synthetic', linewidth=2)
    ax.set_xlabel('Spatial Frequency (pixels)')
    ax.set_ylabel('Power (log scale)')
    ax.set_title('Radial Power Spectrum')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # FFT visualization
    ax = plt.subplot(4, 3, 10)
    ax.imshow(np.log1p(np.fft.fftshift(real_fft_power)), cmap='viridis')
    ax.set_title('Real FFT Power')
    ax.axis('off')
    
    ax = plt.subplot(4, 3, 11)
    ax.imshow(np.log1p(np.fft.fftshift(syn_fft_power)), cmap='viridis')
    ax.set_title('Synthetic FFT Power')
    ax.axis('off')
    
    # Difference map
    ax = plt.subplot(4, 3, 12)
    diff_fft = np.log1p(np.fft.fftshift(syn_fft_power)) - np.log1p(np.fft.fftshift(real_fft_power))
    im = ax.imshow(diff_fft, cmap='RdBu_r', vmin=-np.abs(diff_fft).max(), vmax=np.abs(diff_fft).max())
    ax.set_title('FFT Difference (Syn - Real)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    plt.savefig("detailed_comparison.png", dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to detailed_comparison.png")
    plt.close(fig)
    
    # 6. Generate Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR IMPROVEMENT")
    print("="*60)
    
    recommendations = []
    
    # Mean intensity
    if syn_stats['mean'] < real_stats['mean'] * 0.9:
        recommendations.append("• Increase foreground/background contrast in SIM simulation")
        recommendations.append("  → Adjust contrast_fg_range and contrast_bg_range in SyntheticCCPDataset")
    
    # Standard deviation
    if syn_stats['std'] < real_stats['std'] * 0.9:
        recommendations.append("• Increase intensity variation")
        recommendations.append("  → Add more noise or increase Perlin noise amplitude")
    
    # Cell density
    avg_cells = (MIN_CELLS + MAX_CELLS) / 2
    recommendations.append(f"• Current cell count: {MIN_CELLS}-{MAX_CELLS} (avg {avg_cells:.1f})")
    recommendations.append("  → Analyze real data cell density to match better")
    
    # Radius
    recommendations.append(f"• Current cell radius: {RADIUS}")
    recommendations.append("  → Measure actual cell sizes in validation data")
    
    # Texture
    if np.mean(syn_radial[5:15]) < np.mean(real_radial[5:15]) * 0.8:
        recommendations.append("• Synthetic data lacks mid-frequency texture")
        recommendations.append("  → Increase Perlin noise resolution or add additional texture layers")
    
    for rec in recommendations:
        print(rec)
    
    return real_stats, syn_stats, recommendations

if __name__ == "__main__":
    analyze_data_similarity()
