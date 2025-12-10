import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import os
from modules.dataset import SyntheticCCPDataset
from modules.config import PATCH_SIZE, SIM_CONFIG, MIN_CELLS, MAX_CELLS, RADIUS

def compare_filled_cells():
    """
    Compare new filled cell synthetic data with validation data.
    Shows before/after comparison and statistical differences.
    """
    # 1. Load Real Data
    val_tif_path = "val_data/val.tif"
    if not os.path.exists(val_tif_path):
        print("Validation data not found.")
        return
    
    real_images = skimage.io.imread(val_tif_path)
    print(f"Real data shape: {real_images.shape}")
    
    # Sample real patches
    num_samples = 4
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
    
    # 2. Generate NEW Synthetic Data (filled cells)
    print(f"\nGenerating FILLED synthetic cells...")
    print(f"Parameters: MIN_CELLS={MIN_CELLS}, MAX_CELLS={MAX_CELLS}, RADIUS={RADIUS}")
    
    dataset = SyntheticCCPDataset(min_n=MIN_CELLS, max_n=MAX_CELLS, radius=RADIUS, 
                                  patch_size=PATCH_SIZE, sim_config=SIM_CONFIG)
    
    syn_patches = []
    syn_masks = []
    for i in range(num_samples):
        syn_img, syn_mask = dataset.data_sample()
        syn_img_norm = (syn_img - syn_img.min()) / (syn_img.max() - syn_img.min() + 1e-6)
        syn_patches.append(syn_img_norm)
        syn_masks.append(syn_mask)
    
    # 3. Statistical Analysis
    print("\n" + "="*70)
    print("STATISTICAL COMPARISON (Normalized 0-1)")
    print("="*70)
    
    real_stats = {
        'mean': np.mean([p.mean() for p in real_patches]),
        'std': np.mean([p.std() for p in real_patches]),
        'median': np.mean([np.median(p) for p in real_patches]),
        'q25': np.mean([np.percentile(p, 25) for p in real_patches]),
        'q75': np.mean([np.percentile(p, 75) for p in real_patches]),
    }
    
    syn_stats = {
        'mean': np.mean([p.mean() for p in syn_patches]),
        'std': np.mean([p.std() for p in syn_patches]),
        'median': np.mean([np.median(p) for p in syn_patches]),
        'q25': np.mean([np.percentile(p, 25) for p in syn_patches]),
        'q75': np.mean([np.percentile(p, 75) for p in syn_patches]),
    }
    
    print(f"\n{'Metric':<12} {'Real':>10} {'Synthetic':>10} {'Difference':>12} {'% Diff':>10}")
    print("-" * 70)
    for key in real_stats.keys():
        diff = syn_stats[key] - real_stats[key]
        pct = (diff / real_stats[key] * 100) if real_stats[key] != 0 else 0
        print(f"{key:<12} {real_stats[key]:>10.4f} {syn_stats[key]:>10.4f} {diff:>+12.4f} {pct:>+9.1f}%")
    
    # 4. Visualization
    fig = plt.figure(figsize=(16, 12))
    
    # Row 1: Real samples
    for i in range(num_samples):
        ax = plt.subplot(4, num_samples, i + 1)
        ax.imshow(real_patches[i], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"Real {i+1}", fontsize=10)
        ax.axis('off')
    
    # Row 2: Synthetic samples (NEW - filled cells)
    for i in range(num_samples):
        ax = plt.subplot(4, num_samples, i + num_samples + 1)
        ax.imshow(syn_patches[i], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"Synthetic {i+1}\n(Filled)", fontsize=10)
        ax.axis('off')
    
    # Row 3: Synthetic masks
    for i in range(num_samples):
        ax = plt.subplot(4, num_samples, i + 2*num_samples + 1)
        ax.imshow(syn_masks[i], cmap='hot', vmin=0, vmax=1)
        ax.set_title(f"Mask {i+1}", fontsize=10)
        ax.axis('off')
    
    # Row 4: Analysis plots
    # Histogram comparison
    ax = plt.subplot(4, num_samples, 2*num_samples + num_samples + 1)
    all_real = np.concatenate([p.ravel() for p in real_patches])
    all_syn = np.concatenate([p.ravel() for p in syn_patches])
    ax.hist(all_real, bins=50, alpha=0.6, label='Real', color='blue', density=True)
    ax.hist(all_syn, bins=50, alpha=0.6, label='Synthetic', color='green', density=True)
    ax.set_xlabel('Intensity', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title('Intensity Distribution', fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.tick_params(labelsize=8)
    
    # Cumulative distribution
    ax = plt.subplot(4, num_samples, 2*num_samples + num_samples + 2)
    ax.hist(all_real, bins=100, alpha=0.7, label='Real', color='blue', 
            density=True, cumulative=True, histtype='step', linewidth=2)
    ax.hist(all_syn, bins=100, alpha=0.7, label='Synthetic', color='green', 
            density=True, cumulative=True, histtype='step', linewidth=2)
    ax.set_xlabel('Intensity', fontsize=9)
    ax.set_ylabel('Cumulative', fontsize=9)
    ax.set_title('CDF Comparison', fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)
    
    # Box plot comparison
    ax = plt.subplot(4, num_samples, 2*num_samples + num_samples + 3)
    data_to_plot = [all_real, all_syn]
    bp = ax.boxplot(data_to_plot, labels=['Real', 'Synthetic'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightgreen')
    ax.set_ylabel('Intensity', fontsize=9)
    ax.set_title('Distribution Comparison', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(labelsize=8)
    
    # Difference metrics
    ax = plt.subplot(4, num_samples, 2*num_samples + num_samples + 4)
    metrics = list(real_stats.keys())
    differences = [(syn_stats[k] - real_stats[k]) / real_stats[k] * 100 for k in metrics]
    colors = ['red' if d < 0 else 'green' for d in differences]
    ax.barh(metrics, differences, color=colors, alpha=0.6)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('% Difference', fontsize=9)
    ax.set_title('Synthetic vs Real\n(% Difference)', fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    ax.tick_params(labelsize=8)
    
    plt.suptitle('Filled Cells: Synthetic vs Validation Data Comparison', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig("filled_cells_comparison.png", dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to filled_cells_comparison.png")
    plt.close(fig)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"[OK] Modified cell generation from RINGS to FILLED structures")
    print(f"[OK] Using Gaussian decay from cell centers")
    print(f"[OK] Cell parameters: radius={RADIUS}, count={MIN_CELLS}-{MAX_CELLS}")
    
    # Check similarity
    mean_diff_pct = abs((syn_stats['mean'] - real_stats['mean']) / real_stats['mean'] * 100)
    std_diff_pct = abs((syn_stats['std'] - real_stats['std']) / real_stats['std'] * 100)
    
    print(f"\nSimilarity Assessment:")
    print(f"  Mean intensity difference: {mean_diff_pct:.1f}%")
    print(f"  Std deviation difference: {std_diff_pct:.1f}%")
    
    if mean_diff_pct < 10 and std_diff_pct < 20:
        print(f"  [GOOD] Synthetic data closely matches validation data")
    elif mean_diff_pct < 20 and std_diff_pct < 30:
        print(f"  [FAIR] Synthetic data reasonably matches validation data")
    else:
        print(f"  [NEEDS IMPROVEMENT] Consider adjusting parameters")
    
    return real_stats, syn_stats

if __name__ == "__main__":
    compare_filled_cells()
