import matplotlib.pyplot as plt
import pandas as pd
import skimage.io
import os

val_tif_path = "val_data/val.tif"
val_csv_path = "val_data/val.csv"
save_path = "val_data_viz.png"

def visualize_validation_data():
    if not os.path.exists(val_tif_path) or not os.path.exists(val_csv_path):
        print("Validation data not found.")
        return

    print(f"Loading {val_tif_path}...")
    images = skimage.io.imread(val_tif_path)
    print(f"Loaded images shape: {images.shape}")
    
    print(f"Loading {val_csv_path}...")
    df = pd.read_csv(val_csv_path)
    
    # Select a few frames to visualize
    frames_to_show = [0, 10, 20, 30]
    frames_to_show = [f for f in frames_to_show if f < images.shape[0]]
    
    if not frames_to_show:
        print("No frames to show.")
        return

    fig, axes = plt.subplots(1, len(frames_to_show), figsize=(4 * len(frames_to_show), 4))
    if len(frames_to_show) == 1:
        axes = [axes]
        
    for i, frame_idx in enumerate(frames_to_show):
        ax = axes[i]
        img = images[frame_idx]
        
        # Filter annotations for this frame
        frame_data = df[df['frame'] == frame_idx]
        
        ax.imshow(img, cmap='gray')
        ax.scatter(frame_data['x'], frame_data['y'], c='red', s=10, marker='x', label='Ground Truth')
        ax.set_title(f"Frame {frame_idx}")
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    plt.close(fig)

if __name__ == "__main__":
    visualize_validation_data()
