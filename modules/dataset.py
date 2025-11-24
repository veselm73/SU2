import numpy as np
import torch
import torch.utils.data as torch_data
from .simulation import SyntheticDataset
from .simulation import SyntheticDataset
from .config import MIN_CELLS, MAX_CELLS, PATCH_SIZE, SIM_CONFIG, RADIUS
import matplotlib.pyplot as plt

class SyntheticCCPDataset(SyntheticDataset):
    """Synthetic CCP dataset with ground truth masks."""
    def __init__(self, min_n=MIN_CELLS, max_n=MAX_CELLS, radius=RADIUS, 
                 contrast_fg_range=(0.4, 1.2), contrast_bg_range=(0.2, 0.5), 
                 patch_size=PATCH_SIZE, sim_config=SIM_CONFIG):
        super().__init__(contrast_fg_range, contrast_bg_range, patch_size=patch_size, sim_config=sim_config)
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
        
        # Calculate distance from each cell center
        distance = np.hypot(self.yyy[..., None] - positions[:, 0], self.xxx[..., None] - positions[:, 1])
        
        # Create filled cells using Gaussian-like decay instead of rings
        # Each cell has intensity that decreases from center
        cell_radius = classes * self.radius
        
        # Gaussian-like falloff: intensity = exp(-(distance/radius)^2)
        # Adding some variation with the thickness parameter for edge sharpness
        sigma = cell_radius / 2.0  # Controls how quickly intensity falls off
        parts = np.exp(-0.5 * (distance / (sigma + 1e-6))**2)
        
        # Apply threshold to create defined edges (optional, adjust for softer/harder edges)
        parts = np.where(distance < cell_radius * 1.5, parts, 0)
        
        # Combine all cells
        full_image = np.sum(parts, -1)
        
        # Create mask for training (filled circles)
        distances = np.maximum(1.0 - distance / (cell_radius + 1e-6), 0)
        y = np.minimum(np.sum(distances, -1), 1)
        
        x = super()._simulate_sim(full_image)

        return x, y

class CCPDatasetWrapper(torch_data.Dataset):
    """
    PyTorch Dataset that yields synthetic CCP images and masks on-the-fly.
    """
    def __init__(self, length=500, min_n=MIN_CELLS, max_n=MAX_CELLS, patch_size=PATCH_SIZE, sim_config=SIM_CONFIG):
        super().__init__()
        self.length = length
        self._synthetic = SyntheticCCPDataset(min_n=min_n, max_n=max_n, patch_size=patch_size, sim_config=sim_config)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img, mask = self._synthetic.data_sample()
        
        # Data Augmentation
        # 1. Random Horizontal Flip
        if np.random.rand() > 0.5:
            img = np.fliplr(img)
            mask = np.fliplr(mask)
            
        # 2. Random Vertical Flip
        if np.random.rand() > 0.5:
            img = np.flipud(img)
            mask = np.flipud(mask)
            
        # 3. Random 90-degree Rotation
        k = np.random.randint(0, 4)
        if k > 0:
            img = np.rot90(img, k)
            mask = np.rot90(mask, k)
            
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        
        # Ensure correct copy for torch
        img = img.copy()
        mask = mask.copy()
        
        img_tensor = torch.from_numpy(img).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        return img_tensor, mask_tensor

def visualize_generated_data(dataset, num_samples=4, save_path=None):
    """
    Generates and plots a few samples from the dataset.
    """
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)
        
    for i in range(num_samples):
        img, mask = dataset[i]
        
        # Convert tensor to numpy if needed
        if isinstance(img, torch.Tensor):
            img = img.squeeze().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.squeeze().numpy()
            
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f"Sample {i+1} - Image")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title(f"Sample {i+1} - Mask")
        axes[i, 1].axis('off')
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)
