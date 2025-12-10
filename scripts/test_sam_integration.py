import numpy as np
import torch
from modules.sam_detector import SAM3Detector
import matplotlib.pyplot as plt

def test_sam():
    print("Initializing SAM 3 Detector...")
    detector = SAM3Detector()
    
    if detector.model is None:
        print("Failed to initialize SAM 3. Exiting.")
        return

    # Create a dummy image (black background, white circle)
    print("Creating dummy image...")
    image = np.zeros((512, 512), dtype=np.uint8)
    y, x = np.ogrid[:512, :512]
    mask = (x - 256)**2 + (y - 256)**2 <= 50**2
    image[mask] = 255
    
    # Add some noise
    image = image + np.random.normal(0, 10, image.shape).astype(np.uint8)
    image = np.clip(image, 0, 255).astype(np.uint8)

    print("Running detection...")
    mask_pred, detections = detector.detect(image, text_prompt="circle")
    
    print(f"Found {len(detections)} detections.")
    print(f"Centroids: {detections}")
    
    if len(detections) > 0:
        print("Test PASSED: Detections found.")
    else:
        print("Test FAILED: No detections found.")
        
    # Visualize
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Input Image")
    plt.subplot(1, 2, 2)
    plt.imshow(mask_pred, cmap='gray')
    plt.title("SAM 3 Prediction")
    plt.savefig("sam3_test_result.png")
    print("Saved visualization to sam3_test_result.png")

if __name__ == "__main__":
    test_sam()
