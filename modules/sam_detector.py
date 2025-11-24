import torch
import numpy as np
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from .config import DEVICE, SAM3_CHECKPOINT, SAM3_MODEL_TYPE

class SAM3Detector:
    """
    Wrapper for Segment Anything Model 3 (SAM 3) to be used in the detection pipeline.
    """
    def __init__(self, checkpoint_path=None, model_type=None, device=DEVICE):
        self.device = device
        # SAM 3 seems to handle checkpoints internally via Hugging Face or default paths
        # But we can probably pass a path if needed. 
        # For now, let's rely on the default build_sam3_image_model() which might download or look for checkpoints.
        # If SAM3_CHECKPOINT is set and exists, we might need to pass it.
        # However, the README example just says build_sam3_image_model().
        
        print(f"Loading SAM 3 model...")
        try:
            # Explicitly pass device to avoid internal defaults checking CUDA if we want CPU
            # Convert torch.device to string if needed, though SAM 3 seems to handle string 'cuda'/'cpu'
            device_str = str(device).split(':')[0] # 'cuda' or 'cpu'
            self.model = build_sam3_image_model(device=device_str) 
            self.model.to(device)
            self.processor = Sam3Processor(self.model)
            self.inference_state = None
        except Exception as e:
            print(f"Error loading SAM 3: {e}")
            print("Please ensure you have access to SAM 3 checkpoints and are authenticated with Hugging Face if required.")
            self.model = None
            self.processor = None

    def detect(self, image: np.ndarray, text_prompt="cell"):
        """
        Runs SAM 3 inference on the image using a text prompt.
        
        Args:
            image: Input image (H, W) or (H, W, C). 
            text_prompt: Text prompt to detect objects (e.g., "cell", "white spot").
        
        Returns:
            mask: Binary mask of detected objects.
            detections: List of (x, y) centroids.
        """
        if self.processor is None:
            return np.zeros_like(image), []

        # Preprocess image for SAM
        # SAM expects PIL Image or numpy uint8 RGB
        if image.dtype != np.uint8:
            img_min = image.min()
            img_max = image.max()
            if img_max > img_min:
                image_uint8 = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                image_uint8 = np.zeros_like(image, dtype=np.uint8)
        else:
            image_uint8 = image

        if image_uint8.ndim == 2:
            image_rgb = np.stack([image_uint8] * 3, axis=-1)
        else:
            image_rgb = image_uint8
            
        pil_image = Image.fromarray(image_rgb)

        # Set image
        self.inference_state = self.processor.set_image(pil_image)
        
        # Prompt with text
        output = self.processor.set_text_prompt(state=self.inference_state, prompt=text_prompt)
        
        # Output contains masks, boxes, scores
        masks = output["masks"] # Shape: (N, H, W) or similar?
        # Need to check shape. Usually (N, 1, H, W) or (N, H, W)
        
        # Combine masks
        full_mask = np.zeros(image.shape[:2], dtype=np.float32)
        detections = []
        
        # Convert tensor to numpy if needed
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
            
        # Iterate over masks
        # masks shape might be (N, 1, H, W)
        if masks.ndim == 4:
            masks = masks[:, 0, :, :]
            
        for m in masks:
            # m is (H, W) boolean or float
            m_binary = m > 0.5
            full_mask = np.maximum(full_mask, m_binary.astype(float))
            
            # Extract centroid
            if np.any(m_binary):
                y, x = np.argwhere(m_binary).mean(axis=0)
                detections.append((int(x), int(y)))
            
        return full_mask, detections
