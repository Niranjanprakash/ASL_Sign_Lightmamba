import random
import numpy as np
from PIL import Image, ImageEnhance

class VideoAugmentation:
    """
    Applies consistent transformations to an entire sequence of frames [T, H, W, C].
    """
    def __init__(self, use_horizontal_flip: bool = False, brightness_range: tuple = (0.8, 1.2), contrast_range: tuple = (0.8, 1.2)):
        self.use_horizontal_flip = use_horizontal_flip
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        # frames: [T, H, W, C] (numpy array uint8)
        augmented_frames = []
        
        # Decide factors once per video to keep temporal consistency
        do_flip = self.use_horizontal_flip and (random.random() > 0.5)
        brightness_factor = random.uniform(*self.brightness_range)
        contrast_factor = random.uniform(*self.contrast_range)
        
        for frame in frames:
            img = Image.fromarray(frame)
            
            # Flip
            if do_flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                
            # Brightness
            if brightness_factor != 1.0:
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(brightness_factor)
                
            # Contrast
            if contrast_factor != 1.0:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(contrast_factor)
                
            augmented_frames.append(np.array(img))
            
        return np.stack(augmented_frames, axis=0)
