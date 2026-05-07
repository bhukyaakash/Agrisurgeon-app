"""
Image Processing Utilities for AgriSurgeon
"""

import logging
from typing import Tuple
import numpy as np
from PIL import Image
import io

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Image processing and validation"""
    
    ALLOWED_FORMATS = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}
    MAX_SIZE = 10 * 1024 * 1024  # 10MB
    TARGET_SIZE = (224, 224)
    
    @staticmethod
    def validate_image(file_content: bytes) -> Tuple[bool, str]:
        """Validate image file"""
        try:
            # Check file size
            if len(file_content) > ImageProcessor.MAX_SIZE:
                return False, "File too large (max 10MB)"
            
            # Try opening as image
            img = Image.open(io.BytesIO(file_content))
            
            # Check format
            if img.format.lower() not in ImageProcessor.ALLOWED_FORMATS:
                return False, f"Unsupported format. Allowed: {', '.join(ImageProcessor.ALLOWED_FORMATS)}"
            
            # Check image has content
            if img.size[0] < 10 or img.size[1] < 10:
                return False, "Image too small (minimum 10x10 pixels)"
            
            return True, "Valid"
        
        except Exception as e:
            return False, f"Invalid image file: {str(e)}"
    
    @staticmethod
    def process_image(file_content: bytes) -> Tuple[np.ndarray, str]:
        """Process image for model inference"""
        try:
            # Open image
            image = Image.open(io.BytesIO(file_content))
            
            # Convert to RGB (handle RGBA, grayscale, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize
            image = image.resize(ImageProcessor.TARGET_SIZE, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            image_array = np.array(image, dtype=np.float32)
            
            # Normalize to [0, 1]
            image_array = image_array / 255.0
            
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            logger.info(f"Image processed: shape={image_array.shape}, dtype={image_array.dtype}")
            
            return image_array, "Success"
        
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
            return None, f"Processing failed: {str(e)}"
    
    @staticmethod
    def get_image_quality_metrics(image_array: np.ndarray) -> dict:
        """Calculate image quality metrics"""
        try:
            # Remove batch dimension if present
            if len(image_array.shape) == 4:
                image_array = image_array[0]
            
            metrics = {
                "brightness": float(np.mean(image_array)),
                "contrast": float(np.std(image_array)),
                "min_value": float(np.min(image_array)),
                "max_value": float(np.max(image_array)),
                "shape": list(image_array.shape)
            }
            
            return metrics
        
        except Exception as e:
            logger.error(f"Quality metrics error: {str(e)}")
            return {}
    
    @staticmethod
    def apply_augmentation(image_array: np.ndarray, mode: str = 'mild') -> np.ndarray:
        """Apply data augmentation"""
        try:
            # Remove batch dimension
            if len(image_array.shape) == 4:
                image_array = image_array[0]
            
            if mode == 'mild':
                # Slight brightness adjustment
                brightness_factor = np.random.uniform(0.9, 1.1)
                image_array = np.clip(image_array * brightness_factor, 0, 1)
            
            elif mode == 'moderate':
                # Brightness and contrast
                brightness_factor = np.random.uniform(0.8, 1.2)
                image_array = np.clip(image_array * brightness_factor, 0, 1)
                
                # Contrast adjustment
                mean = np.mean(image_array)
                image_array = np.clip((image_array - mean) * 1.1 + mean, 0, 1)
            
            # Re-add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
        
        except Exception as e:
            logger.error(f"Augmentation error: {str(e)}")
            return image_array
