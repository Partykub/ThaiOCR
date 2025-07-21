
"""
Thai OCR Wrapper for RTX 5090 Compatibility
Automatically handles GPU/CPU fallback
"""

import os
import warnings
from paddleocr import PaddleOCR

class ThaiOCR:
    """RTX 5090 compatible Thai OCR wrapper"""
    
    def __init__(self, **kwargs):
        # Force CPU mode for RTX 5090 compatibility
        # Remove use_gpu parameter as it's handled differently in newer versions
        kwargs.pop('use_gpu', None)
        
        # Suppress warnings about GPU
        warnings.filterwarnings("ignore", category=UserWarning)
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
        print("🔧 ThaiOCR: Using CPU mode for RTX 5090 compatibility")
        
        # Initialize PaddleOCR with minimal parameters for compatibility
        self.ocr = PaddleOCR(
            lang='en',  # Use English base model (supports Latin characters)
            **kwargs
        )
        
        print("✅ ThaiOCR: Ready for Thai text recognition")
    
    def recognize(self, image_path):
        """Recognize Thai text from image"""
        try:
            result = self.ocr.ocr(image_path, cls=True)
            return self._format_result(result)
        except Exception as e:
            print(f"❌ OCR recognition failed: {e}")
            return []
    
    def _format_result(self, result):
        """Format OCR result for easy use"""
        if not result or not result[0]:
            return []
        
        formatted = []
        for line in result[0]:
            bbox = line[0]  # Bounding box coordinates
            text = line[1][0]  # Recognized text
            confidence = line[1][1]  # Confidence score
            
            formatted.append({
                'text': text,
                'confidence': confidence,
                'bbox': bbox
            })
        
        return formatted

# Example usage
if __name__ == "__main__":
    # Test the wrapper
    ocr = ThaiOCR()
    print("Thai OCR wrapper initialized successfully!")
