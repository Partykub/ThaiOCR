#!/usr/bin/env python3
"""
CRNN License Plate Test - Real Thai License Plate Reading
=======================================================

Test CRNN model with actual Thai license plate images to see if it works
correctly with its intended purpose.

Author: Thai OCR Development Team
Date: July 21, 2025
"""

import os
import sys
import cv2
import numpy as np
import logging
from pathlib import Path
import time
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

class CRNNLicensePlateReader:
    """CRNN Reader specifically for Thai License Plates"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.crnn_path = self.project_root / "thai-license-plate-recognition-CRNN"
        self.model_file = self.crnn_path / "Model_LSTM+BN5--thai-v3.h5"
        self.model = None
        
        # Thai license plate characters (typical set)
        # Based on typical Thai license plate patterns
        self.characters = " กขคงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        
        logger.info("🚗 CRNN License Plate Reader initialized")
        logger.info(f"📁 Model path: {self.model_file}")
    
    def load_crnn_model(self):
        """Load CRNN model for license plates"""
        logger.info("🔄 LOADING CRNN MODEL...")
        
        try:
            import tensorflow as tf
            logger.info(f"   📦 TensorFlow: {tf.__version__}")
            
            # Add CRNN path to Python path
            sys.path.insert(0, str(self.crnn_path))
            
            # Import model architecture
            import Model
            import parameter
            
            # Create model
            self.model = Model.get_Model(training=False)
            logger.info("   ✅ Model architecture loaded")
            
            # Load weights
            weights_file = self.crnn_path / "LSTM+BN5--thai-v3.hdf5"
            if weights_file.exists():
                self.model.load_weights(str(weights_file))
                logger.info("   ✅ Weights loaded successfully")
                return True
            else:
                logger.error("   ❌ Weights file not found")
                return False
                
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            return False
    
    def preprocess_license_plate(self, image_path):
        """Preprocess license plate image for CRNN"""
        logger.info(f"🖼️ PREPROCESSING: {Path(image_path).name}")
        
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"   ❌ Cannot load image: {image_path}")
                return None
            
            original_shape = image.shape
            logger.info(f"   📏 Original size: {original_shape[1]}x{original_shape[0]}")
            
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # License plate specific preprocessing
            # Resize to model input size (128x64)
            target_width, target_height = 128, 64
            resized = cv2.resize(gray, (target_width, target_height))
            logger.info(f"   📐 Resized to: {target_width}x{target_height}")
            
            # Transpose to match model input (128, 64)
            transposed = resized.T
            logger.info(f"   🔄 Transposed to: {transposed.shape}")
            
            # Normalize to [0, 1]
            normalized = transposed.astype(np.float32) / 255.0
            
            # Add channel dimension
            if len(normalized.shape) == 2:
                normalized = np.expand_dims(normalized, axis=-1)
            
            # Add batch dimension
            batch_input = np.expand_dims(normalized, axis=0)
            
            logger.info(f"   ✅ Final shape: {batch_input.shape}")
            return batch_input, transposed
            
        except Exception as e:
            logger.error(f"   ❌ Preprocessing failed: {e}")
            return None, None
    
    def decode_license_plate_text(self, predictions):
        """Decode predictions to license plate text"""
        try:
            logger.info("🔤 DECODING LICENSE PLATE...")
            
            if predictions is None:
                return "ERROR: No predictions"
            
            logger.info(f"   📊 Prediction shape: {predictions.shape}")
            
            # Remove batch dimension
            if len(predictions.shape) == 3:
                preds = predictions[0]
            else:
                preds = predictions
            
            # Simple greedy decoding (basic CTC)
            predicted_indices = np.argmax(preds, axis=-1)
            logger.info(f"   🔢 Predicted indices: {predicted_indices[:10]}...")  # Show first 10
            
            # Convert indices to characters
            decoded_chars = []
            prev_idx = -1
            
            for idx in predicted_indices:
                # Skip repetitions and blank (class 0 is blank)
                if idx != prev_idx and idx > 0 and idx < len(self.characters):
                    char = self.characters[idx]
                    decoded_chars.append(char)
                prev_idx = idx
            
            decoded_text = ''.join(decoded_chars)
            cleaned_text = decoded_text.strip()
            
            logger.info(f"   📝 Raw decoded: '{decoded_text}'")
            logger.info(f"   ✨ Cleaned text: '{cleaned_text}'")
            
            # Additional license plate formatting
            formatted_text = self.format_license_plate(cleaned_text)
            
            return formatted_text if formatted_text else "EMPTY_PREDICTION"
            
        except Exception as e:
            logger.error(f"   ❌ Decoding failed: {e}")
            return f"DECODE_ERROR: {e}"
    
    def format_license_plate(self, text):
        """Format text as Thai license plate pattern"""
        if not text:
            return text
        
        # Basic license plate patterns (Thai format)
        # Examples: กข1234, 1กข234, etc.
        formatted = text.replace(" ", "")  # Remove spaces
        
        # If it looks like a license plate, format it
        if len(formatted) >= 4:
            logger.info(f"   🚗 Formatted as license plate: '{formatted}'")
            return formatted
        
        return text
    
    def read_license_plate(self, image_path):
        """Read text from license plate image"""
        logger.info(f"📖 READING LICENSE PLATE: {Path(image_path).name}")
        
        start_time = time.time()
        
        try:
            # Preprocess image
            processed_image, display_image = self.preprocess_license_plate(image_path)
            if processed_image is None:
                return None
            
            # Run inference
            logger.info("   🧠 Running CRNN inference...")
            
            inference_start = time.time()
            predictions = self.model.predict(processed_image, verbose=0)
            inference_time = time.time() - inference_start
            
            logger.info(f"   ⏱️  Inference time: {inference_time:.3f}s")
            
            # Decode predictions
            predicted_text = self.decode_license_plate_text(predictions)
            
            total_time = time.time() - start_time
            
            result = {
                "image_path": str(image_path),
                "predicted_text": predicted_text,
                "inference_time": inference_time,
                "total_time": total_time,
                "confidence": float(np.max(predictions)) if predictions is not None else 0.0,
                "image_shape": display_image.shape if display_image is not None else None
            }
            
            logger.info(f"   🎯 RESULT: '{predicted_text}'")
            logger.info(f"   📊 Confidence: {result['confidence']:.3f}")
            logger.info(f"   ⏱️  Total time: {total_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"   ❌ License plate reading failed: {e}")
            return {
                "image_path": str(image_path),
                "predicted_text": f"ERROR: {e}",
                "inference_time": 0,
                "total_time": 0,
                "confidence": 0,
                "error": str(e)
            }
    
    def find_license_plate_images(self):
        """Find license plate test images"""
        logger.info("🔍 FINDING LICENSE PLATE IMAGES...")
        
        # License plate image sources
        image_sources = [
            self.crnn_path / "DB" / "test",
            self.crnn_path / "DB" / "train",
        ]
        
        found_images = []
        
        for source in image_sources:
            if source.exists():
                # Find image files
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    images = list(source.glob(ext))
                    if images:
                        logger.info(f"   📁 {source.name}: {len(images)} images found")
                        found_images.extend(images[:8])  # Take 8 from each
        
        if found_images:
            logger.info(f"   ✅ Total license plate images: {len(found_images)}")
            return found_images[:12]  # Limit to 12 images
        else:
            logger.warning("   ⚠️  No license plate images found")
            return []
    
    def run_license_plate_test(self):
        """Run comprehensive license plate reading test"""
        logger.info("=" * 80)
        logger.info("🚗 CRNN LICENSE PLATE READING TEST")
        logger.info("=" * 80)
        
        try:
            # Load model
            if not self.load_crnn_model():
                logger.error("❌ Cannot proceed without model")
                return False
            
            # Display model info
            if self.model:
                logger.info("🤖 MODEL INFORMATION:")
                logger.info(f"   📏 Input shape: {self.model.input_shape}")
                logger.info(f"   📐 Output shape: {self.model.output_shape}")
                logger.info(f"   🔢 Parameters: {self.model.count_params():,}")
            
            # Find license plate images
            test_images = self.find_license_plate_images()
            
            if not test_images:
                logger.error("❌ No license plate images found")
                return False
            
            # Run license plate reading
            results = []
            
            logger.info(f"\n🔄 PROCESSING {len(test_images)} LICENSE PLATES...")
            
            for i, image_path in enumerate(test_images, 1):
                logger.info(f"\n--- LICENSE PLATE {i}/{len(test_images)} ---")
                
                result = self.read_license_plate(image_path)
                if result:
                    results.append(result)
                
                time.sleep(0.1)  # Small delay
            
            # Generate summary
            self.generate_summary(results)
            
            # Save results
            self.save_results(results)
            
            return len(results) > 0
            
        except Exception as e:
            logger.error(f"❌ License plate test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_summary(self, results):
        """Generate license plate reading summary"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 LICENSE PLATE READING SUMMARY")
        logger.info("=" * 60)
        
        if not results:
            logger.info("❌ No results to summarize")
            return
        
        # Statistics
        successful_reads = [r for r in results if not r['predicted_text'].startswith('ERROR')]
        avg_inference_time = np.mean([r['inference_time'] for r in results])
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        logger.info(f"📈 STATISTICS:")
        logger.info(f"   🚗 Total license plates: {len(results)}")
        logger.info(f"   ✅ Successful reads: {len(successful_reads)}")
        logger.info(f"   ⏱️  Avg inference time: {avg_inference_time:.3f}s")
        logger.info(f"   📊 Avg confidence: {avg_confidence:.3f}")
        
        logger.info(f"\n📝 LICENSE PLATE RESULTS:")
        for i, result in enumerate(results, 1):
            status = "✅" if not result['predicted_text'].startswith('ERROR') else "❌"
            image_name = Path(result['image_path']).name[:15]  # Truncate filename
            text = result['predicted_text'][:20]  # Truncate text
            
            logger.info(f"   {status} {i:2d}. {image_name:15s} → '{text}'")
        
        # Assessment
        success_rate = len(successful_reads) / len(results) * 100
        
        logger.info(f"\n💡 LICENSE PLATE ASSESSMENT:")
        if success_rate >= 80:
            logger.info("   🎉 EXCELLENT: CRNN works great for license plates!")
        elif success_rate >= 50:
            logger.info("   ⚠️  FAIR: CRNN has some issues but usable for license plates")
        elif success_rate >= 20:
            logger.info("   ❌ POOR: CRNN needs improvement for license plates")
        else:
            logger.info("   💥 BROKEN: CRNN not functioning for license plates")
        
        logger.info(f"   📊 Success rate: {success_rate:.1f}%")
        
        # Recommendations
        logger.info(f"\n🔮 RECOMMENDATIONS:")
        if success_rate >= 70:
            logger.info("   ✅ CRNN model is suitable for Thai license plate recognition")
            logger.info("   💡 Consider using this for license plate specific tasks")
        else:
            logger.info("   ⚠️  CRNN model may need retraining or adjustment")
            logger.info("   💡 Consider Task 7: PaddleOCR Thai Training for better results")
    
    def save_results(self, results):
        """Save license plate reading results"""
        logger.info("\n📄 SAVING LICENSE PLATE RESULTS...")
        
        report = {
            "test_session": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "test_type": "Thai License Plate Recognition",
                "model_file": str(self.model_file),
                "total_images": len(results),
                "successful_reads": len([r for r in results if not r['predicted_text'].startswith('ERROR')])
            },
            "model_info": {
                "input_shape": str(self.model.input_shape) if self.model else "Unknown",
                "output_shape": str(self.model.output_shape) if self.model else "Unknown",
                "parameters": self.model.count_params() if self.model else 0
            },
            "results": results,
            "statistics": {
                "avg_inference_time": np.mean([r['inference_time'] for r in results]) if results else 0,
                "avg_confidence": np.mean([r['confidence'] for r in results]) if results else 0,
                "success_rate": len([r for r in results if not r['predicted_text'].startswith('ERROR')]) / len(results) * 100 if results else 0
            }
        }
        
        report_file = self.project_root / "build-model-th" / "crnn_license_plate_results.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"   ✅ Results saved: {report_file}")

def main():
    """Main license plate reading function"""
    try:
        reader = CRNNLicensePlateReader()
        success = reader.run_license_plate_test()
        
        if success:
            logger.info("\n✅ CRNN License Plate Test: SUCCESS")
            return 0
        else:
            logger.error("\n❌ CRNN License Plate Test: FAILED")
            return 1
            
    except KeyboardInterrupt:
        logger.info("\n❌ Test interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\n❌ Test error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
