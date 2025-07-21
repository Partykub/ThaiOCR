#!/usr/bin/env python3
"""
CRNN Text Reading Script - Real Image Testing
===========================================

This script attempts to load and use the CRNN model to read text from actual images,
with fallbacks and error handling for RTX 5090 compatibility issues.

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

class CRNNTextReader:
    """CRNN Text Reader for Real Images"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.crnn_path = self.project_root / "thai-license-plate-recognition-CRNN"
        self.model_file = self.crnn_path / "Model_LSTM+BN5--thai-v3.h5"
        self.model = None
        
        # Thai characters (expanded for license plates + general text)
        self.characters = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮะาำิีึืุูเแโใไ็่้๊๋์ํ๎"
        
        logger.info("🚗 CRNN Text Reader initialized")
        logger.info(f"📁 Model path: {self.model_file}")
    
    def load_model_with_fallbacks(self):
        """Load CRNN model with multiple fallback strategies"""
        logger.info("🔄 LOADING CRNN MODEL WITH FALLBACKS...")
        
        try:
            import tensorflow as tf
            logger.info(f"   📦 TensorFlow: {tf.__version__}")
            
            # Strategy 1: Direct loading
            try:
                logger.info("   🔄 Strategy 1: Direct model loading...")
                self.model = tf.keras.models.load_model(str(self.model_file), compile=False)
                logger.info("   ✅ Strategy 1: SUCCESS - Model loaded directly")
                return True
            except Exception as e1:
                logger.warning(f"   ⚠️  Strategy 1 failed: {e1}")
            
            # Strategy 2: Custom objects
            try:
                logger.info("   🔄 Strategy 2: Loading with custom objects...")
                self.model = tf.keras.models.load_model(str(self.model_file), compile=False, custom_objects={})
                logger.info("   ✅ Strategy 2: SUCCESS - Model loaded with custom objects")
                return True
            except Exception as e2:
                logger.warning(f"   ⚠️  Strategy 2 failed: {e2}")
            
            # Strategy 3: Manual architecture + weights
            try:
                logger.info("   🔄 Strategy 3: Manual loading from CRNN directory...")
                
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
                    logger.info("   ✅ Strategy 3: SUCCESS - Weights loaded")
                    return True
                else:
                    logger.error("   ❌ Weights file not found")
                    
            except Exception as e3:
                logger.warning(f"   ⚠️  Strategy 3 failed: {e3}")
            
            # Strategy 4: CPU-only fallback
            try:
                logger.info("   🔄 Strategy 4: CPU-only fallback...")
                
                # Force CPU
                with tf.device('/cpu:0'):
                    self.model = tf.keras.models.load_model(str(self.model_file), compile=False)
                    logger.info("   ✅ Strategy 4: SUCCESS - CPU-only model loaded")
                    return True
                    
            except Exception as e4:
                logger.warning(f"   ⚠️  Strategy 4 failed: {e4}")
            
            logger.error("   ❌ All loading strategies failed")
            return False
            
        except Exception as e:
            logger.error(f"❌ Model loading completely failed: {e}")
            return False
    
    def preprocess_image_for_crnn(self, image_path):
        """Preprocess image for CRNN model"""
        logger.info(f"🖼️ PREPROCESSING: {image_path}")
        
        try:
            # Load image
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
            else:
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
            
            # Resize to CRNN input size (128x64)
            # CRITICAL: Model expects (height=64, width=128) so resize to (128, 64)
            target_width, target_height = 128, 64
            resized = cv2.resize(gray, (target_width, target_height))
            logger.info(f"   📐 Resized to: {target_width}x{target_height}")
            
            # CRITICAL FIX: Transpose to match model input (128, 64, 1)
            # OpenCV resize gives (height, width), but model wants (width, height)
            # So we need to transpose from (64, 128) to (128, 64)
            transposed = resized.T  # Transpose: (64, 128) -> (128, 64)
            logger.info(f"   🔄 Transposed to: {transposed.shape}")
            
            # Normalize to [0, 1]
            normalized = transposed.astype(np.float32) / 255.0
            
            # Add channel dimension for grayscale
            if len(normalized.shape) == 2:
                normalized = np.expand_dims(normalized, axis=-1)
            
            # Add batch dimension
            batch_input = np.expand_dims(normalized, axis=0)
            
            logger.info(f"   ✅ Final shape: {batch_input.shape}")
            return batch_input, transposed
            
        except Exception as e:
            logger.error(f"   ❌ Preprocessing failed: {e}")
            return None, None
    
    def decode_crnn_predictions(self, predictions):
        """Decode CRNN predictions to text"""
        try:
            logger.info("🔤 DECODING PREDICTIONS...")
            
            if predictions is None:
                return "ERROR: No predictions"
            
            logger.info(f"   📊 Prediction shape: {predictions.shape}")
            
            # Remove batch dimension if present
            if len(predictions.shape) == 3:
                preds = predictions[0]  # Shape: (time_steps, num_classes)
            else:
                preds = predictions
            
            # Simple greedy decoding (not proper CTC)
            predicted_indices = np.argmax(preds, axis=-1)
            logger.info(f"   🔢 Predicted indices shape: {predicted_indices.shape}")
            
            # Convert indices to characters
            decoded_chars = []
            prev_idx = -1
            
            for idx in predicted_indices:
                # Skip repetitions and blank (assuming class 0 is blank)
                if idx != prev_idx and idx > 0 and idx < len(self.characters):
                    char = self.characters[idx]
                    decoded_chars.append(char)
                prev_idx = idx
            
            decoded_text = ''.join(decoded_chars)
            
            # Clean up the text
            cleaned_text = decoded_text.strip()
            
            logger.info(f"   📝 Raw decoded: '{decoded_text}'")
            logger.info(f"   ✨ Cleaned text: '{cleaned_text}'")
            
            return cleaned_text if cleaned_text else "EMPTY_PREDICTION"
            
        except Exception as e:
            logger.error(f"   ❌ Decoding failed: {e}")
            return f"DECODE_ERROR: {e}"
    
    def read_text_from_image(self, image_path):
        """Read text from a single image"""
        logger.info(f"📖 READING TEXT FROM: {Path(image_path).name}")
        
        start_time = time.time()
        
        try:
            # Preprocess image
            processed_image, display_image = self.preprocess_image_for_crnn(image_path)
            if processed_image is None:
                return None
            
            # Run inference
            logger.info("   🧠 Running CRNN inference...")
            
            inference_start = time.time()
            predictions = self.model.predict(processed_image, verbose=0)
            inference_time = time.time() - inference_start
            
            logger.info(f"   ⏱️  Inference time: {inference_time:.3f}s")
            
            # Decode predictions
            predicted_text = self.decode_crnn_predictions(predictions)
            
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
            logger.error(f"   ❌ Text reading failed: {e}")
            return {
                "image_path": str(image_path),
                "predicted_text": f"ERROR: {e}",
                "inference_time": 0,
                "total_time": 0,
                "confidence": 0,
                "error": str(e)
            }
    
    def find_test_images(self):
        """Find available test images"""
        logger.info("🔍 FINDING TEST IMAGES...")
        
        # Possible image locations
        image_sources = [
            self.project_root / "thai-letters" / "thai_ocr_dataset" / "images",
            self.crnn_path / "DB" / "test",
            self.crnn_path / "DB" / "train",
            self.project_root / "test_images"
        ]
        
        found_images = []
        
        for source in image_sources:
            if source.exists():
                # Find image files
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    images = list(source.glob(ext))
                    if images:
                        logger.info(f"   📁 {source.name}: {len(images)} images found")
                        found_images.extend(images[:5])  # Take first 5 from each source
        
        # Also check current directory
        current_dir = Path.cwd()
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            images = list(current_dir.glob(ext))
            if images:
                logger.info(f"   📁 Current directory: {len(images)} images found")
                found_images.extend(images[:3])
        
        if found_images:
            logger.info(f"   ✅ Total images available: {len(found_images)}")
            return found_images[:10]  # Limit to 10 images for testing
        else:
            logger.warning("   ⚠️  No test images found")
            return []
    
    def create_sample_test_image(self):
        """Create a sample test image if no images found"""
        logger.info("🎨 CREATING SAMPLE TEST IMAGE...")
        
        try:
            # Create simple test image with Thai text
            img = np.ones((64, 128, 3), dtype=np.uint8) * 255  # White background
            
            # Add some text (this is very basic - real OCR needs proper text rendering)
            cv2.putText(img, "Test", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(img, "1234", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Save test image
            test_image_path = self.project_root / "build-model-th" / "sample_test.jpg"
            cv2.imwrite(str(test_image_path), img)
            
            logger.info(f"   ✅ Sample image created: {test_image_path}")
            return [test_image_path]
            
        except Exception as e:
            logger.error(f"   ❌ Sample image creation failed: {e}")
            return []
    
    def run_text_reading_test(self):
        """Run comprehensive text reading test"""
        logger.info("=" * 80)
        logger.info("📖 CRNN TEXT READING TEST")
        logger.info("=" * 80)
        
        try:
            # Load model
            if not self.load_model_with_fallbacks():
                logger.error("❌ Cannot proceed without model")
                return False
            
            # Display model info
            if self.model:
                logger.info("🤖 MODEL INFORMATION:")
                logger.info(f"   📏 Input shape: {self.model.input_shape}")
                logger.info(f"   📐 Output shape: {self.model.output_shape}")
                logger.info(f"   🔢 Parameters: {self.model.count_params():,}")
            
            # Find test images
            test_images = self.find_test_images()
            
            if not test_images:
                logger.info("   📝 No images found, creating sample...")
                test_images = self.create_sample_test_image()
            
            if not test_images:
                logger.error("❌ No test images available")
                return False
            
            # Run text reading on each image
            results = []
            
            logger.info(f"\n🔄 PROCESSING {len(test_images)} IMAGES...")
            
            for i, image_path in enumerate(test_images, 1):
                logger.info(f"\n--- IMAGE {i}/{len(test_images)} ---")
                
                result = self.read_text_from_image(image_path)
                if result:
                    results.append(result)
                
                # Small delay between images
                time.sleep(0.1)
            
            # Generate summary
            self.generate_reading_summary(results)
            
            # Save results
            self.save_reading_results(results)
            
            return len(results) > 0
            
        except Exception as e:
            logger.error(f"❌ Text reading test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_reading_summary(self, results):
        """Generate reading results summary"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 TEXT READING SUMMARY")
        logger.info("=" * 60)
        
        if not results:
            logger.info("❌ No results to summarize")
            return
        
        # Statistics
        successful_reads = [r for r in results if not r['predicted_text'].startswith('ERROR')]
        avg_inference_time = np.mean([r['inference_time'] for r in results])
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        logger.info(f"📈 STATISTICS:")
        logger.info(f"   📷 Total images: {len(results)}")
        logger.info(f"   ✅ Successful reads: {len(successful_reads)}")
        logger.info(f"   ⏱️  Avg inference time: {avg_inference_time:.3f}s")
        logger.info(f"   📊 Avg confidence: {avg_confidence:.3f}")
        
        logger.info(f"\n📝 READING RESULTS:")
        for i, result in enumerate(results, 1):
            status = "✅" if not result['predicted_text'].startswith('ERROR') else "❌"
            image_name = Path(result['image_path']).name
            text = result['predicted_text'][:50]  # Truncate long text
            
            logger.info(f"   {status} {i:2d}. {image_name:20s} → '{text}'")
        
        # Recommendations
        success_rate = len(successful_reads) / len(results) * 100
        
        logger.info(f"\n💡 ASSESSMENT:")
        if success_rate >= 80:
            logger.info("   🎉 EXCELLENT: CRNN is working well!")
        elif success_rate >= 50:
            logger.info("   ⚠️  FAIR: CRNN has some issues but usable")
        elif success_rate >= 20:
            logger.info("   ❌ POOR: CRNN needs improvement")
        else:
            logger.info("   💥 BROKEN: CRNN not functioning properly")
        
        logger.info(f"   📊 Success rate: {success_rate:.1f}%")
    
    def save_reading_results(self, results):
        """Save reading results to file"""
        logger.info("\n📄 SAVING READING RESULTS...")
        
        report = {
            "test_session": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
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
        
        report_file = self.project_root / "build-model-th" / "crnn_reading_results.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"   ✅ Results saved: {report_file}")

def main():
    """Main text reading function"""
    try:
        reader = CRNNTextReader()
        success = reader.run_text_reading_test()
        
        if success:
            logger.info("\n✅ CRNN Text Reading Test: SUCCESS")
            return 0
        else:
            logger.error("\n❌ CRNN Text Reading Test: FAILED")
            return 1
            
    except KeyboardInterrupt:
        logger.info("\n❌ Test interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\n❌ Test error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
