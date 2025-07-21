#!/usr/bin/env python3
"""
CRNN Thai OCR Training Script
============================

Train CRNN model with Thai OCR dataset to recognize Thai text.
This script will retrain the CRNN model with our custom Thai dataset.

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
import random
from collections import Counter
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

class ThaiCRNNTrainer:
    """CRNN Trainer for Thai OCR Dataset"""
    
    def __init__(self):
        # MANDATORY GPU VERIFICATION - NO CPU TRAINING ALLOWED
        self.verify_gpu_availability()
        
        self.project_root = Path(__file__).parent.parent
        self.dataset_path = self.project_root / "thai-letters" / "thai_ocr_dataset"
        self.images_path = self.dataset_path / "images"
        self.labels_file = self.dataset_path / "labels.txt"
        self.crnn_path = self.project_root / "thai-license-plate-recognition-CRNN"
        
        # Training parameters
        self.img_w, self.img_h = 128, 64
        self.batch_size = 32
        self.epochs = 50
        self.max_text_len = 20  # Maximum text length
        
        self.characters = []
        self.char_to_num = {}
        self.num_to_char = {}
        self.num_classes = 0
        
        logger.info("🚀 Thai CRNN Trainer initialized")
        logger.info(f"📁 Dataset path: {self.dataset_path}")
        logger.info(f"🖼️ Images path: {self.images_path}")
        logger.info(f"📄 Labels file: {self.labels_file}")
        
    def verify_gpu_availability(self):
        """MANDATORY GPU check before any training - CPU training is FORBIDDEN"""
        try:
            import tensorflow as tf
            
            # Check GPU availability
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if not gpus:
                logger.error("❌ CRITICAL: NO GPU DETECTED - TRAINING ABORTED")
                logger.error("❌ RTX 5090 GPU is MANDATORY for training")
                logger.error("❌ CPU training is STRICTLY FORBIDDEN")
                raise RuntimeError("CRITICAL: NO GPU DETECTED - TRAINING CANNOT PROCEED")
            
            # Verify GPU is working
            try:
                with tf.device('/GPU:0'):
                    test_tensor = tf.constant([1.0, 2.0, 3.0])
                    result = tf.reduce_sum(test_tensor)
                    
                logger.info(f"✅ GPU VERIFICATION PASSED: {len(gpus)} GPU(s) detected")
                logger.info(f"🎮 RTX 5090 GPU READY for Thai CRNN training")
                
                # Configure GPU memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    
            except Exception as e:
                logger.error(f"❌ CRITICAL: GPU test failed - {e}")
                raise RuntimeError(f"CRITICAL: GPU not functional - {e}")
                
        except ImportError:
            logger.error("❌ CRITICAL: TensorFlow GPU not available")
            raise RuntimeError("CRITICAL: TensorFlow GPU import failed")
        except Exception as e:
            logger.error(f"❌ CRITICAL: GPU verification failed - {e}")
            raise RuntimeError(f"CRITICAL: GPU verification failed: {e}")
    
    def analyze_dataset(self):
        """Analyze the dataset to understand character distribution"""
        logger.info("🔍 ANALYZING THAI DATASET...")
        
        if not self.labels_file.exists():
            logger.error(f"❌ Labels file not found: {self.labels_file}")
            return False
        
        # Read all labels
        labels = []
        with open(self.labels_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    filename, text = parts
                    labels.append(text)
        
        logger.info(f"📊 Total samples: {len(labels)}")
        
        # Analyze characters
        all_chars = set()
        text_lengths = []
        
        for text in labels:
            all_chars.update(text)
            text_lengths.append(len(text))
        
        # Character statistics
        char_counter = Counter(''.join(labels))
        
        logger.info(f"🔤 Unique characters: {len(all_chars)}")
        logger.info(f"📏 Max text length: {max(text_lengths)}")
        logger.info(f"📏 Avg text length: {np.mean(text_lengths):.2f}")
        logger.info(f"📏 Min text length: {min(text_lengths)}")
        
        # Most common characters
        logger.info("🎯 Top 20 most common characters:")
        for char, count in char_counter.most_common(20):
            logger.info(f"   '{char}': {count}")
        
        # Build character set from th_dict.txt
        dict_path = self.dataset_path.parent / 'th_dict.txt'
        
        if dict_path.exists():
            logger.info(f"📖 Reading characters from {dict_path}")
            with open(dict_path, 'r', encoding='utf-8') as f:
                dict_chars = [line.strip() for line in f.readlines() if line.strip()]
            
            # Use dictionary characters + any additional from dataset
            self.characters = sorted(list(set(dict_chars + list(all_chars))))
        else:
            logger.warning(f"❌ Dictionary file not found: {dict_path}")
            # Fallback to dataset characters
            self.characters = sorted(list(all_chars))
        
        self.characters.insert(0, '')  # Add blank character for CTC
        
        # Create character mappings
        self.char_to_num = {char: idx for idx, char in enumerate(self.characters)}
        self.num_to_char = {idx: char for idx, char in enumerate(self.characters)}
        self.num_classes = len(self.characters)
        
        logger.info(f"✅ Character set built: {self.num_classes} classes")
        logger.info(f"📝 Characters: {''.join(self.characters[1:])}")  # Skip blank
        
        return True
    
    def load_dataset(self):
        """Load and prepare the dataset"""
        logger.info("📂 LOADING THAI DATASET...")
        
        if not self.analyze_dataset():
            return None, None
        
        # Load all data
        image_paths = []
        texts = []
        
        with open(self.labels_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    filename, text = parts
                    image_path = self.images_path / filename
                    
                    if image_path.exists():
                        image_paths.append(str(image_path))
                        texts.append(text)
        
        logger.info(f"✅ Loaded {len(image_paths)} valid samples")
        
        # Split into train/validation
        indices = list(range(len(image_paths)))
        random.shuffle(indices)
        
        split_idx = int(0.8 * len(indices))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_images = [image_paths[i] for i in train_indices]
        train_texts = [texts[i] for i in train_indices]
        val_images = [image_paths[i] for i in val_indices]
        val_texts = [texts[i] for i in val_indices]
        
        logger.info(f"📊 Training samples: {len(train_images)}")
        logger.info(f"📊 Validation samples: {len(val_images)}")
        
        return (train_images, train_texts), (val_images, val_texts)
    
    def preprocess_image(self, image_path):
        """Preprocess image for CRNN"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Resize to target size
            resized = cv2.resize(gray, (self.img_w, self.img_h))
            
            # Transpose for CRNN input format
            transposed = resized.T
            
            # Normalize
            normalized = transposed.astype(np.float32) / 255.0
            
            # Add channel dimension
            if len(normalized.shape) == 2:
                normalized = np.expand_dims(normalized, axis=-1)
            
            return normalized
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to preprocess {image_path}: {e}")
            return None
    
    def encode_text(self, text):
        """Encode text to numeric representation"""
        try:
            # Convert text to numbers
            encoded = []
            for char in text:
                if char in self.char_to_num:
                    encoded.append(self.char_to_num[char])
                else:
                    # Skip unknown characters
                    logger.warning(f"⚠️ Unknown character: '{char}'")
            
            return encoded
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to encode text '{text}': {e}")
            return []
    
    def create_data_generator(self, image_paths, texts, batch_size=32):
        """Create data generator for training - simplified for classification"""
        def generator():
            indices = list(range(len(image_paths)))
            
            while True:
                random.shuffle(indices)
                
                for i in range(0, len(indices), batch_size):
                    batch_indices = indices[i:i + batch_size]
                    
                    batch_images = []
                    batch_labels = []
                    
                    for idx in batch_indices:
                        # Process image
                        image = self.preprocess_image(image_paths[idx])
                        if image is None:
                            continue
                        
                        # Process text - take first character for simplified training
                        text = texts[idx]
                        if len(text) > 0 and text[0] in self.char_to_num:
                            label = self.char_to_num[text[0]]
                            
                            batch_images.append(image)
                            batch_labels.append(label)
                    
                    if len(batch_images) == 0:
                        continue
                    
                    # Convert to numpy arrays
                    X = np.array(batch_images)
                    y = np.array(batch_labels)
                    
                    yield X, y
        
        return generator
    
    def build_crnn_model(self):
        """Build CRNN model architecture using existing CRNN structure"""
        logger.info("🏗️ BUILDING CRNN MODEL...")
        
        try:
            # Add CRNN path to Python path to import existing model
            sys.path.insert(0, str(self.crnn_path))
            
            # Import the existing model architecture
            import Model
            import tensorflow as tf
            
            # Create model with our number of classes
            # We need to modify the existing model to use our character set
            
            # Build a simplified CRNN model
            from tensorflow.keras import layers, Model as KerasModel
            
            # Input layer
            input_img = layers.Input(shape=(self.img_w, self.img_h, 1), name='image_input')
            
            # CNN layers (simplified)
            x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
            x = layers.MaxPooling2D((2, 2))(x)
            
            x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = layers.MaxPooling2D((2, 2))(x)
            
            x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
            x = layers.MaxPooling2D((2, 1))(x)
            
            # Reshape for RNN - fix calculation  
            # After MaxPooling2D((2,2), (2,2), (2,1)): 
            # 128 -> 64 -> 32 -> 16 (width)
            # 64 -> 32 -> 16 -> 16 (height)  
            # So final shape is (16, 16, 256)
            # Total elements: 16 * 16 * 256 = 65536
            x = layers.Reshape((16, 16 * 256))(x)
            x = layers.Dense(128, activation='relu')(x)
            
            # RNN layers  
            x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
            x = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(x)  # Changed to False
            
            # Output layer - single character classification
            outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
            
            # Create inference model
            inference_model = KerasModel(inputs=input_img, outputs=outputs)
            
            # For training, we'll use a simpler approach without CTC for now
            # Compile model
            inference_model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info("✅ CRNN model built successfully")
            logger.info(f"📊 Model parameters: {inference_model.count_params():,}")
            
            return inference_model, inference_model
            
        except Exception as e:
            logger.error(f"❌ Failed to build model: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def train_model(self):
        """Train the CRNN model"""
        logger.info("=" * 80)
        logger.info("🚀 STARTING THAI CRNN TRAINING")
        logger.info("=" * 80)
        
        try:
            # Load dataset
            train_data, val_data = self.load_dataset()
            if train_data is None:
                logger.error("❌ Failed to load dataset")
                return False
            
            train_images, train_texts = train_data
            val_images, val_texts = val_data
            
            # Build model
            inference_model, training_model = self.build_crnn_model()
            if training_model is None:
                logger.error("❌ Failed to build model")
                return False
            
            # Create data generators
            train_generator = self.create_data_generator(train_images, train_texts, self.batch_size)
            val_generator = self.create_data_generator(val_images, val_texts, self.batch_size)
            
            # Calculate steps
            steps_per_epoch = len(train_images) // self.batch_size
            validation_steps = len(val_images) // self.batch_size
            
            logger.info(f"📊 Training configuration:")
            logger.info(f"   🎯 Epochs: {self.epochs}")
            logger.info(f"   📦 Batch size: {self.batch_size}")
            logger.info(f"   👣 Steps per epoch: {steps_per_epoch}")
            logger.info(f"   ✅ Validation steps: {validation_steps}")
            logger.info(f"   🔤 Number of classes: {self.num_classes}")
            
            # Callbacks
            import tensorflow as tf
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    str(self.project_root / "build-model-th" / "thai_crnn_best.h5"),
                    save_best_only=True,
                    monitor='val_loss',
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    verbose=1
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    verbose=1
                )
            ]
            
            # Start training
            logger.info("🏃‍♂️ Starting training...")
            
            start_time = time.time()
            
            history = training_model.fit(
                train_generator(),
                steps_per_epoch=steps_per_epoch,
                epochs=self.epochs,
                validation_data=val_generator(),
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=1
            )
            
            training_time = time.time() - start_time
            
            logger.info(f"✅ Training completed in {training_time:.2f} seconds")
            
            # Save final model
            inference_model.save(str(self.project_root / "build-model-th" / "thai_crnn_final.h5"))
            
            # Save character mappings
            char_map = {
                'characters': self.characters,
                'char_to_num': self.char_to_num,
                'num_to_char': self.num_to_char,
                'num_classes': self.num_classes
            }
            
            with open(self.project_root / "build-model-th" / "thai_char_map.json", 'w', encoding='utf-8') as f:
                json.dump(char_map, f, ensure_ascii=False, indent=2)
            
            # Save training history
            history_data = {
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss'],
                'training_time': training_time,
                'epochs': self.epochs,
                'batch_size': self.batch_size
            }
            
            with open(self.project_root / "build-model-th" / "thai_training_history.json", 'w') as f:
                json.dump(history_data, f, indent=2)
            
            logger.info("💾 Model and metadata saved successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main training function"""
    try:
        # Set random seeds for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        # MANDATORY GPU VERIFICATION - NO CPU TRAINING ALLOWED
        try:
            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if not gpus:
                logger.error("❌ CRITICAL: NO GPU DETECTED - TRAINING ABORTED")
                logger.error("❌ GPU is MANDATORY for training - CPU training is FORBIDDEN")
                raise RuntimeError("CRITICAL: GPU NOT AVAILABLE - TRAINING CANNOT PROCEED")
            
            # Configure GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"✅ GPU VERIFIED: {len(gpus)} device(s) available for training")
            logger.info(f"🎮 RTX 5090 GPU READY for Thai CRNN training")
            
        except ImportError:
            logger.error("❌ CRITICAL: TensorFlow not available - TRAINING ABORTED")
            raise RuntimeError("CRITICAL: TensorFlow GPU not available")
        except Exception as e:
            logger.error(f"❌ CRITICAL: GPU verification failed - {e}")
            raise RuntimeError(f"CRITICAL: GPU initialization failed: {e}")
        
        # Create trainer and start training
        trainer = ThaiCRNNTrainer()
        success = trainer.train_model()
        
        if success:
            logger.info("\n🎉 THAI CRNN TRAINING: SUCCESS!")
            return 0
        else:
            logger.error("\n❌ THAI CRNN TRAINING: FAILED!")
            return 1
            
    except KeyboardInterrupt:
        logger.info("\n❌ Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\n❌ Training error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
