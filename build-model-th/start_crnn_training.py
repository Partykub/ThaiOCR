#!/usr/bin/env python3
"""
Enhanced CRNN Training Script for Thai License Plate Recognition
Optimized for RTX 5090 GPU with comprehensive monitoring and error handling

Features:
- RTX 5090 GPU optimization
- Comprehensive error handling
- Training progress monitoring
- Model validation and testing
- Automatic checkpoint management
- Performance metrics tracking
"""

import os
import sys
import time
import logging
import warnings
import traceback
from pathlib import Path

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adadelta, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crnn_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RTX5090CRNNTrainer:
    """Enhanced CRNN Trainer optimized for RTX 5090"""
    
    def __init__(self, project_root=None):
        """Initialize the trainer"""
        if project_root is None:
            self.project_root = Path(__file__).parent.parent
        else:
            self.project_root = Path(project_root)
        
        self.crnn_dir = self.project_root / "thai-license-plate-recognition-CRNN"
        self.checkpoint_dir = self.project_root / "build-model-th" / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Training parameters
        self.batch_size = 64  # Optimized for RTX 5090
        self.val_batch_size = 16
        self.epochs = 50
        self.initial_lr = 0.001
        
        # GPU optimization settings
        self.configure_gpu()
        
    def configure_gpu(self):
        """Configure GPU settings for RTX 5090 - MANDATORY GPU ONLY"""
        try:
            # Import GPU enforcer
            from enforce_gpu_training import GPUTrainingEnforcer
            
            # Create enforcer instance
            enforcer = GPUTrainingEnforcer()
            
            # MANDATORY: Enforce GPU training policy
            if not enforcer.enforce_gpu_training():
                raise RuntimeError("CRITICAL: GPU TRAINING AUTHORIZATION DENIED - TRAINING ABORTED")
            
            # Configure GPU memory growth
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                logger.info(f"Found {len(gpus)} GPU(s)")
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"Configured memory growth for GPU: {gpu}")
                
                # RTX 5090 specific optimizations
                tf.config.optimizer.set_jit(True)  # Enable XLA JIT compilation
                tf.config.optimizer.set_experimental_options({
                    'layout_optimizer': True,
                    'constant_folding': True,
                    'shape_optimization': True,
                    'remapping': True,
                    'arithmetic_optimization': True,
                    'dependency_optimization': True,
                    'loop_optimization': True,
                    'function_optimization': True,
                    'debug_stripper': True,
                    'disable_model_pruning': False,
                    'scoped_allocator_optimization': True,
                    'pin_to_host_optimization': True,
                    'implementation_selector': True,
                    'auto_mixed_precision': True,
                    'disable_meta_optimizer': False,
                    'min_graph_nodes': -1
                })
                
                logger.info("RTX 5090 GPU optimizations configured")
            else:
                raise RuntimeError("CRITICAL: NO GPU DETECTED - TRAINING NOT ALLOWED")
                
        except Exception as e:
            logger.error(f"GPU configuration failed: {e}")
            raise RuntimeError(f"CRITICAL: GPU TRAINING CANNOT PROCEED - {e}")
    
    def check_dependencies(self):
        """Check if all required dependencies are available"""
        logger.info("Checking dependencies...")
        
        try:
            # Check CRNN directory
            if not self.crnn_dir.exists():
                raise FileNotFoundError(f"CRNN directory not found: {self.crnn_dir}")
            
            # Check required files
            required_files = [
                "Model.py",
                "parameter.py", 
                "Image_Generator.py",
                "training.py"
            ]
            
            missing_files = []
            for file in required_files:
                if not (self.crnn_dir / file).exists():
                    missing_files.append(file)
            
            if missing_files:
                raise FileNotFoundError(f"Missing required files: {missing_files}")
            
            # Check training data
            train_dir = self.crnn_dir / "DB" / "train"
            test_dir = self.crnn_dir / "DB" / "test"
            
            if not train_dir.exists():
                raise FileNotFoundError(f"Training data directory not found: {train_dir}")
            
            if not test_dir.exists():
                raise FileNotFoundError(f"Test data directory not found: {test_dir}")
            
            # Count training samples
            train_images = list(train_dir.glob("*.jpg"))
            test_images = list(test_dir.glob("*.jpg"))
            
            logger.info(f"Training images: {len(train_images)}")
            logger.info(f"Test images: {len(test_images)}")
            
            if len(train_images) == 0:
                raise ValueError("No training images found")
            
            logger.info("✅ All dependencies check passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Dependency check failed: {e}")
            return False
    
    def load_crnn_modules(self):
        """Load CRNN modules with error handling"""
        try:
            # Add CRNN directory to Python path
            sys.path.insert(0, str(self.crnn_dir))
            
            # Import CRNN modules
            from Model import get_Model
            from Image_Generator import TextImageGenerator
            import parameter
            
            logger.info("✅ CRNN modules loaded successfully")
            return get_Model, TextImageGenerator, parameter
            
        except ImportError as e:
            logger.error(f"❌ Failed to import CRNN modules: {e}")
            logger.error(f"Make sure you're in the correct directory: {self.crnn_dir}")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error loading modules: {e}")
            raise
    
    def create_data_generators(self, TextImageGenerator, parameter):
        """Create training and validation data generators"""
        try:
            logger.info("Creating data generators...")
            
            # Training data generator
            train_path = str(self.crnn_dir / "DB" / "train") + "/"
            train_generator = TextImageGenerator(
                train_path,
                parameter.img_w,
                parameter.img_h,
                self.batch_size,
                parameter.downsample_factor
            )
            train_generator.build_data()
            
            # Validation data generator  
            test_path = str(self.crnn_dir / "DB" / "test") + "/"
            val_generator = TextImageGenerator(
                test_path,
                parameter.img_w, 
                parameter.img_h,
                self.val_batch_size,
                parameter.downsample_factor
            )
            val_generator.build_data()
            
            logger.info(f"✅ Training samples: {train_generator.n}")
            logger.info(f"✅ Validation samples: {val_generator.n}")
            logger.info(f"✅ Steps per epoch: {train_generator.n // self.batch_size}")
            logger.info(f"✅ Validation steps: {val_generator.n // self.val_batch_size}")
            
            return train_generator, val_generator
            
        except Exception as e:
            logger.error(f"❌ Failed to create data generators: {e}")
            raise
    
    def create_model(self, get_Model):
        """Create and configure the CRNN model"""
        try:
            logger.info("Creating CRNN model...")
            
            # Create model
            model = get_Model(training=True)
            
            # Try to load previous weights
            weight_files = [
                self.crnn_dir / "LSTM+BN5--thai-v3.hdf5",
                self.crnn_dir / "Model_LSTM+BN5--thai-v3.h5",
                self.checkpoint_dir / "best_model.h5"
            ]
            
            weights_loaded = False
            for weight_file in weight_files:
                if weight_file.exists():
                    try:
                        logger.info(f"Loading weights from: {weight_file}")
                        model.load_weights(str(weight_file))
                        logger.info("✅ Previous weights loaded successfully")
                        weights_loaded = True
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load weights from {weight_file}: {e}")
                        continue
            
            if not weights_loaded:
                logger.info("🆕 Starting with new weights")
            
            # Configure optimizer
            optimizer = Adam(learning_rate=self.initial_lr)
            
            # Compile model
            model.compile(
                loss={'ctc': lambda y_true, y_pred: y_pred},
                optimizer=optimizer
            )
            
            logger.info("✅ Model created and compiled successfully")
            logger.info(f"Model parameters: {model.count_params():,}")
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to create model: {e}")
            raise
    
    def create_callbacks(self):
        """Create training callbacks"""
        try:
            # Model checkpoint
            checkpoint_path = self.checkpoint_dir / "best_model.h5"
            checkpoint = ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor='val_loss',
                verbose=1,
                save_best_only=True,
                mode='min',
                save_freq='epoch'
            )
            
            # Early stopping
            early_stop = EarlyStopping(
                monitor='val_loss',
                min_delta=0.001,
                patience=8,
                mode='min',
                verbose=1,
                restore_best_weights=True
            )
            
            # Learning rate reduction
            lr_reducer = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                mode='min',
                verbose=1,
                min_lr=1e-7
            )
            
            callbacks = [checkpoint, early_stop, lr_reducer]
            
            logger.info("✅ Callbacks created successfully")
            return callbacks
            
        except Exception as e:
            logger.error(f"❌ Failed to create callbacks: {e}")
            raise
    
    def train_model(self):
        """Main training function"""
        try:
            logger.info("🚀 Starting CRNN Training Process")
            logger.info("=" * 60)
            
            # Check dependencies
            if not self.check_dependencies():
                return False
            
            # Load modules
            get_Model, TextImageGenerator, parameter = self.load_crnn_modules()
            
            # Create data generators
            train_generator, val_generator = self.create_data_generators(
                TextImageGenerator, parameter
            )
            
            # Create model
            model = self.create_model(get_Model)
            
            # Print model summary
            logger.info("Model Architecture:")
            model.summary(print_fn=logger.info)
            
            # Create callbacks
            callbacks = self.create_callbacks()
            
            # Calculate steps
            steps_per_epoch = max(1, train_generator.n // self.batch_size)
            validation_steps = max(1, val_generator.n // self.val_batch_size)
            
            logger.info("Training Configuration:")
            logger.info(f"  Epochs: {self.epochs}")
            logger.info(f"  Batch size: {self.batch_size}")
            logger.info(f"  Steps per epoch: {steps_per_epoch}")
            logger.info(f"  Validation steps: {validation_steps}")
            logger.info(f"  Initial learning rate: {self.initial_lr}")
            
            # Start training
            logger.info("🏋️ Starting model training...")
            start_time = time.time()
            
            history = model.fit(
                train_generator.next_batch(),
                steps_per_epoch=steps_per_epoch,
                epochs=self.epochs,
                callbacks=callbacks,
                validation_data=val_generator.next_batch(),
                validation_steps=validation_steps,
                verbose=1
            )
            
            # Training completed
            training_time = time.time() - start_time
            logger.info(f"✅ Training completed in {training_time:.2f} seconds")
            logger.info(f"✅ Training time: {training_time/3600:.2f} hours")
            
            # Save final model
            final_model_path = self.checkpoint_dir / "final_model.h5"
            model.save(str(final_model_path))
            logger.info(f"✅ Final model saved to: {final_model_path}")
            
            # Save training history
            self.save_training_history(history)
            
            return True
            
        except KeyboardInterrupt:
            logger.warning("⚠️ Training interrupted by user")
            return False
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def save_training_history(self, history):
        """Save training history for analysis"""
        try:
            import json
            
            # Convert history to JSON-serializable format
            history_dict = {}
            for key, values in history.history.items():
                history_dict[key] = [float(v) for v in values]
            
            # Save to file
            history_path = self.checkpoint_dir / "training_history.json"
            with open(history_path, 'w') as f:
                json.dump(history_dict, f, indent=2)
            
            logger.info(f"✅ Training history saved to: {history_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save training history: {e}")
    
    def test_model(self):
        """Test the trained model"""
        try:
            logger.info("🧪 Testing trained model...")
            
            # Load best model
            model_path = self.checkpoint_dir / "best_model.h5"
            if not model_path.exists():
                logger.warning("No trained model found")
                return False
            
            # Load modules
            get_Model, TextImageGenerator, parameter = self.load_crnn_modules()
            
            # Create test data generator
            test_path = str(self.crnn_dir / "DB" / "test") + "/"
            test_generator = TextImageGenerator(
                test_path,
                parameter.img_w,
                parameter.img_h,
                1,  # Batch size 1 for testing
                parameter.downsample_factor
            )
            test_generator.build_data()
            
            # Load model
            model = keras.models.load_model(str(model_path), compile=False)
            
            # Test on a few samples
            num_test_samples = min(10, test_generator.n)
            logger.info(f"Testing on {num_test_samples} samples...")
            
            for i in range(num_test_samples):
                try:
                    batch = next(test_generator.next_batch())
                    prediction = model.predict(batch[0], verbose=0)
                    logger.info(f"Test sample {i+1}/{num_test_samples}: Prediction shape {prediction.shape}")
                except Exception as e:
                    logger.warning(f"Test sample {i+1} failed: {e}")
            
            logger.info("✅ Model testing completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model testing failed: {e}")
            return False

def main():
    """Main function"""
    try:
        # Print banner
        print("=" * 80)
        print("🇹🇭 Thai CRNN Training System - RTX 5090 Optimized")
        print("=" * 80)
        
        # Create trainer
        trainer = RTX5090CRNNTrainer()
        
        # Start training
        success = trainer.train_model()
        
        if success:
            print("\n🎉 Training completed successfully!")
            
            # Test model
            trainer.test_model()
            
            print("\n📊 Training Results:")
            print(f"  ✅ Model saved to: {trainer.checkpoint_dir}")
            print(f"  📝 Logs saved to: crnn_training.log")
            print(f"  📈 History saved to: {trainer.checkpoint_dir}/training_history.json")
            
        else:
            print("\n❌ Training failed. Check logs for details.")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
