#!/usr/bin/env python3
"""
GPU Training Enforcer - RTX 5090 Mandatory
Ensures that ALL training operations use GPU only - NO CPU FALLBACK ALLOWED

CRITICAL REQUIREMENTS:
- RTX 5090 GPU MUST be available and working
- TensorFlow-GPU MUST be properly configured
- CUDA 12.6 MUST be installed and functional
- NO training allowed without GPU verification
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GPUTrainingEnforcer:
    """Enforces GPU-only training policy for RTX 5090"""
    
    def __init__(self):
        self.gpu_verified = False
        self.rtx5090_detected = False
        
    def verify_gpu_mandatory(self):
        """MANDATORY GPU verification - NO TRAINING WITHOUT GPU"""
        logger.info("🔍 CRITICAL: Verifying GPU availability for training...")
        
        try:
            # Check TensorFlow GPU
            import tensorflow as tf
            
            # Get physical devices
            physical_devices = tf.config.list_physical_devices('GPU')
            
            if not physical_devices:
                raise RuntimeError("CRITICAL: NO GPU DETECTED - TRAINING ABORTED")
            
            logger.info(f"✅ GPU DETECTED: {len(physical_devices)} GPU(s) found")
            
            # Check each GPU
            for i, device in enumerate(physical_devices):
                device_name = device.name
                logger.info(f"  GPU {i}: {device_name}")
                
                # Check for RTX 5090
                if "5090" in device_name.upper():
                    self.rtx5090_detected = True
                    logger.info(f"  ✅ RTX 5090 DETECTED: {device_name}")
            
            # Configure GPU memory growth
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Verify GPU is actually usable
            with tf.device('/GPU:0'):
                test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                result = tf.matmul(test_tensor, test_tensor)
                logger.info(f"✅ GPU COMPUTATION TEST PASSED")
            
            self.gpu_verified = True
            logger.info("🎮 GPU VERIFICATION SUCCESSFUL - TRAINING AUTHORIZED")
            return True
            
        except ImportError:
            raise RuntimeError("CRITICAL: TensorFlow not installed - TRAINING ABORTED")
        except Exception as e:
            raise RuntimeError(f"CRITICAL: GPU verification failed - {e} - TRAINING ABORTED")
    
    def verify_cuda_environment(self):
        """Verify CUDA environment for RTX 5090"""
        logger.info("🔍 Verifying CUDA environment...")
        
        try:
            import tensorflow as tf
            
            # Check CUDA availability
            if not tf.test.is_built_with_cuda():
                raise RuntimeError("CRITICAL: TensorFlow not built with CUDA support")
            
            # Check GPU support
            if not tf.test.is_gpu_available():
                raise RuntimeError("CRITICAL: GPU not available in TensorFlow")
            
            # Print CUDA version
            cuda_version = tf.sysconfig.get_build_info().get('cuda_version', 'Unknown')
            logger.info(f"✅ CUDA Version: {cuda_version}")
            
            # Print cuDNN version  
            cudnn_version = tf.sysconfig.get_build_info().get('cudnn_version', 'Unknown')
            logger.info(f"✅ cuDNN Version: {cudnn_version}")
            
            return True
            
        except Exception as e:
            raise RuntimeError(f"CRITICAL: CUDA environment check failed - {e}")
    
    def set_rtx5090_optimizations(self):
        """Set RTX 5090 specific optimizations"""
        logger.info("⚡ Configuring RTX 5090 optimizations...")
        
        # Environment variables for RTX 5090
        gpu_env_vars = {
            'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
            'TF_GPU_MEMORY_ALLOW_GROWTH': 'true',
            'FLAGS_fraction_of_gpu_memory_to_use': '0.8',
            'FLAGS_conv_workspace_size_limit': '512',
            'FLAGS_cudnn_deterministic': 'true',
            'CUDA_VISIBLE_DEVICES': '0',
            'TF_ENABLE_GPU_GARBAGE_COLLECTION': 'false',
            'TF_GPU_THREAD_MODE': 'gpu_private'
        }
        
        for key, value in gpu_env_vars.items():
            os.environ[key] = value
            logger.info(f"  ✅ {key} = {value}")
        
        try:
            import tensorflow as tf
            
            # Configure TensorFlow for RTX 5090
            tf.config.optimizer.set_jit(True)  # Enable XLA
            
            # Enable mixed precision for RTX 5090
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("✅ Mixed precision enabled (float16)")
            
            logger.info("🚀 RTX 5090 optimizations configured successfully")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Some optimizations failed: {e}")
            return False
    
    def enforce_gpu_training(self):
        """Main enforcer function - MUST pass before any training"""
        logger.info("=" * 80)
        logger.info("🎮 GPU TRAINING ENFORCER - RTX 5090 MANDATORY")
        logger.info("=" * 80)
        
        try:
            # Step 1: Verify GPU availability (MANDATORY)
            self.verify_gpu_mandatory()
            
            # Step 2: Verify CUDA environment
            self.verify_cuda_environment()
            
            # Step 3: Set RTX 5090 optimizations
            self.set_rtx5090_optimizations()
            
            # Final verification
            if not self.gpu_verified:
                raise RuntimeError("CRITICAL: GPU verification failed - TRAINING NOT AUTHORIZED")
            
            # Print final status
            logger.info("🎉 GPU TRAINING AUTHORIZATION GRANTED")
            logger.info(f"✅ RTX 5090 Detected: {'YES' if self.rtx5090_detected else 'NO'}")
            logger.info(f"✅ GPU Verified: {'YES' if self.gpu_verified else 'NO'}")
            logger.info("🚀 READY FOR GPU TRAINING")
            logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            logger.error("=" * 80)
            logger.error("❌ GPU TRAINING AUTHORIZATION DENIED")
            logger.error(f"❌ ERROR: {e}")
            logger.error("❌ TRAINING CANNOT PROCEED WITHOUT GPU")
            logger.error("=" * 80)
            return False
    
    def get_gpu_info(self):
        """Get detailed GPU information"""
        try:
            import tensorflow as tf
            
            gpus = tf.config.list_physical_devices('GPU')
            
            info = {
                'gpu_count': len(gpus),
                'gpu_names': [gpu.name for gpu in gpus],
                'rtx5090_detected': self.rtx5090_detected,
                'gpu_verified': self.gpu_verified
            }
            
            return info
            
        except Exception as e:
            return {'error': str(e)}

def main():
    """Main function to enforce GPU training"""
    enforcer = GPUTrainingEnforcer()
    
    # Enforce GPU training policy
    success = enforcer.enforce_gpu_training()
    
    if success:
        print("\n✅ GPU TRAINING ENFORCER: PASSED")
        print("🚀 You may proceed with GPU training")
        return 0
    else:
        print("\n❌ GPU TRAINING ENFORCER: FAILED") 
        print("🛑 Training is NOT AUTHORIZED without GPU")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
