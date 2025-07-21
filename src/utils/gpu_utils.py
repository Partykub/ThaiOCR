#!/usr/bin/env python3
"""
🔍 GPU Detection and Setup Guide for Thai CRNN Training
ตรวจสอบ GPU และแนะนำการติดตั้งสำหรับการเทรน CRNN
"""

import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_gpu_availability():
    """ตรวจสอบ GPU และแนะนำการติดตั้ง"""
    
    logger.info("🔍 GPU DETECTION AND SETUP GUIDE")
    logger.info("=" * 50)
    
    # 1. Check TensorFlow GPU
    try:
        import tensorflow as tf
        logger.info("✅ TensorFlow imported successfully")
        
        # Check GPU devices
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            logger.info(f"🎮 GPU DETECTED: {len(gpus)} device(s)")
            for i, gpu in enumerate(gpus):
                logger.info(f"   GPU {i}: {gpu}")
                
            # Test GPU functionality
            try:
                with tf.device('/GPU:0'):
                    test_tensor = tf.constant([1.0, 2.0, 3.0])
                    result = tf.reduce_sum(test_tensor).numpy()
                logger.info(f"✅ GPU TEST PASSED: Result = {result}")
                return True
                
            except Exception as e:
                logger.error(f"❌ GPU TEST FAILED: {e}")
                
        else:
            logger.warning("⚠️ NO GPU DETECTED")
            
    except ImportError:
        logger.error("❌ TensorFlow not installed or not GPU version")
    except Exception as e:
        logger.error(f"❌ TensorFlow GPU check failed: {e}")
    
    # 2. Check CUDA
    logger.info("\n🔧 CUDA INSTALLATION CHECK")
    logger.info("-" * 30)
    
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("✅ NVIDIA-SMI found - GPU driver installed")
            # Parse nvidia-smi output for GPU info
            lines = result.stdout.split('\n')
            for line in lines:
                if 'RTX' in line or 'GeForce' in line or 'Tesla' in line:
                    logger.info(f"   {line.strip()}")
        else:
            logger.warning("⚠️ nvidia-smi not found - NVIDIA driver may not be installed")
    except FileNotFoundError:
        logger.warning("⚠️ nvidia-smi command not found")
    except Exception as e:
        logger.error(f"❌ CUDA check failed: {e}")
    
    # 3. Environment variables
    logger.info("\n🌐 ENVIRONMENT VARIABLES")
    logger.info("-" * 25)
    
    cuda_path = os.environ.get('CUDA_PATH')
    if cuda_path:
        logger.info(f"✅ CUDA_PATH: {cuda_path}")
    else:
        logger.warning("⚠️ CUDA_PATH not set")
        
    path = os.environ.get('PATH', '')
    if 'cuda' in path.lower():
        logger.info("✅ CUDA found in PATH")
    else:
        logger.warning("⚠️ CUDA not found in PATH")
    
    return False

def provide_setup_instructions():
    """แนะนำการติดตั้งสำหรับ RTX 5090"""
    
    logger.info("\n🚀 SETUP INSTRUCTIONS FOR RTX 5090 TRAINING")
    logger.info("=" * 50)
    
    logger.info("📋 REQUIRED COMPONENTS:")
    logger.info("1. 🎮 RTX 5090 GPU (24GB VRAM)")
    logger.info("2. 🔧 CUDA Toolkit 12.6")
    logger.info("3. 📦 cuDNN 9.0")
    logger.info("4. 🐍 TensorFlow-GPU 2.15+")
    
    logger.info("\n📥 INSTALLATION STEPS:")
    logger.info("Step 1: Install NVIDIA Driver")
    logger.info("   Download from: https://www.nvidia.com/drivers")
    logger.info("   Minimum version: 560.xx for RTX 5090")
    
    logger.info("\nStep 2: Install CUDA Toolkit 12.6")
    logger.info("   Download: https://developer.nvidia.com/cuda-toolkit")
    logger.info("   Add to PATH: C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6\\bin")
    
    logger.info("\nStep 3: Install cuDNN 9.0")
    logger.info("   Download: https://developer.nvidia.com/cudnn")
    logger.info("   Extract to CUDA directory")
    
    logger.info("\nStep 4: Install TensorFlow-GPU")
    logger.info("   pip install tensorflow[and-cuda]")
    logger.info("   or")
    logger.info("   pip install tensorflow-gpu")
    
    logger.info("\n🔧 VERIFICATION COMMANDS:")
    logger.info("nvidia-smi")
    logger.info("nvcc --version")
    logger.info("python -c \"import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))\"")
    
    logger.info("\n⚠️ TROUBLESHOOTING:")
    logger.info("- Restart computer after driver installation")
    logger.info("- Check Windows GPU Task Manager")
    logger.info("- Verify CUDA and cuDNN versions compatibility")
    logger.info("- Run as Administrator if needed")

def provide_training_alternatives():
    """แนะนำทางเลือกสำหรับการเทรน"""
    
    logger.info("\n🔄 TRAINING ALTERNATIVES")
    logger.info("=" * 30)
    
    logger.info("Option 1: 🏠 Local GPU Setup")
    logger.info("   - Install RTX 5090 + CUDA 12.6")
    logger.info("   - Best performance and control")
    logger.info("   - One-time setup cost")
    
    logger.info("\nOption 2: ☁️ Cloud GPU Training")
    logger.info("   - Google Colab Pro (V100/A100)")
    logger.info("   - AWS EC2 P3/P4 instances")
    logger.info("   - Azure NC/ND series")
    
    logger.info("\nOption 3: 🎯 Model Development Mode")
    logger.info("   - Use smaller models for testing")
    logger.info("   - Reduce dataset size for debugging")
    logger.info("   - Save final training for GPU environment")
    
    logger.info("\n❌ NOT RECOMMENDED:")
    logger.info("   - CPU training (too slow, forbidden by project policy)")
    logger.info("   - Integrated graphics (insufficient memory)")

def main():
    """Main function"""
    logger.info("🎯 THAI CRNN GPU SETUP CHECKER")
    logger.info("Checking system for RTX 5090 Thai OCR training compatibility")
    logger.info("=" * 70)
    
    # Check current GPU status
    gpu_available = check_gpu_availability()
    
    if gpu_available:
        logger.info("\n🎉 SUCCESS: GPU READY FOR TRAINING!")
        logger.info("You can proceed with Thai CRNN training")
        logger.info("Run: python build-model-th/train_thai_crnn.py")
    else:
        logger.info("\n❌ SETUP REQUIRED: GPU NOT READY")
        provide_setup_instructions()
        provide_training_alternatives()
        
        logger.info("\n🚫 TRAINING BLOCKED:")
        logger.info("Thai CRNN training requires GPU - CPU training is not allowed")
        logger.info("Please complete GPU setup before proceeding")
    
    logger.info("\n" + "=" * 70)
    logger.info("🔗 For more help, see: docs/installation_guide.md")

if __name__ == "__main__":
    main()
