#!/usr/bin/env python3
"""
RTX 5090 Compatibility Helper
Special configuration for RTX 5090 + PaddlePaddle
"""

import subprocess
import sys
import os

def check_rtx_5090():
    """Check if RTX 5090 is present"""
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"], 
                              capture_output=True, text=True, check=True)
        gpu_info = result.stdout.strip()
        print(f"🎮 Detected GPU: {gpu_info}")
        
        if "5090" in gpu_info and "12.0" in gpu_info:
            print("✅ RTX 5090 with Compute Capability 12.0 detected")
            return True
    except:
        print("❌ Could not detect GPU information")
    return False

def setup_cpu_fallback():
    """Configure PaddlePaddle for CPU-only operation"""
    print("\n🔧 Setting up CPU fallback configuration...")
    
    # Set environment variables for CPU operation
    cpu_env = {
        "CUDA_VISIBLE_DEVICES": "-1",  # Disable CUDA
        "PADDLE_ONLY_CPU": "1",        # Force CPU mode
        "FLAGS_use_mkldnn": "true",    # Use Intel MKL-DNN for CPU acceleration
    }
    
    for var, value in cpu_env.items():
        os.environ[var] = value
        print(f"✅ Set {var}={value}")

def create_paddle_config():
    """Create PaddlePaddle configuration file"""
    config_content = """
# PaddlePaddle Configuration for RTX 5090 Compatibility
# This file configures PaddlePaddle to work with unsupported GPU architectures

import os
import paddle

# Force CPU mode when GPU is not supported
def configure_paddle():
    try:
        if paddle.device.is_compiled_with_cuda():
            gpu_count = paddle.device.cuda.device_count()
            if gpu_count > 0:
                # Try to use GPU
                paddle.device.set_device('gpu:0')
                # Test basic operation
                x = paddle.ones([2, 2])
                print("GPU mode: OK")
                return "gpu"
    except Exception as e:
        print(f"GPU mode failed: {e}")
        print("Falling back to CPU mode...")
    
    # Use CPU mode
    paddle.device.set_device('cpu')
    print("CPU mode: OK")
    return "cpu"

# Auto-configure on import
device_mode = configure_paddle()
print(f"PaddlePaddle configured for: {device_mode}")
"""
    
    # Create config directory
    config_dir = os.path.expanduser("~/.paddle")
    os.makedirs(config_dir, exist_ok=True)
    
    config_file = os.path.join(config_dir, "rtx5090_config.py")
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"✅ Created PaddlePaddle config: {config_file}")

def test_cpu_performance():
    """Test CPU performance optimization"""
    print("\n🧪 Testing CPU performance...")
    
    cpu_test = """
import paddle
import time
import os

# Configure CPU settings
os.environ['FLAGS_use_mkldnn'] = 'true'
os.environ['FLAGS_mkldnn_cache_capacity'] = '1024'

paddle.device.set_device('cpu')

# Test CPU performance
start_time = time.time()
x = paddle.randn([1000, 1000])
y = paddle.randn([1000, 1000])
z = paddle.matmul(x, y)
end_time = time.time()

print(f"CPU Matrix multiplication (1000x1000): {end_time - start_time:.3f} seconds")
print("CPU optimization: OK")
"""
    
    try:
        result = subprocess.run([sys.executable, "-c", cpu_test], 
                              capture_output=True, text=True, check=True,
                              encoding='utf-8', errors='replace')
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"CPU test failed: {e.stderr}")
        return False

def install_cpu_optimizations():
    """Install CPU optimization packages"""
    print("\n📦 Installing CPU optimization packages...")
    
    packages = [
        "intel-openmp",  # Intel OpenMP for better CPU performance
        "mkl",           # Intel Math Kernel Library
    ]
    
    for package in packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True)
            print(f"✅ Installed {package}")
        except:
            print(f"⚠️  Could not install {package} (optional)")

def create_ocr_wrapper():
    """Create OCR wrapper that handles RTX 5090 compatibility"""
    wrapper_content = '''
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
        kwargs['use_gpu'] = False
        
        # Suppress warnings about GPU
        warnings.filterwarnings("ignore", category=UserWarning)
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
        print("🔧 ThaiOCR: Using CPU mode for RTX 5090 compatibility")
        
        # Initialize PaddleOCR with Thai language
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='th',
            show_log=False,
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
'''
    
    wrapper_file = "thai_ocr_rtx5090.py"
    with open(wrapper_file, 'w', encoding='utf-8') as f:
        f.write(wrapper_content)
    
    print(f"✅ Created OCR wrapper: {wrapper_file}")

def main():
    """Main RTX 5090 compatibility setup"""
    print("🎮 RTX 5090 Compatibility Helper")
    print("=" * 50)
    
    # Check if RTX 5090 is present
    is_rtx_5090 = check_rtx_5090()
    
    if is_rtx_5090:
        print("\n⚠️  RTX 5090 detected with unsupported Compute Capability 12.0")
        print("🔧 Setting up CPU fallback configuration...")
        
        # Setup CPU fallback
        setup_cpu_fallback()
        create_paddle_config()
        install_cpu_optimizations()
        test_cpu_performance()
        create_ocr_wrapper()
        
        print("\n✅ RTX 5090 compatibility setup completed!")
        print("💡 Use thai_ocr_rtx5090.py for Thai OCR with automatic CPU fallback")
        
    else:
        print("\n💡 RTX 5090 not detected, standard GPU configuration should work")
    
    print("\n📚 Next steps:")
    print("1. Test PaddleOCR with: python thai_ocr_rtx5090.py")
    print("2. Run OCR on sample images")
    print("3. Train custom models (CPU mode)")

if __name__ == "__main__":
    main()
