#!/usr/bin/env python3
"""
Install Thai OCR Dependencies
Installation script for Thai OCR project dependencies
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description=""):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"📦 {description}")
    print(f"{'='*50}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 10:
        print("⚠️  Warning: Python 3.10+ recommended for RTX 5090 compatibility")
        return False
    return True

def install_base_packages():
    """Install base packages"""
    packages = [
        "pip --upgrade",
        "wheel",
        "setuptools",
        "numpy>=1.21.0",
        "opencv-python>=4.8.0", 
        "pillow>=9.0.0",
        "matplotlib>=3.5.0"
    ]
    
    for package in packages:
        if not run_command(f"pip install {package}", f"Installing {package.split()[0]}"):
            return False
    return True

def install_paddlepaddle():
    """Install PaddlePaddle GPU version for RTX 5090"""
    print("\n🚀 Installing PaddlePaddle GPU for RTX 5090...")
    
    # Try official GPU version first
    if run_command("pip install paddlepaddle-gpu>=2.6.0", "Installing PaddlePaddle GPU"):
        return True
    
    # Fallback to CPU version if GPU fails
    print("⚠️  GPU version failed, trying CPU version...")
    return run_command("pip install paddlepaddle>=2.6.0", "Installing PaddlePaddle CPU")

def install_paddleocr():
    """Install PaddleOCR"""
    return run_command("pip install paddleocr>=2.7.0", "Installing PaddleOCR")

def install_thai_specific():
    """Install Thai-specific packages"""
    packages = [
        "pythainlp",
        "fonttools",
        "Wand",  # For ImageMagick
        "reportlab",  # For PDF generation
    ]
    
    for package in packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️  Warning: Failed to install {package}, continuing...")
    return True

def install_ml_packages():
    """Install ML and data science packages"""
    packages = [
        "scikit-learn",
        "pandas",
    ]
    
    # Try TensorFlow installation with better error handling
    print("\n🔧 Attempting TensorFlow installation...")
    tf_packages = [
        "tensorflow>=2.15.0",  # Try latest first
        "tensorflow-cpu>=2.15.0",  # CPU fallback
        "keras>=2.15.0",  # Standalone Keras
    ]
    
    tf_installed = False
    for package in tf_packages:
        if run_command(f"pip install {package}", f"Installing {package}"):
            tf_installed = True
            break
        print(f"⚠️  {package} failed, trying next option...")
    
    if not tf_installed:
        print("⚠️  TensorFlow installation failed. CRNN functionality may be limited.")
        print("💡 You can install TensorFlow manually later if needed.")
    
    # Install other ML packages
    for package in packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️  Warning: Failed to install {package}, continuing...")
    
    # Always try to install h5py (essential for model loading)
    run_command("pip install h5py>=3.1.0", "Installing h5py")
    
    return True

def setup_environment():
    """Setup environment variables for RTX 5090"""
    print("\n🔧 Setting up RTX 5090 environment variables...")
    
    env_vars = {
        "FLAGS_fraction_of_gpu_memory_to_use": "0.8",
        "FLAGS_conv_workspace_size_limit": "512", 
        "FLAGS_cudnn_deterministic": "true"
    }
    
    for var, value in env_vars.items():
        os.environ[var] = value
        print(f"✅ Set {var}={value}")

def verify_installation():
    """Verify that installations were successful"""
    print("\n🧪 Verifying installations...")
    
    tests = [
        ("import numpy; print(f'NumPy: {numpy.__version__}')", "NumPy"),
        ("import cv2; print(f'OpenCV: {cv2.__version__}')", "OpenCV"),
        ("import PIL; print(f'Pillow: {PIL.__version__}')", "Pillow"),
        ("import paddle; print(f'PaddlePaddle: {paddle.__version__}')", "PaddlePaddle"),
        ("import paddleocr; print('PaddleOCR: OK')", "PaddleOCR"),
        ("import pythainlp; print(f'PyThaiNLP: {pythainlp.__version__}')", "PyThaiNLP"),
    ]
    
    success_count = 0
    for test_code, name in tests:
        try:
            result = subprocess.run([sys.executable, "-c", test_code], 
                                  capture_output=True, text=True, check=True,
                                  encoding='utf-8', errors='replace')
            print(f"✅ {name}: {result.stdout.strip()}")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"❌ {name}: Failed - {e.stderr.strip() if e.stderr else 'Unknown error'}")
    
    print(f"\n📊 Installation Summary: {success_count}/{len(tests)} packages verified")
    return success_count >= 4  # Allow some failures but require core packages

def test_gpu_support():
    """Test GPU support specifically"""
    print("\n🎮 Testing GPU support...")
    
    gpu_test = """
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
import paddle
print(f'CUDA compiled: {paddle.device.is_compiled_with_cuda()}')
print(f'GPU count: {paddle.device.cuda.device_count()}')
if paddle.device.cuda.device_count() > 0:
    print(f'GPU device: {paddle.device.get_device()}')
    try:
        # Test GPU tensor operation
        x = paddle.ones([2, 2], dtype='float32')
        print('GPU tensor creation: OK')
    except Exception as e:
        print(f'GPU tensor test failed: {e}')
        print('Note: RTX 5090 (Compute 12.0) may need custom PaddlePaddle build')
        print('Fallback: Use CPU version for development')
else:
    print('No GPU detected')
"""
    
    try:
        result = subprocess.run([sys.executable, "-c", gpu_test], 
                              capture_output=True, text=True, check=True, 
                              encoding='utf-8', errors='replace')
        print(result.stdout)
        return "CUDA compiled: True" in result.stdout
    except subprocess.CalledProcessError as e:
        print(f"GPU test failed: {e.stderr}")
        print("\n🔧 RTX 5090 Compatibility Note:")
        print("- Compute Capability 12.0 requires custom PaddlePaddle build")
        print("- Current installation will use CPU mode")
        print("- GPU functionality limited until official support")
        return False

def main():
    """Main installation process"""
    print("🚀 Thai OCR Dependencies Installation")
    print("=" * 50)
    print("Installing packages for RTX 5090 + PaddleOCR + Thai language support")
    
    # Check Python version
    if not check_python_version():
        print("⚠️  Continuing with current Python version...")
    
    # Installation steps
    steps = [
        ("Installing base packages", install_base_packages),
        ("Installing PaddlePaddle", install_paddlepaddle),
        ("Installing PaddleOCR", install_paddleocr),
        ("Installing Thai-specific packages", install_thai_specific),
        ("Installing ML packages", install_ml_packages),
        ("Setting up environment", setup_environment),
        ("Verifying installation", verify_installation),
        ("Testing GPU support", test_gpu_support),
    ]
    
    failed_steps = []
    for step_name, step_func in steps:
        print(f"\n{'🔄' if step_func != verify_installation else '🧪'} {step_name}...")
        if not step_func():
            failed_steps.append(step_name)
            if step_name in ["Installing PaddlePaddle", "Installing PaddleOCR"]:
                print(f"❌ Critical step failed: {step_name}")
                print("Installation cannot continue.")
                return False
    
    # Final summary
    print("\n" + "=" * 50)
    print("📋 INSTALLATION SUMMARY")
    print("=" * 50)
    
    if not failed_steps:
        print("🎉 All installations completed successfully!")
        print("✅ Your Thai OCR environment is ready for RTX 5090")
    else:
        print(f"⚠️  Installation completed with {len(failed_steps)} warnings:")
        for step in failed_steps:
            print(f"   - {step}")
        print("\n💡 You can continue development, but some features may be limited.")
    
    print("\n📚 Next steps:")
    print("1. Run 'Test PaddleOCR Installation' task")
    print("2. Run 'Check GPU Status' task") 
    print("3. Generate Thai text dataset")
    print("4. Start OCR training")
    
    return len(failed_steps) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
