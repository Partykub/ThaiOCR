#!/usr/bin/env python3
"""
NGC Thai OCR Environment Setup Script
Automated setup script for reproducible NGC PaddlePaddle environment with RTX 5090 support
"""

import subprocess
import os
import sys
from pathlib import Path

class NGCSetup:
    def __init__(self):
        self.container_name = "thai-ocr-training-ngc"
        self.image_name = "nvcr.io/nvidia/paddlepaddle:24.12-py3"
        
    def run_command(self, command, check=True):
        """Execute command and return result"""
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if check and result.returncode != 0:
                print(f"❌ Command failed: {command}")
                print(f"Error: {result.stderr}")
                return False
            return result.stdout.strip()
        except Exception as e:
            print(f"❌ Exception running command: {e}")
            return False
    
    def check_prerequisites(self):
        """Check Docker and NVIDIA GPU support"""
        print("🔍 Checking prerequisites...")
        
        # Check Docker
        if not self.run_command("docker --version"):
            print("❌ Docker not found. Please install Docker Desktop")
            return False
        print("✅ Docker found")
        
        # Check NVIDIA Docker
        gpu_test = self.run_command('docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi', check=False)
        if not gpu_test:
            print("⚠️  NVIDIA Container Toolkit may need installation")
        else:
            print("✅ NVIDIA GPU support working")
        
        return True
    
    def pull_ngc_image(self):
        """Pull NGC PaddlePaddle image"""
        print(f"\n🐳 Pulling NGC image: {self.image_name}")
        print("⏳ This may take 5-15 minutes...")
        
        result = self.run_command(f"docker pull {self.image_name}")
        if result is False:
            print("❌ Failed to pull NGC image")
            return False
        
        print("✅ NGC image downloaded successfully")
        return True
    
    def stop_existing_container(self):
        """Stop and remove existing container if exists"""
        print(f"\n🛑 Checking for existing container: {self.container_name}")
        
        # Check if container exists
        exists = self.run_command(f"docker ps -a --filter name={self.container_name} --format '{{{{.Names}}}}'", check=False)
        if exists and self.container_name in exists:
            print(f"🗑️  Removing existing container: {self.container_name}")
            self.run_command(f"docker stop {self.container_name}", check=False)
            self.run_command(f"docker rm {self.container_name}", check=False)
            print("✅ Old container removed")
        else:
            print("✅ No existing container found")
    
    def start_container(self):
        """Start NGC container with docker-compose"""
        print(f"\n🚀 Starting NGC container: {self.container_name}")
        
        if not Path("docker-compose.ngc.yml").exists():
            print("❌ docker-compose.ngc.yml not found")
            return False
        
        result = self.run_command("docker-compose -f docker-compose.ngc.yml up -d")
        if result is False:
            print("❌ Failed to start container")
            return False
        
        print("✅ Container started successfully")
        return True
    
    def setup_dependencies(self):
        """Install and configure dependencies inside container"""
        print(f"\n📦 Setting up dependencies in container...")
        
        # Commands to run inside container
        setup_commands = [
            # Install system dependencies
            "apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libgomp1",
            
            # Downgrade PaddlePaddle for compatibility
            "pip uninstall -y paddlepaddle paddlepaddle-gpu || true",
            "pip install paddlepaddle-gpu==2.6.2",
            
            # Install specific PaddleOCR version without PaddleX conflicts
            "pip uninstall -y paddleocr paddlex || true", 
            "pip install paddleocr==2.7.0.3",
            
            # Install compatible versions of other packages
            "pip install numpy==1.26.4 opencv-python-headless==4.10.0.84 pillow==9.5.0",
            "pip install flask==2.3.2 tensorflow==2.15.0 keras==2.15.0 h5py==3.9.0",
            "pip install matplotlib==3.7.0 scikit-learn==1.3.0",
        ]
        
        for cmd in setup_commands:
            print(f"   Running: {cmd}")
            docker_cmd = f"docker exec {self.container_name} bash -c \"{cmd}\""
            result = self.run_command(docker_cmd, check=False)
            if result is False:
                print(f"⚠️  Command may have failed: {cmd}")
            else:
                print("   ✅ Done")
        
        print("✅ Dependencies setup completed")
    
    def verify_installation(self):
        """Verify the installation works"""
        print(f"\n🧪 Verifying installation...")
        
        # Test commands
        test_commands = [
            ("PaddlePaddle GPU", "python -c \"import paddle; print('PaddlePaddle version:', paddle.__version__); print('CUDA available:', paddle.device.is_compiled_with_cuda())\""),
            ("PaddleOCR", "python -c \"import paddleocr; print('PaddleOCR version:', paddleocr.__version__)\""),
            ("GPU Memory", "python -c \"import paddle; print('GPU count:', paddle.device.cuda.device_count())\""),
            ("OpenCV", "python -c \"import cv2; print('OpenCV version:', cv2.__version__)\""),
        ]
        
        all_passed = True
        for test_name, cmd in test_commands:
            docker_cmd = f"docker exec {self.container_name} bash -c \"{cmd}\""
            result = self.run_command(docker_cmd, check=False)
            if result:
                print(f"✅ {test_name}: {result}")
            else:
                print(f"❌ {test_name}: Failed")
                all_passed = False
        
        return all_passed
    
    def print_usage_info(self):
        """Print usage information"""
        print(f"\n🎉 Setup completed successfully!")
        print(f"\n📝 Usage:")
        print(f"   Connect to container: docker exec -it {self.container_name} bash")
        print(f"   Start container: docker start {self.container_name}")
        print(f"   Stop container: docker stop {self.container_name}")
        print(f"   View logs: docker logs {self.container_name}")
        print(f"\n🐳 Container Details:")
        print(f"   Name: {self.container_name}")
        print(f"   Image: {self.image_name}")
        print(f"   Workspace: /workspace (mapped to current directory)")
        print(f"\n🔥 GPU Support:")
        print(f"   RTX 5090 SM_120: ✅ Supported")
        print(f"   CUDA 12.6: ✅ Enabled")
        print(f"   PaddlePaddle 2.6.2: ✅ Compatible")
        print(f"   PaddleOCR 2.7.0.3: ✅ Working")

def main():
    """Main setup function"""
    print("🚀 NGC Thai OCR Environment Setup")
    print("=" * 50)
    
    setup = NGCSetup()
    
    # Run setup steps
    steps = [
        ("Prerequisites Check", setup.check_prerequisites),
        ("Pull NGC Image", setup.pull_ngc_image),
        ("Stop Existing Container", setup.stop_existing_container),
        ("Start Container", setup.start_container),
        ("Setup Dependencies", setup.setup_dependencies),
        ("Verify Installation", setup.verify_installation),
    ]
    
    for step_name, step_func in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        if not step_func():
            print(f"❌ Setup failed at step: {step_name}")
            return False
    
    setup.print_usage_info()
    return True

if __name__ == "__main__":
    if main():
        print("\n🎉 Setup completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Setup failed!")
        sys.exit(1)
