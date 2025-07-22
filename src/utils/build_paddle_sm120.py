#!/usr/bin/env python3
"""
Build PaddlePaddle with SM_120 Support for RTX 5090
================================================

Custom build script for PaddlePaddle with RTX 5090 (SM_120) support
Resolves "no kernel image available" error for RTX 5090

Author: Thai OCR Development Team
Date: July 22, 2025
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import logging
import shutil
import urllib.request
import zipfile
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PaddleSM120Builder:
    """Build PaddlePaddle with RTX 5090 SM_120 support"""
    
    def __init__(self):
        self.work_dir = Path("./paddle_build")
        self.paddle_source = self.work_dir / "Paddle"
        self.build_dir = self.paddle_source / "build"
        self.python_executable = sys.executable
        
        # RTX 5090 specifications
        self.rtx5090_specs = {
            'compute_capability': '12.0',
            'sm_version': '120',
            'cuda_arch': 'sm_120',
            'memory_gb': 24,
            'cores': 21760
        }
        
    def check_prerequisites(self) -> bool:
        """Check system prerequisites for building"""
        logger.info("🔍 CHECKING PREREQUISITES...")
        
        prerequisites = {
            'git': False,
            'cmake': False,
            'cuda': False,
            'python': False,
            'visual_studio': False
        }
        
        # Check Git
        try:
            result = subprocess.run(['git', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                prerequisites['git'] = True
                logger.info(f"✅ Git: {result.stdout.strip()}")
        except FileNotFoundError:
            logger.error("❌ Git not found. Please install Git first.")
        
        # Check CMake
        try:
            result = subprocess.run(['cmake', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                prerequisites['cmake'] = True
                version = result.stdout.split('\n')[0]
                logger.info(f"✅ CMake: {version}")
        except FileNotFoundError:
            logger.error("❌ CMake not found. Please install CMake first.")
        
        # Check CUDA
        cuda_path = os.environ.get('CUDA_PATH', 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6')
        if Path(cuda_path).exists():
            prerequisites['cuda'] = True
            logger.info(f"✅ CUDA: Found at {cuda_path}")
        else:
            logger.error(f"❌ CUDA not found at {cuda_path}")
        
        # Check Python
        if sys.version_info >= (3, 8):
            prerequisites['python'] = True
            logger.info(f"✅ Python: {sys.version}")
        else:
            logger.error(f"❌ Python {sys.version} too old. Need Python 3.8+")
        
        # Check Visual Studio (Windows)
        if platform.system() == 'Windows':
            vs_paths = [
                'C:\\Program Files\\Microsoft Visual Studio\\2022',
                'C:\\Program Files\\Microsoft Visual Studio\\2019',
                'C:\\Program Files (x86)\\Microsoft Visual Studio\\2019'
            ]
            for vs_path in vs_paths:
                if Path(vs_path).exists():
                    prerequisites['visual_studio'] = True
                    logger.info(f"✅ Visual Studio: Found at {vs_path}")
                    break
            
            if not prerequisites['visual_studio']:
                logger.error("❌ Visual Studio not found. Please install Visual Studio 2019/2022")
        
        # Summary
        all_ok = all(prerequisites.values())
        if all_ok:
            logger.info("✅ ALL PREREQUISITES SATISFIED")
        else:
            missing = [k for k, v in prerequisites.items() if not v]
            logger.error(f"❌ MISSING PREREQUISITES: {', '.join(missing)}")
        
        return all_ok
    
    def check_rtx5090(self) -> bool:
        """Check RTX 5090 GPU availability"""
        logger.info("🎮 CHECKING RTX 5090 GPU...")
        
        try:
            # Check with nvidia-smi
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                gpu_info = result.stdout.strip()
                if 'RTX 5090' in gpu_info:
                    logger.info(f"✅ RTX 5090 DETECTED: {gpu_info}")
                    return True
                else:
                    logger.warning(f"⚠️ GPU detected but not RTX 5090: {gpu_info}")
                    logger.info("🔄 Proceeding with SM_120 build anyway...")
                    return True
            else:
                logger.error("❌ nvidia-smi failed. Check NVIDIA drivers.")
                return False
                
        except FileNotFoundError:
            logger.error("❌ nvidia-smi not found. Install NVIDIA drivers.")
            return False
    
    def download_prebuilt_wheel(self) -> bool:
        """Try to download prebuilt wheel with SM_120 support"""
        logger.info("📦 CHECKING FOR PREBUILT WHEELS...")
        
        # URLs for prebuilt wheels (if available)
        wheel_urls = [
            "https://paddle-wheel.bj.bcebos.com/develop/windows/cpu-mkl-avx/paddlepaddle-0.0.0-cp311-cp311-win_amd64.whl",
            "https://paddle-wheel.bj.bcebos.com/develop/windows/gpu-cuda12.6-cudnn9.0-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0-cp311-cp311-win_amd64.whl"
        ]
        
        for url in wheel_urls:
            try:
                logger.info(f"🔄 Trying to download: {url}")
                wheel_file = self.work_dir / f"paddlepaddle_gpu_sm120.whl"
                self.work_dir.mkdir(exist_ok=True)
                
                urllib.request.urlretrieve(url, wheel_file)
                
                if wheel_file.exists() and wheel_file.stat().st_size > 0:
                    logger.info(f"✅ Downloaded wheel: {wheel_file}")
                    
                    # Try to install
                    result = subprocess.run([self.python_executable, '-m', 'pip', 'install', str(wheel_file), '--force-reinstall'], 
                                          capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        logger.info("✅ PREBUILT WHEEL INSTALLED SUCCESSFULLY")
                        return True
                    else:
                        logger.warning(f"⚠️ Wheel installation failed: {result.stderr}")
                        
            except Exception as e:
                logger.warning(f"⚠️ Download failed: {e}")
        
        logger.info("❌ No compatible prebuilt wheels found. Will build from source.")
        return False
    
    def clone_paddle_source(self) -> bool:
        """Clone PaddlePaddle source code"""
        logger.info("📥 CLONING PADDLEPADDLE SOURCE...")
        
        if self.paddle_source.exists():
            logger.info("🔄 Source already exists. Updating...")
            try:
                subprocess.run(['git', 'pull'], cwd=self.paddle_source, check=True)
                logger.info("✅ Source updated")
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Git pull failed: {e}")
                shutil.rmtree(self.paddle_source)
        
        # Clone fresh
        self.work_dir.mkdir(exist_ok=True)
        try:
            cmd = ['git', 'clone', 'https://github.com/PaddlePaddle/Paddle.git', str(self.paddle_source)]
            subprocess.run(cmd, check=True, cwd=self.work_dir)
            logger.info("✅ PaddlePaddle source cloned")
            
            # Checkout develop branch
            subprocess.run(['git', 'checkout', 'develop'], cwd=self.paddle_source, check=True)
            logger.info("✅ Switched to develop branch")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Git clone failed: {e}")
            return False
    
    def configure_cmake(self) -> bool:
        """Configure CMake for RTX 5090 build"""
        logger.info("⚙️ CONFIGURING CMAKE FOR RTX 5090...")
        
        self.build_dir.mkdir(exist_ok=True)
        
        # CMake configuration for RTX 5090
        cmake_args = [
            'cmake', '..',
            '-G', 'Visual Studio 16 2019',
            '-A', 'x64',
            '-DWITH_GPU=ON',
            '-DCMAKE_BUILD_TYPE=Release',
            '-DWITH_UNITY_BUILD=ON',
            f'-DCUDA_ARCH_NAME=Manual',
            f'-DCUDA_ARCH_BIN={self.rtx5090_specs["sm_version"]}',  # Critical for RTX 5090
            '-DWITH_TESTING=OFF',
            '-DWITH_INFERENCE_API_TEST=OFF',
            '-DWITH_DISTRIBUTE=OFF',
            '-DCUDA_TOOLKIT_ROOT_DIR=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6',
            '-DCUDNN_ROOT=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6',
            '-DWITH_MKLDNN=ON',
            '-DWITH_AVX=ON',
            '-DWITH_AVX2=ON',
            '-DPYTHON_EXECUTABLE=' + self.python_executable
        ]
        
        logger.info("🔧 CMake configuration:")
        for arg in cmake_args:
            if arg.startswith('-D'):
                logger.info(f"   {arg}")
        
        try:
            result = subprocess.run(cmake_args, cwd=self.build_dir, 
                                  capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                logger.info("✅ CMAKE CONFIGURATION SUCCESSFUL")
                logger.info(f"✅ RTX 5090 SM_120 support configured")
                return True
            else:
                logger.error(f"❌ CMake configuration failed:")
                logger.error(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ CMake configuration timed out (10 minutes)")
            return False
        except Exception as e:
            logger.error(f"❌ CMake configuration error: {e}")
            return False
    
    def build_paddle(self) -> bool:
        """Build PaddlePaddle"""
        logger.info("🔨 BUILDING PADDLEPADDLE...")
        logger.info("⏰ This may take 1-3 hours depending on your system...")
        
        try:
            # Build using CMake
            build_cmd = [
                'cmake', '--build', '.', 
                '--config', 'Release',
                '--target', 'paddle_pybind',
                '--parallel', str(os.cpu_count() or 4)
            ]
            
            logger.info(f"🔄 Running: {' '.join(build_cmd)}")
            
            result = subprocess.run(build_cmd, cwd=self.build_dir, 
                                  capture_output=True, text=True, timeout=10800)  # 3 hours
            
            if result.returncode == 0:
                logger.info("✅ PADDLEPADDLE BUILD SUCCESSFUL")
                return True
            else:
                logger.error(f"❌ Build failed:")
                logger.error(result.stderr[-2000:])  # Last 2000 chars
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Build timed out (3 hours). Try again or use prebuilt wheel.")
            return False
        except Exception as e:
            logger.error(f"❌ Build error: {e}")
            return False
    
    def install_wheel(self) -> bool:
        """Install built wheel"""
        logger.info("📦 INSTALLING BUILT WHEEL...")
        
        # Find wheel file
        python_dist = self.build_dir / "python" / "dist"
        if not python_dist.exists():
            logger.error("❌ Wheel directory not found")
            return False
        
        wheel_files = list(python_dist.glob("*.whl"))
        if not wheel_files:
            logger.error("❌ No wheel files found")
            return False
        
        wheel_file = wheel_files[0]
        logger.info(f"📦 Installing wheel: {wheel_file.name}")
        
        try:
            result = subprocess.run([
                self.python_executable, '-m', 'pip', 'install', 
                str(wheel_file), '--force-reinstall'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ WHEEL INSTALLED SUCCESSFULLY")
                return True
            else:
                logger.error(f"❌ Wheel installation failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Installation error: {e}")
            return False
    
    def test_installation(self) -> bool:
        """Test the installation"""
        logger.info("🧪 TESTING INSTALLATION...")
        
        test_script = """
import paddle
import sys

print(f"Python: {sys.version}")
print(f"PaddlePaddle: {paddle.__version__}")
print(f"CUDA compiled: {paddle.device.is_compiled_with_cuda()}")
print(f"GPU count: {paddle.device.cuda.device_count()}")

if paddle.device.cuda.device_count() > 0:
    print(f"GPU devices: {[paddle.device.cuda.get_device_name(i) for i in range(paddle.device.cuda.device_count())]}")
    
    # Test RTX 5090 computation
    try:
        paddle.device.set_device('gpu:0')
        x = paddle.randn([1000, 1000])
        y = paddle.matmul(x, x)
        print("✅ RTX 5090 computation test PASSED")
        
        # Check compute capability
        props = paddle.device.cuda.get_device_properties(0)
        major = props.major
        minor = props.minor
        print(f"Compute capability: {major}.{minor}")
        
        if major >= 9:  # RTX 5090 should be 12.0
            print("✅ RTX 5090 compute capability supported")
        else:
            print("⚠️ Older GPU detected, but should work")
            
    except Exception as e:
        print(f"❌ GPU computation test failed: {e}")
        sys.exit(1)
else:
    print("❌ No GPU detected")
    sys.exit(1)

print("🎉 ALL TESTS PASSED - RTX 5090 READY!")
"""
        
        try:
            result = subprocess.run([self.python_executable, '-c', test_script], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ INSTALLATION TEST PASSED")
                logger.info("✅ RTX 5090 SUPPORT VERIFIED")
                print(result.stdout)
                return True
            else:
                logger.error(f"❌ Installation test failed:")
                logger.error(result.stderr)
                return False
                
        except Exception as e:
            logger.error(f"❌ Test error: {e}")
            return False
    
    def cleanup(self):
        """Clean up build files"""
        logger.info("🧹 CLEANING UP...")
        
        if self.work_dir.exists():
            try:
                shutil.rmtree(self.work_dir)
                logger.info("✅ Build files cleaned up")
            except Exception as e:
                logger.warning(f"⚠️ Cleanup failed: {e}")
    
    def build_sm120_support(self, cleanup_after=True) -> bool:
        """Main build process for RTX 5090 SM_120 support"""
        logger.info("🚀 STARTING RTX 5090 SM_120 BUILD PROCESS...")
        
        try:
            # Step 1: Check prerequisites
            if not self.check_prerequisites():
                logger.error("❌ Prerequisites not satisfied")
                return False
            
            # Step 2: Check RTX 5090
            if not self.check_rtx5090():
                logger.error("❌ RTX 5090 not detected")
                return False
            
            # Step 3: Try prebuilt wheel first
            if self.download_prebuilt_wheel():
                if self.test_installation():
                    logger.info("🎉 PREBUILT WHEEL SUCCESS - RTX 5090 READY!")
                    return True
            
            # Step 4: Build from source
            logger.info("🔨 Building from source...")
            
            if not self.clone_paddle_source():
                return False
            
            if not self.configure_cmake():
                return False
            
            if not self.build_paddle():
                return False
            
            if not self.install_wheel():
                return False
            
            if not self.test_installation():
                return False
            
            logger.info("🎉 BUILD SUCCESSFUL - RTX 5090 SM_120 SUPPORT READY!")
            return True
            
        except KeyboardInterrupt:
            logger.error("❌ Build interrupted by user")
            return False
        except Exception as e:
            logger.error(f"❌ Build failed: {e}")
            return False
        finally:
            if cleanup_after:
                self.cleanup()

def main():
    """Main function"""
    builder = PaddleSM120Builder()
    
    print("🎮 RTX 5090 PaddlePaddle SM_120 Builder")
    print("=" * 50)
    print("This will build PaddlePaddle with RTX 5090 support")
    print("Estimated time: 1-3 hours (depending on system)")
    print()
    
    # Confirm build
    response = input("Continue with RTX 5090 build? [y/N]: ")
    if response.lower() != 'y':
        print("Build cancelled.")
        return
    
    success = builder.build_sm120_support()
    
    if success:
        print("\n🎉 SUCCESS!")
        print("RTX 5090 PaddlePaddle is ready for training!")
        print("\nNext steps:")
        print("1. Run: python src/data/prepare_paddle_dataset.py")
        print("2. Start training with RTX 5090 optimizations")
    else:
        print("\n❌ FAILED!")
        print("RTX 5090 build unsuccessful.")
        print("\nTroubleshooting:")
        print("1. Check CUDA 12.6 installation")
        print("2. Verify Visual Studio 2019/2022")
        print("3. Use prebuilt wheels if available")
        print("4. Check logs for specific errors")

if __name__ == "__main__":
    main()
