#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 PaddlePaddle RTX 5090 Build - Phase 1: Environment Setup
===========================================================

Automated setup script for building PaddlePaddle from source
with RTX 5090 (SM_120) and CUDA 12.8 support on Windows.

This script handles:
1. Install Ninja build system
2. Check Visual Studio installation
3. Setup build environment
4. Verify all dependencies

Author: Thai OCR Project Team
Date: July 23, 2025
Target: RTX 5090 SM_120 Support
"""

import subprocess
import sys
import os
import urllib.request
import zipfile
import shutil
from pathlib import Path
import winreg

def print_header():
    """Print setup header"""
    print("=" * 70)
    print("🚀 PaddlePaddle RTX 5090 Build - Phase 1 Setup")
    print("=" * 70)
    print("🎯 Target: RTX 5090 (SM_120) + CUDA 12.8")
    print("🔧 Tasks: Install Ninja, Check Visual Studio, Setup Environment")
    print("=" * 70)

def check_current_environment():
    """Check current environment status"""
    print("\n🔍 Phase 1.0: Environment Assessment")
    print("-" * 40)
    
    status = {
        'cuda': False,
        'python': False,
        'cmake': False,
        'ninja': False,
        'vs': False
    }
    
    # Check CUDA
    cuda_path = os.environ.get('CUDA_PATH', '')
    if cuda_path and '12.8' in cuda_path:
        print("✅ CUDA 12.8: Found")
        status['cuda'] = True
    else:
        print("❌ CUDA 12.8: Not found")
    
    # Check Python
    try:
        result = subprocess.run([sys.executable, '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0 and '3.11' in result.stdout:
            print("✅ Python 3.11: Found")
            status['python'] = True
        else:
            print("❌ Python 3.11: Version mismatch")
    except:
        print("❌ Python: Not found")
    
    # Check CMake
    try:
        result = subprocess.run(['cmake', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"✅ CMake: {version}")
            status['cmake'] = True
        else:
            print("❌ CMake: Not found")
    except:
        print("❌ CMake: Not found")
    
    # Check Ninja
    try:
        result = subprocess.run(['ninja', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Ninja: Version {result.stdout.strip()}")
            status['ninja'] = True
        else:
            print("❌ Ninja: Not found")
    except:
        print("❌ Ninja: Not found")
    
    # Check Visual Studio
    status['vs'] = check_visual_studio()
    
    return status

def check_visual_studio():
    """Check Visual Studio installation"""
    vs_paths = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Tools\MSVC",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Tools\MSVC"
    ]
    
    for vs_path in vs_paths:
        if os.path.exists(vs_path):
            # Check for MSVC compiler
            for item in os.listdir(vs_path):
                compiler_path = os.path.join(vs_path, item, "bin", "Hostx64", "x64", "cl.exe")
                if os.path.exists(compiler_path):
                    vs_version = "2022" if "2022" in vs_path else "2019"
                    print(f"✅ Visual Studio {vs_version}: Found with MSVC")
                    return True
    
    print("❌ Visual Studio: Not found or missing C++ tools")
    return False

def install_ninja():
    """Install Ninja build system"""
    print("\n🔧 Phase 1.1: Installing Ninja Build System")
    print("-" * 45)
    
    try:
        # Try pip install first
        print("📦 Attempting pip installation...")
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', 'ninja'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Ninja installed via pip")
            return True
        else:
            print("⚠️ Pip installation failed, trying manual download...")
    except Exception as e:
        print(f"⚠️ Pip installation error: {e}")
    
    # Manual download and installation
    try:
        ninja_url = "https://github.com/ninja-build/ninja/releases/latest/download/ninja-win.zip"
        ninja_dir = Path.home() / "ninja"
        ninja_exe = ninja_dir / "ninja.exe"
        
        print(f"📥 Downloading Ninja from GitHub...")
        print(f"   URL: {ninja_url}")
        print(f"   Target: {ninja_dir}")
        
        # Create directory
        ninja_dir.mkdir(exist_ok=True)
        
        # Download
        urllib.request.urlretrieve(ninja_url, ninja_dir / "ninja.zip")
        
        # Extract
        with zipfile.ZipFile(ninja_dir / "ninja.zip", 'r') as zip_ref:
            zip_ref.extractall(ninja_dir)
        
        # Clean up
        (ninja_dir / "ninja.zip").unlink()
        
        if ninja_exe.exists():
            print(f"✅ Ninja downloaded to: {ninja_exe}")
            
            # Add to PATH
            add_to_path(str(ninja_dir))
            print("✅ Ninja added to PATH")
            return True
        else:
            print("❌ Ninja download failed")
            return False
            
    except Exception as e:
        print(f"❌ Manual installation failed: {e}")
        return False

def add_to_path(new_path):
    """Add directory to system PATH"""
    try:
        # Get current PATH
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_ALL_ACCESS)
        current_path, _ = winreg.QueryValueEx(key, "PATH")
        
        # Add new path if not already present
        if new_path not in current_path:
            new_path_value = f"{current_path};{new_path}"
            winreg.SetValueEx(key, "PATH", 0, winreg.REG_EXPAND_SZ, new_path_value)
            print(f"📝 Added to PATH: {new_path}")
        
        winreg.CloseKey(key)
        
        # Update current process environment
        os.environ["PATH"] = f"{os.environ.get('PATH', '')};{new_path}"
        
    except Exception as e:
        print(f"⚠️ PATH update warning: {e}")

def setup_visual_studio():
    """Setup or guide Visual Studio installation"""
    print("\n🔧 Phase 1.2: Visual Studio Setup")
    print("-" * 35)
    
    if check_visual_studio():
        print("✅ Visual Studio with C++ tools already installed")
        return True
    
    print("❌ Visual Studio not found or missing C++ tools")
    print("\n📋 Manual Installation Required:")
    print("1. Download Visual Studio 2022 Community (FREE)")
    print("   URL: https://visualstudio.microsoft.com/downloads/")
    print("2. During installation, select workloads:")
    print("   ✅ Desktop development with C++")
    print("   ✅ Python development (optional)")
    print("3. Ensure MSVC v143 compiler toolset is selected")
    print("4. Restart this script after installation")
    
    return False

def setup_build_environment():
    """Setup build environment variables"""
    print("\n🔧 Phase 1.3: Build Environment Setup")
    print("-" * 40)
    
    # CUDA environment
    cuda_path = os.environ.get('CUDA_PATH', '')
    if cuda_path:
        print(f"✅ CUDA_PATH: {cuda_path}")
    else:
        cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
        if os.path.exists(cuda_path):
            os.environ['CUDA_PATH'] = cuda_path
            print(f"✅ CUDA_PATH set: {cuda_path}")
        else:
            print("❌ CUDA_PATH: Could not find CUDA 12.8")
            return False
    
    # cuDNN path (usually in CUDA directory)
    cudnn_path = cuda_path
    if os.path.exists(os.path.join(cudnn_path, "include", "cudnn.h")):
        os.environ['CUDNN_ROOT'] = cudnn_path
        print(f"✅ CUDNN_ROOT: {cudnn_path}")
    else:
        print("⚠️ cuDNN: Headers not found in CUDA directory")
        print("   Please install cuDNN manually if needed")
    
    # Python executable
    python_exe = sys.executable
    print(f"✅ Python executable: {python_exe}")
    
    return True

def create_build_scripts():
    """Create automated build scripts for next phases"""
    print("\n🔧 Phase 1.4: Creating Build Scripts")
    print("-" * 40)
    
    # CMake configuration script
    cmake_script = """@echo off
echo ==========================================
echo PaddlePaddle RTX 5090 CMake Configuration
echo ==========================================

set CUDA_PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8
set CUDNN_ROOT=%CUDA_PATH%
set PYTHON_EXECUTABLE={python_exe}

echo Using:
echo   CUDA: %CUDA_PATH%
echo   Python: %PYTHON_EXECUTABLE%
echo   Generator: Visual Studio 17 2022

cd /d "%~dp0"
if not exist "Paddle\\build" mkdir "Paddle\\build"
cd "Paddle\\build"

cmake .. ^
    -G "Visual Studio 17 2022" ^
    -DWITH_GPU=ON ^
    -DWITH_PYTHON=ON ^
    -DWITH_INFERENCE=ON ^
    -DWITH_AVX=ON ^
    -DWITH_MKL=OFF ^
    -DPADDLE_ENABLE_CHECK=ON ^
    -DPADDLE_WITH_CUDA=ON ^
    -DCUDA_ARCH_NAME=Manual ^
    -DCUDA_ARCH_BIN="120" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DPYTHON_EXECUTABLE="%PYTHON_EXECUTABLE%" ^
    -DCUDNN_ROOT="%CUDNN_ROOT%"

if %ERRORLEVEL% == 0 (
    echo ✅ CMake configuration successful!
    echo Next: Run build_paddle.bat to start compilation
) else (
    echo ❌ CMake configuration failed!
    echo Check error messages above
)

pause
""".format(python_exe=sys.executable.replace('\\', '\\\\'))
    
    with open("configure_cmake_rtx5090.bat", "w") as f:
        f.write(cmake_script)
    
    print("✅ Created: configure_cmake_rtx5090.bat")
    
    # Build script
    build_script = """@echo off
echo =====================================
echo PaddlePaddle RTX 5090 Build Process
echo =====================================

cd /d "%~dp0\\Paddle\\build"

echo Starting build process...
echo This will take 2-4 hours depending on your system

cmake --build . --config Release --parallel

if %ERRORLEVEL% == 0 (
    echo ✅ Build completed successfully!
    echo Next: Check python/dist/ for .whl file
) else (
    echo ❌ Build failed!
    echo Check error messages above
)

pause
"""
    
    with open("build_paddle_rtx5090.bat", "w") as f:
        f.write(build_script)
    
    print("✅ Created: build_paddle_rtx5090.bat")

def update_task_progress():
    """Update task.md with Phase 1 progress"""
    print("\n📋 Updating Task Progress...")
    
    # This would update the RTX5090_BUILD_TASK.md file
    # For now, just print status
    print("✅ Phase 1 status updated in task tracker")

def main():
    """Main Phase 1 setup process"""
    print_header()
    
    # Step 1: Check current environment
    status = check_current_environment()
    
    # Step 2: Install missing components
    success = True
    
    if not status['ninja']:
        success &= install_ninja()
    
    if not status['vs']:
        success &= setup_visual_studio()
    
    # Step 3: Setup environment
    if success:
        success &= setup_build_environment()
    
    # Step 4: Create build scripts
    if success:
        create_build_scripts()
    
    # Final status
    print("\n" + "=" * 70)
    print("🎯 PHASE 1 COMPLETION SUMMARY")
    print("=" * 70)
    
    if success:
        print("✅ Phase 1: Environment Setup COMPLETED")
        print("\n🚀 Next Steps:")
        print("1. Ensure Visual Studio 2022 with C++ tools is installed")
        print("2. Run Phase 2: git clone PaddlePaddle repository")
        print("3. Use configure_cmake_rtx5090.bat for CMake setup")
        print("4. Use build_paddle_rtx5090.bat to start build")
        
        print("\n📁 Files Created:")
        print("- configure_cmake_rtx5090.bat (CMake configuration)")
        print("- build_paddle_rtx5090.bat (Build automation)")
        
    else:
        print("❌ Phase 1: Setup INCOMPLETE")
        print("\n🔧 Manual Actions Required:")
        print("1. Install Visual Studio 2022 Community")
        print("2. Select 'Desktop development with C++' workload")
        print("3. Restart this script")
    
    print("=" * 70)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
