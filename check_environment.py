#!/usr/bin/env python3
"""
Quick Environment Check for PaddlePaddle RTX 5090 Build
"""

import subprocess
import os
import sys

def check_msvc():
    """Check MSVC compiler availability"""
    print("🔍 Checking MSVC Compiler...")
    
    vs_path = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    
    if not os.path.exists(vs_path):
        print("❌ Visual Studio 2022 Community not found")
        return False
    
    try:
        # Test MSVC compiler
        cmd = f'"{vs_path}" >nul && cl 2>&1'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        
        if "Microsoft (R) C/C++ Optimizing Compiler" in result.stderr:
            # Extract version
            lines = result.stderr.split('\n')
            for line in lines:
                if "Microsoft (R) C/C++ Optimizing Compiler" in line:
                    print(f"✅ MSVC Compiler: {line.strip()}")
                    return True
        else:
            print("❌ MSVC Compiler: Not responding correctly")
            return False
            
    except Exception as e:
        print(f"❌ MSVC Test Error: {e}")
        return False

def check_ninja():
    """Check Ninja build system"""
    print("🔍 Checking Ninja...")
    
    try:
        import ninja
        print("✅ Ninja: Available via Python module")
        return True
    except ImportError:
        print("❌ Ninja: Not available")
        return False

def check_cmake():
    """Check CMake"""
    print("🔍 Checking CMake...")
    
    try:
        result = subprocess.run(['cmake', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"✅ CMake: {version}")
            return True
        else:
            print("❌ CMake: Not found")
            return False
    except:
        print("❌ CMake: Not found")
        return False

def check_cuda():
    """Check CUDA"""
    print("🔍 Checking CUDA...")
    
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'release' in line and 'V12.8' in line:
                    print("✅ CUDA 12.8: Available")
                    return True
            print("⚠️ CUDA: Found but not version 12.8")
            return False
        else:
            print("❌ CUDA: Not found")
            return False
    except:
        print("❌ CUDA: Not found")
        return False

def main():
    print("=" * 50)
    print("🚀 PaddlePaddle RTX 5090 Environment Check")
    print("=" * 50)
    
    checks = {
        'MSVC': check_msvc(),
        'Ninja': check_ninja(),
        'CMake': check_cmake(),
        'CUDA': check_cuda()
    }
    
    print("\n" + "=" * 50)
    print("📋 ENVIRONMENT SUMMARY")
    print("=" * 50)
    
    all_pass = True
    for tool, status in checks.items():
        status_symbol = "✅" if status else "❌"
        print(f"{status_symbol} {tool}: {'READY' if status else 'MISSING'}")
        if not status:
            all_pass = False
    
    print("\n" + "=" * 50)
    if all_pass:
        print("🎉 ALL REQUIREMENTS MET!")
        print("✅ Ready for Phase 2: Clone PaddlePaddle Source")
        print("✅ Ready for Phase 3: CMake Configuration")
    else:
        print("⚠️ Some requirements missing")
        print("❌ Complete Phase 1 setup first")
    
    print("=" * 50)
    
    return all_pass

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
