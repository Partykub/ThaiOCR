#!/usr/bin/env python3
"""
NGC Environment Verification Script
Verifies that the NGC Thai OCR environment works correctly after cloning
"""

import subprocess
import sys
import json
from datetime import datetime

class NGCVerification:
    def __init__(self):
        self.container_name = "thai-ocr-training-ngc"
        self.results = {}
        
    def run_test(self, test_name, command, expected_in_output=None):
        """Run a test command and check results"""
        print(f"\n🧪 Testing: {test_name}")
        print(f"   Command: {command}")
        
        try:
            # For Docker commands, wrap in docker exec
            if not command.startswith("docker"):
                docker_cmd = f"docker exec {self.container_name} bash -c \"{command}\""
            else:
                docker_cmd = command
                
            result = subprocess.run(docker_cmd, shell=True, capture_output=True, text=True, timeout=60)
            
            success = result.returncode == 0
            if expected_in_output and success:
                success = expected_in_output.lower() in result.stdout.lower()
            
            if success:
                print(f"   ✅ PASS: {result.stdout.strip()}")
                self.results[test_name] = {"status": "PASS", "output": result.stdout.strip()}
            else:
                print(f"   ❌ FAIL: {result.stderr.strip()}")
                self.results[test_name] = {"status": "FAIL", "error": result.stderr.strip()}
                
        except subprocess.TimeoutExpired:
            print(f"   ⏰ TIMEOUT")
            self.results[test_name] = {"status": "TIMEOUT", "error": "Command timed out"}
        except Exception as e:
            print(f"   💥 ERROR: {e}")
            self.results[test_name] = {"status": "ERROR", "error": str(e)}
    
    def verify_environment(self):
        """Run comprehensive environment verification"""
        print("🔍 NGC Thai OCR Environment Verification")
        print("=" * 50)
        
        # Infrastructure tests
        self.run_test("Docker Available", "docker --version", "docker version")
        self.run_test("Container Running", f"docker ps --filter name={self.container_name} --format '{{{{.Status}}}}'", "up")
        
        # GPU tests
        self.run_test("NVIDIA SMI", "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv", "rtx")
        self.run_test("Container GPU Access", "nvidia-smi", "rtx")
        
        # Python environment tests
        self.run_test("Python Version", "python --version", "python 3")
        self.run_test("Pip Working", "pip list | head -5", "package")
        
        # PaddlePaddle tests
        self.run_test("PaddlePaddle Import", "python -c 'import paddle; print(f\"PaddlePaddle {paddle.__version__}\")'", "paddlepaddle")
        self.run_test("PaddlePaddle CUDA", "python -c 'import paddle; print(f\"CUDA: {paddle.device.is_compiled_with_cuda()}\")'", "true")
        self.run_test("GPU Device Count", "python -c 'import paddle; print(f\"GPUs: {paddle.device.cuda.device_count()}\")'", "gpus")
        
        # PaddleOCR tests
        self.run_test("PaddleOCR Import", "python -c 'import paddleocr; print(f\"PaddleOCR {paddleocr.__version__}\")'", "paddleocr")
        self.run_test("PaddleOCR Initialization", "python -c 'from paddleocr import PaddleOCR; ocr = PaddleOCR(use_gpu=True, show_log=False); print(\"PaddleOCR Ready\")'", "ready")
        
        # Dependencies tests
        self.run_test("OpenCV", "python -c 'import cv2; print(f\"OpenCV {cv2.__version__}\")'", "opencv")
        self.run_test("NumPy Compatibility", "python -c 'import numpy as np; print(f\"NumPy {np.__version__}\")'", "numpy")
        self.run_test("TensorFlow", "python -c 'import tensorflow as tf; print(f\"TensorFlow {tf.__version__}\")'", "tensorflow")
        
        # Environment variables
        self.run_test("CUDA Environment", "echo $CUDA_VISIBLE_DEVICES", "")
        self.run_test("Python Path", "echo $PYTHONPATH", "")
        
    def generate_report(self):
        """Generate verification report"""
        print("\n📊 Verification Report")
        print("=" * 50)
        
        passed = sum(1 for test in self.results.values() if test["status"] == "PASS")
        total = len(self.results)
        
        print(f"Tests Passed: {passed}/{total}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        print("\n📋 Test Details:")
        for test_name, result in self.results.items():
            status = result["status"]
            if status == "PASS":
                print(f"   ✅ {test_name}")
            elif status == "FAIL":
                print(f"   ❌ {test_name}: {result.get('error', 'Unknown error')}")
            elif status == "TIMEOUT":
                print(f"   ⏰ {test_name}: Timeout")
            else:
                print(f"   💥 {test_name}: {result.get('error', 'Unknown error')}")
        
        # Save detailed report
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "container_name": self.container_name,
            "summary": {
                "passed": passed,
                "total": total,
                "success_rate": (passed/total)*100
            },
            "tests": self.results
        }
        
        with open("ngc_verification_report.json", "w") as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Detailed report saved: ngc_verification_report.json")
        
        if passed == total:
            print("\n🎉 All tests passed! NGC environment is ready for Thai OCR")
            return True
        else:
            print(f"\n⚠️  {total - passed} tests failed. Please check the environment")
            return False

def main():
    """Main verification function"""
    verifier = NGCVerification()
    
    try:
        verifier.verify_environment()
        success = verifier.generate_report()
        
        if success:
            print("\n✅ Verification completed successfully!")
            print("🚀 Your NGC Thai OCR environment is ready to use!")
            sys.exit(0)
        else:
            print("\n❌ Verification failed!")
            print("🔧 Please check the failed tests and fix any issues")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Verification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Verification error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
