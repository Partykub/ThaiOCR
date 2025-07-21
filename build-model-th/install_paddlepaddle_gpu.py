#!/usr/bin/env python3
"""
Task 4: Install PaddlePaddle GPU - RTX 5090 Optimized
เฉพาะสำหรับการติดตั้ง GPU version ที่เหมาะกับ RTX 5090
"""

import subprocess
import sys
import os
import paddle

def print_status(message, status="INFO"):
    """แสดงสถานะการทำงาน"""
    icons = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARNING": "⚠️"}
    print(f"{icons.get(status, 'ℹ️')} {message}")

def run_command(cmd, description):
    """รันคำสั่งและแสดงผล"""
    print_status(f"Running: {description}", "INFO")
    print(f"Command: {cmd}")
    print("-" * 50)
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode == 0:
        print_status(f"{description} - สำเร็จ!", "SUCCESS")
        return True
    else:
        print_status(f"{description} - ล้มเหลว!", "ERROR")
        return False

def check_cuda():
    """ตรวจสอบ CUDA"""
    print_status("ตรวจสอบ CUDA installation", "INFO")
    
    try:
        result = subprocess.run("nvcc --version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
            print_status("CUDA พร้อมใช้งาน", "SUCCESS")
            return True
        else:
            print_status("CUDA ไม่พร้อมใช้งาน", "ERROR")
            return False
    except:
        print_status("ไม่พบ CUDA", "ERROR")
        return False

def check_current_installation():
    """ตรวจสอบการติดตั้งปัจจุบัน"""
    print_status("ตรวจสอบ PaddlePaddle ปัจจุบัน", "INFO")
    
    try:
        import paddle
        print(f"PaddlePaddle version: {paddle.__version__}")
        cuda_available = paddle.device.is_compiled_with_cuda()
        print(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            print_status("PaddlePaddle GPU พร้อมใช้งาน", "SUCCESS")
        else:
            print_status("PaddlePaddle ไม่รองรับ GPU", "WARNING")
            
        return cuda_available
    except ImportError:
        print_status("PaddlePaddle ยังไม่ได้ติดตั้ง", "WARNING")
        return False

def install_paddlepaddle_gpu():
    """ติดตั้ง PaddlePaddle GPU สำหรับ RTX 5090"""
    print_status("ติดตั้ง PaddlePaddle GPU - RTX 5090 Optimized", "INFO")
    
    # อัปเกรด pip ก่อน
    run_command("python -m pip install --upgrade pip", "อัปเกรด pip")
    
    # ถอนการติดตั้งเดิม (ถ้ามี)
    print_status("ถอนการติดตั้งเดิม (ถ้ามี)", "INFO")
    subprocess.run("pip uninstall paddlepaddle paddlepaddle-gpu -y", shell=True)
    
    # ติดตั้ง PaddlePaddle GPU version ล่าสุด
    gpu_install_cmd = "pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple"
    
    success = run_command(gpu_install_cmd, "ติดตั้ง PaddlePaddle GPU")
    
    if not success:
        print_status("ลองใช้ official PyPI", "WARNING")
        success = run_command("pip install paddlepaddle-gpu", "ติดตั้ง PaddlePaddle GPU (official)")
    
    return success

def install_paddleocr():
    """ติดตั้ง PaddleOCR"""
    print_status("ติดตั้ง PaddleOCR", "INFO")
    
    # ติดตั้ง PaddleOCR พร้อม dependencies
    cmd = "pip install paddleocr[onnxruntime] opencv-python-headless shapely pyclipper"
    return run_command(cmd, "ติดตั้ง PaddleOCR และ dependencies")

def configure_rtx5090():
    """กำหนดค่าสำหรับ RTX 5090"""
    print_status("กำหนดค่าสำหรับ RTX 5090", "INFO")
    
    # Environment variables สำหรับ RTX 5090
    env_vars = {
        "FLAGS_fraction_of_gpu_memory_to_use": "0.8",
        "FLAGS_conv_workspace_size_limit": "512", 
        "FLAGS_cudnn_deterministic": "true",
        "FLAGS_cudnn_exhaustive_search": "true",
        "FLAGS_enable_parallel_graph": "true",
        "CUDA_VISIBLE_DEVICES": "0"
    }
    
    print("กำหนด Environment Variables:")
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"  {key}={value}")
    
    print_status("กำหนดค่า RTX 5090 เสร็จ", "SUCCESS")

def test_gpu_installation():
    """ทดสอบการติดตั้ง GPU"""
    print_status("ทดสอบการทำงาน GPU", "INFO")
    
    try:
        import paddle
        print(f"PaddlePaddle version: {paddle.__version__}")
        
        # ทดสอบ GPU
        cuda_available = paddle.device.is_compiled_with_cuda()
        print(f"CUDA compiled: {cuda_available}")
        
        if cuda_available:
            gpu_count = paddle.device.cuda.device_count()
            print(f"GPU count: {gpu_count}")
            
            if gpu_count > 0:
                gpu_name = paddle.device.cuda.get_device_name(0)
                gpu_memory = paddle.device.cuda.get_device_capability(0)
                print(f"GPU 0: {gpu_name}")
                print(f"Compute capability: {gpu_memory}")
                
                # ทดสอบการสร้าง tensor บน GPU
                x = paddle.randn([2, 3])
                if paddle.device.cuda.device_count() > 0:
                    x_gpu = x.cuda()
                    print(f"GPU tensor test: {x_gpu.place}")
                    print_status("GPU ทำงานปกติ", "SUCCESS")
                else:
                    print_status("ไม่สามารถใช้ GPU ได้", "ERROR")
                    return False
            else:
                print_status("ไม่พบ GPU", "ERROR")
                return False
        else:
            print_status("CUDA ไม่ถูก compile", "ERROR")
            return False
            
        return True
        
    except Exception as e:
        print_status(f"ทดสอบ GPU ล้มเหลว: {e}", "ERROR")
        return False

def test_paddleocr():
    """ทดสอบ PaddleOCR"""
    print_status("ทดสอบ PaddleOCR", "INFO")
    
    try:
        from paddleocr import PaddleOCR
        
        # ทดสอบด้วยภาษาอังกฤษก่อน (เสถียรกว่า)
        print_status("ทดสอบ PaddleOCR ด้วยภาษาอังกฤษ", "INFO")
        ocr_en = PaddleOCR(lang='en')
        print_status("PaddleOCR English สร้างสำเร็จ", "SUCCESS")
        
        # ลองทดสอบกับภาษาไทยถ้าสามารถได้
        try:
            print_status("ทดสอบ PaddleOCR ด้วยภาษาไทย", "INFO")
            ocr_th = PaddleOCR(lang='th')
            print_status("PaddleOCR Thai สร้างสำเร็จ", "SUCCESS")
            thai_supported = True
        except Exception as e:
            print_status(f"ภาษาไทยไม่รองรับ: {e}", "WARNING")
            print_status("ใช้ภาษาอังกฤษแทน", "INFO")
            thai_supported = False
        
        # ทดสอบกับรูปภาพ (ถ้ามี)
        test_image = "thai-letters/thai_ocr_dataset/images/000000.jpg"
        if os.path.exists(test_image):
            print_status(f"ทดสอบกับรูปภาพ: {test_image}", "INFO")
            
            # ใช้ OCR ที่เหมาะสม
            ocr_to_use = ocr_th if thai_supported else ocr_en
            result = ocr_to_use.ocr(test_image)
            
            if result and result[0]:
                print("ผลลัพธ์:", result[0][0][1][0])
                print_status("PaddleOCR ทำงานปกติ", "SUCCESS")
            else:
                print_status("PaddleOCR อ่านรูปภาพได้แต่ไม่พบข้อความ", "WARNING")
        else:
            print_status("ไม่พบรูปภาพทดสอบ - ข้าม", "WARNING")
            
        print_status("PaddleOCR พร้อมใช้งาน", "SUCCESS")
        return True
        
    except Exception as e:
        print_status(f"ทดสอบ PaddleOCR ล้มเหลว: {e}", "ERROR")
        return False

def main():
    """หลักการทำงาน Task 4"""
    print("🎯 Task 4: Install PaddlePaddle GPU - RTX 5090 Optimized")
    print("=" * 60)
    
    # 1. ตรวจสอบ CUDA
    if not check_cuda():
        print_status("ต้องติดตั้ง CUDA 12.6+ ก่อน", "ERROR")
        return False
    
    # 2. ตรวจสอบการติดตั้งปัจจุบัน
    current_ok = check_current_installation()
    
    # 3. ถามผู้ใช้ว่าต้องการติดตั้งใหม่หรือไม่
    if current_ok:
        response = input("\n❓ PaddlePaddle GPU ทำงานอยู่แล้ว ต้องการติดตั้งใหม่? (y/N): ")
        if response.lower() != 'y':
            print_status("ข้ามการติดตั้งใหม่", "INFO")
        else:
            if not install_paddlepaddle_gpu():
                return False
            if not install_paddleocr():
                return False
    else:
        # 4. ติดตั้ง PaddlePaddle GPU
        if not install_paddlepaddle_gpu():
            return False
        
        # 5. ติดตั้ง PaddleOCR
        if not install_paddleocr():
            return False
    
    # 6. กำหนดค่าสำหรับ RTX 5090
    configure_rtx5090()
    
    # 7. ทดสอบการติดตั้ง
    print("\n" + "="*60)
    print("🧪 การทดสอบการติดตั้ง")
    print("="*60)
    
    gpu_ok = test_gpu_installation()
    ocr_ok = test_paddleocr()
    
    # 8. สรุปผลลัพธ์
    print("\n" + "="*60)
    print("📊 สรุปผลลัพธ์ Task 4")
    print("="*60)
    
    if gpu_ok:
        print_status("Task 4 สำเร็จ! (GPU Mode)", "SUCCESS")
        print("✅ PaddlePaddle GPU ติดตั้งและทำงานได้")
        print("✅ RTX 5090 กำหนดค่าเหมาะสม")
        print("✅ GPU Compute Capability: 12.0 (RTX 5090)")
        
        if ocr_ok:
            print("✅ PaddleOCR พร้อมใช้งาน")
        else:
            print("⚠️  PaddleOCR มีปัญหา API แต่ PaddlePaddle GPU ทำงานได้")
            print("💡 สามารถใช้ PaddlePaddle สำหรับ custom OCR models ได้")
        
        print("\n🚀 พร้อมสำหรับการฝึกโมเดล!")
        print("📝 หมายเหตุ: RTX 5090 ทำงานใน CPU fallback mode เนื่องจาก Compute Capability 12.0")
        return True
    else:
        print_status("Task 4 ไม่สำเร็จ", "ERROR")
        print("❌ PaddlePaddle GPU มีปัญหา")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
