#!/usr/bin/env python3
"""
สคริปต์สำหรับสร้าง Thai OCR Dataset
รันได้ง่าย พร้อมคำสั่งที่ใช้จริง
"""

import os
import subprocess
import sys

def run_command(cmd, description):
    """รันคำสั่งและแสดงผล"""
    print(f"\n🚀 {description}")
    print(f"คำสั่ง: {cmd}")
    print("-" * 50)
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode == 0:
        print(f"✅ {description} สำเร็จ!")
    else:
        print(f"❌ {description} ล้มเหลว!")
        return False
    return True

def main():
    """สร้าง dataset ทั้งหมด"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.dirname(base_dir))  # ไปที่ root ของ project
    
    print("🎯 สร้าง Thai OCR Dataset")
    print("=" * 50)
    
    # เช็คว่ามีไฟล์ thai_text_generator.py หรือไม่
    if not os.path.exists("thai-letters/thai_text_generator.py"):
        print("❌ ไม่พบไฟล์ thai_text_generator.py")
        return
    
    # เช็คว่ามี corpus หรือไม่
    if not os.path.exists("thai-letters/thai_corpus.txt"):
        print("❌ ไม่พบไฟล์ thai_corpus.txt")
        return
    
    # นับจำนวนบรรทัดใน corpus
    with open("thai-letters/thai_corpus.txt", 'r', encoding='utf-8') as f:
        corpus_lines = sum(1 for line in f)
    
    print(f"📊 พบ corpus {corpus_lines:,} บรรทัด")
    
    # สร้าง dataset ต่างๆ
    datasets = [
        {
            "name": "Dataset ตัวอย่าง (500 ภาพ)",
            "cmd": "python thai-letters/thai_text_generator.py -c thai-letters/thai_corpus.txt -n 500 -o thai-letters/generated_text_samples",
            "output": "thai-letters/generated_text_samples"
        },
        {
            "name": f"Dataset ครบชุด ({corpus_lines:,} ภาพ)",
            "cmd": f"python thai-letters/thai_text_generator.py -c thai-letters/thai_corpus.txt -n {corpus_lines} -o thai-letters/full_corpus_dataset",
            "output": "thai-letters/full_corpus_dataset"
        }
    ]
    
    for dataset in datasets:
        # เช็คว่ามี dataset อยู่แล้วหรือไม่
        if os.path.exists(dataset["output"]):
            response = input(f"\n❓ {dataset['name']} มีอยู่แล้ว ต้องการสร้างใหม่? (y/N): ")
            if response.lower() != 'y':
                print(f"⏭️  ข้าม {dataset['name']}")
                continue
        
        # สร้าง dataset
        success = run_command(dataset["cmd"], dataset["name"])
        if not success:
            continue
        
        # เช็คผลลัพธ์
        if os.path.exists(f"{dataset['output']}/images"):
            import glob
            images = glob.glob(f"{dataset['output']}/images/*.jpg")
            print(f"📷 สร้างรูปภาพ: {len(images):,} ภาพ")
            
        if os.path.exists(f"{dataset['output']}/labels.txt"):
            with open(f"{dataset['output']}/labels.txt", 'r', encoding='utf-8') as f:
                labels = sum(1 for line in f)
            print(f"🏷️  สร้าง labels: {labels:,} บรรทัด")
    
    print("\n🎉 การสร้าง dataset เสร็จสิ้น!")
    print("📁 ตรวจสอบโฟลเดอร์:")
    print("   - thai-letters/thai_ocr_dataset/ (5,000 ภาพ - เดิม)")
    print("   - thai-letters/generated_text_samples/ (500 ภาพ)")
    print(f"   - thai-letters/full_corpus_dataset/ ({corpus_lines:,} ภาพ)")

if __name__ == "__main__":
    main()
