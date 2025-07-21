# การสร้าง Thai OCR Dataset

## คำสั่งที่ใช้สร้าง Dataset ทั้งหมด

### 1. Dataset ตัวอย่าง (500 ภาพ)
```bash
python thai-letters/thai_text_generator.py -c thai-letters/thai_corpus.txt -n 500 -o thai-letters/generated_text_samples
```

### 2. Dataset ครบชุด (9,672 ภาพ - เท่ากับจำนวนบรรทัดใน corpus)
```bash
python thai-letters/thai_text_generator.py -c thai-letters/thai_corpus.txt -n 9672 -o thai-letters/full_corpus_dataset
```

### 3. Dataset เดิมที่มีอยู่ (5,000 ภาพ)
อยู่ในโฟลเดอร์: `thai-letters/thai_ocr_dataset/`

## สรุป Dataset ทั้งหมด

| Dataset | จำนวนภาพ | โฟลเดอร์ | คำอธิบาย |
|---------|----------|----------|----------|
| เดิม | 5,000 | `thai_ocr_dataset/` | Dataset ที่มีอยู่แล้ว |
| ตัวอย่าง | 500 | `generated_text_samples/` | สำหรับทดสอบ |
| ครบชุด | 9,672 | `full_corpus_dataset/` | ใช้ corpus ทั้งหมด |
| **รวม** | **15,172** | | **Dataset สำหรับ Training** |

## การใช้งาน thai_text_generator.py

### Parameters:
- `-c, --corpus`: ไฟล์ corpus (default: thai_corpus.txt)
- `-o, --output`: โฟลเดอร์ output (default: ./dataset/)
- `-n, --num`: จำนวนภาพที่ต้องการ (default: 1000)

### ตัวอย่างการใช้งาน:
```bash
# สร้าง 1000 ภาพแบบ default
python thai-letters/thai_text_generator.py

# สร้าง custom จำนวน
python thai-letters/thai_text_generator.py -n 2000 -o my_dataset

# ใช้ corpus file อื่น
python thai-letters/thai_text_generator.py -c my_corpus.txt -n 500
```

## โครงสร้างผลลัพธ์

แต่ละ dataset จะมีโครงสร้าง:
```
dataset_folder/
├── images/
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
└── labels.txt
```

## ข้อมูลใน thai_corpus.txt
- จำนวนบรรทัด: 9,672 บรรทัด
- เนื้อหา: คำศัพท์ไทยที่หลากหลาย
- รูปแบบ: คำสั้น คำยาว และประโยค

## สถิติ Dataset

### Dataset เดิม (thai_ocr_dataset):
- ข้อมูล: อักษรไทย ตัวเลข สัญลักษณ์
- รูปแบบ: คำสั้น 1-3 อักษร

### Dataset ใหม่ (generated):
- ฟอนต์: Tahoma
- ขนาดภาพ: เหมาะสำหรับ OCR
- เนื้อหา: จาก thai_corpus.txt
- รูปแบบ: คำและประโยคที่หลากหลาย
