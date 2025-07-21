# Thai OCR Dataset Generator

โปรเจคสำหรับสร้าง dataset ภาษาไทยสำหรับเทรน OCR ที่มีความหลากหลายและแข็งแกร่ง

## 🎯 คุณสมบัติหลัก

### ✅ ครอบคลุมตัวอักษรทุกตัว
- ตัวอักษรไทย: ก-ฮ (รวม **ษ**)
- วรรณยุกต์: ่ ้ ๊ ๋ ็ ์ ํ ั ิ ี ึ ื ุ ู
- สระ: เ แ โ ใ ไ ำ า
- ตัวเลข: 0-9
- อักษรอังกฤษ: A-Z, a-z
- สัญลักษณ์: ( ) [ ] { } / \ @ # % & * + - = < > ? ! , . ; : ' " ~ ^ _ |

### 🎨 Data Augmentation หลากหลาย
- **ขนาดฟอนต์**: 20-60px (จำลองระยะไกล-ใกล้)
- **การหมุน**: -15° ถึง +15° (ป้ายเอียง)
- **ความเบลอ**: Gaussian blur (กล้องไม่โฟกัส)
- **ความสว่าง**: 0.4-1.6x (แสงน้อย-มาก)
- **สัญญาณรบกวน**: ±40 (จุดด่าง, คุณภาพต่ำ)
- **ความคมชัด**: 0.5-2.0x (ภาพจาง-เข้ม)
- **Gradient**: เงา/แสงไม่สม่ำเสมอ
- **ตำแหน่ง**: สุ่มตำแหน่งข้อความ
- **สี**: หลายโทนสีพื้นหลังและตัวอักษร

## 📦 การติดตั้ง

```bash
pip install -r requirements.txt
```

## 🚀 การใช้งาน

### วิธีที่ 1: ใช้ไฟล์สำเร็จรูป
```bash
python create_thai_ocr_dataset.py
```

### วิธีที่ 2: กำหนดพารามิเตอร์เอง
```bash
# สร้าง 10,000 รูป จากตัวอักษรใน th_dict.txt
python thai_text_generator.py -c th_dict.txt -n 10000 -o ./my_dataset/

# สร้างจากประโยคไทย
python thai_text_generator.py -c thai_corpus.txt -n 5000 -o ./sentence_dataset/
```

## ⚙️ พารามิเตอร์

| พารามิเตอร์ | คำอธิบาย | ค่าเริ่มต้น |
|-------------|----------|-------------|
| `-c, --corpus` | ไฟล์ข้อมูลตัวอักษร/ประโยค | `thai_corpus.txt` |
| `-n, --num` | จำนวนรูปภาพที่ต้องการ | `1000` |
| `-o, --output` | โฟลเดอร์บันทึกผลลัพธ์ | `./dataset/` |

## 📁 โครงสร้าง Dataset

```
thai_ocr_dataset/
├── images/              # รูปภาพทั้งหมด
│   ├── 000000.jpg      # รูปตัวอักษร/คำ
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
└── labels.txt          # ไฟล์ label
```

### รูปแบบไฟล์ labels.txt:
```
000000.jpg	ก
000001.jpg	ษ
000002.jpg	ความสุข
000003.jpg	123
```

## 🗂️ ไฟล์ในโปรเจค

| ไฟล์ | คำอธิบาย |
|------|----------|
| `create_thai_ocr_dataset.py` | ไฟล์หลักสำหรับสร้าง dataset |
| `thai_text_generator.py` | โปรแกรมสร้างรูปภาพ (มี CLI) |
| `th_dict.txt` | ตัวอักษรไทยครบถ้วน (แนะนำ) |
| `thai_corpus.txt` | ประโยคภาษาไทย (ทางเลือก) |
| `requirements.txt` | Dependencies ที่ต้องใช้ |
| `README.md` | คู่มือนี้ |

## 🎯 การใช้งานกับ OCR

### สำหรับ PaddleOCR:
```python
# แปลงรูปแบบ
with open('thai_ocr_dataset/labels.txt', 'r', encoding='utf-8') as f:
    for line in f:
        filename, text = line.strip().split('\t')
        print(f'images/{filename}\t{text}')
```

### สำหรับ TrOCR/Transformers:
```python
from torch.utils.data import Dataset
from PIL import Image

class ThaiOCRDataset(Dataset):
    def __init__(self, labels_file, images_dir):
        with open(labels_file, 'r', encoding='utf-8') as f:
            self.data = [line.strip().split('\t') for line in f]
        self.images_dir = images_dir
    
    def __getitem__(self, idx):
        filename, text = self.data[idx]
        image = Image.open(f'{self.images_dir}/{filename}')
        return image, text
```

## 💡 เคล็ดลับการใช้งาน

1. **สำหรับป้ายทะเบียน**: ใช้ `th_dict.txt` เพื่อให้ครอบคลุมตัว "ษ"
2. **สำหรับข้อความทั่วไป**: ใช้ `thai_corpus.txt` เพื่อประโยคที่สมจริง
3. **เพิ่มจำนวนรูป**: ใช้ `-n 10000` หรือมากกว่าเพื่อความแม่นยำสูง
4. **ตรวจสอบผลลัพธ์**: ดูรูปตัวอย่างใน `images/` ก่อนเทรน

## 🔍 การตรวจสอบ Dataset

```python
# ตรวจสอบว่ามีตัว "ษ" หรือไม่
with open('thai_ocr_dataset/labels.txt', 'r', encoding='utf-8') as f:
    content = f.read()
    if 'ษ' in content:
        print(f"✅ พบตัว 'ษ' จำนวน {content.count('ษ')} ครั้ง")
    else:
        print("❌ ไม่พบตัว 'ษ'")
```

## 🚨 ข้อควรระวัง

- ตรวจสอบว่ามีฟอนต์ไทยในระบบ (Tahoma แนะนำ)
- Dataset ขนาดใหญ่อาจใช้เวลานานในการสร้าง
- ตรวจสอบพื้นที่ดิสก์ก่อนสร้าง dataset ขนาดใหญ่

---

**สร้างโดย**: Thai OCR Dataset Generator  
**เวอร์ชัน**: 1.0  
**รองรับ**: Python 3.7+