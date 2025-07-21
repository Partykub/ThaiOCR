from thai_text_generator import ThaiTextGenerator

# สร้าง dataset ที่คำซ้ำได้แต่มีหลายสถานะการณ์
generator = ThaiTextGenerator("th_dict.txt", "./thai_ocr_dataset/")

print("สร้าง dataset ที่คำซ้ำได้ในหลายสถานะการณ์...")
print("- แต่ละคำจะมีหลายรูปแบบ")
print("- ขนาด, มุม, แสง, เบลอ ต่างกัน")
print("- เพิ่มความแข็งแกร่งให้โมเดล")

# สร้าง 5000 รูป (คำจะซ้ำแต่สถานะการณ์ต่างกัน)
generator.generate_dataset(5000)

print("สร้าง Thai OCR dataset เสร็จแล้ว!")
