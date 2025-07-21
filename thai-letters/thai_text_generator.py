import cv2
import numpy as np
import random
import os
from PIL import Image, ImageDraw, ImageFont
import argparse

class ThaiTextGenerator:
    def __init__(self, corpus_path, save_path):
        self.save_path = save_path
        self.load_corpus(corpus_path)
        self.load_thai_font()
        
    def load_corpus(self, corpus_path):
        with open(corpus_path, 'r', encoding='utf-8') as f:
            self.words = [line.strip() for line in f.readlines() if line.strip()]
    
    def load_thai_font(self):
        # ลองหาฟอนต์ไทยในระบบ
        thai_fonts = [
            "C:/Windows/Fonts/tahoma.ttf",
            "C:/Windows/Fonts/tahomabd.ttf", 
            "C:/Windows/Fonts/cordia.ttf",
            "C:/Windows/Fonts/cordiau.ttf",
            "C:/Windows/Fonts/angsana.ttf",
            "C:/Windows/Fonts/browau.ttf"
        ]
        
        self.font = None
        for font_path in thai_fonts:
            if os.path.exists(font_path):
                try:
                    self.font = ImageFont.truetype(font_path, 32)
                    print(f"ใช้ฟอนต์: {font_path}")
                    break
                except:
                    continue
        
        if self.font is None:
            print("ไม่พบฟอนต์ไทย ใช้ฟอนต์ default")
            self.font = ImageFont.load_default()
    
    def generate_line_image(self, text):
        # สุ่มขนาดฟอนต์
        font_size = random.randint(20, 60)
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/tahoma.ttf", font_size)
        except:
            font = self.font
        
        # คำนวณขนาดรูป
        temp_img = Image.new('RGB', (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)
        bbox = temp_draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # สร้างรูปขนาดหลากหลาย
        padding_x = random.randint(20, 80)
        padding_y = random.randint(10, 40)
        img_width = text_width + padding_x
        img_height = max(text_height + padding_y, 40)
        
        # สุ่มสีพื้นหลัง
        bg_colors = ['white', (240,240,240), (250,250,250), (230,230,230)]
        bg_color = random.choice(bg_colors)
        
        # สุ่มสีตัวอักษร
        text_colors = ['black', (50,50,50), (30,30,30), (70,70,70)]
        text_color = random.choice(text_colors)
        
        img = Image.new('RGB', (img_width, img_height), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # วาดข้อความ (สุ่มตำแหน่ง)
        x = random.randint(5, max(5, img_width - text_width - 5))
        y = random.randint(5, max(5, img_height - text_height - 5))
        draw.text((x, y), text, font=font, fill=text_color)
        
        # แปลงเป็น OpenCV format
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # เพิ่ม augmentation (ทุกรูป)
        cv_img = self.add_augmentation(cv_img)
            
        return cv_img
    
    def add_augmentation(self, img):
        # 1. ปรับความสว่าง
        if random.random() > 0.3:
            brightness = random.uniform(0.4, 1.6)
            img = cv2.convertScaleAbs(img, alpha=brightness, beta=random.randint(-30, 30))
        
        # 2. เบลอ
        if random.random() > 0.6:
            blur_size = random.choice([1, 3, 5])
            img = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
        
        # 3. หมุน/เอียง
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            h, w = img.shape[:2]
            center = (w//2, h//2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, matrix, (w, h), borderValue=(255,255,255))
        
        # 4. เพิ่ม noise
        if random.random() > 0.4:
            noise = np.random.randint(-40, 40, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # 5. ปรับ contrast
        if random.random() > 0.5:
            contrast = random.uniform(0.5, 2.0)
            img = cv2.convertScaleAbs(img, alpha=contrast, beta=0)
        
        # 6. เพิ่มเงา/gradient
        if random.random() > 0.7:
            h, w = img.shape[:2]
            gradient = np.linspace(0.7, 1.3, w).reshape(1, -1, 1)
            img = (img * gradient).astype(np.uint8)
        
        return img
    
    def clean_filename(self, text):
        # แทนที่อักขระที่ไม่สามารถใช้ในชื่อไฟล์ได้
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        for char in invalid_chars:
            text = text.replace(char, '_')
        return text
    
    def generate_dataset(self, num_images, variations_per_word=5):
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'images'), exist_ok=True)
        
        # สร้างไฟล์ labels
        labels_file = os.path.join(self.save_path, 'labels.txt')
        
        with open(labels_file, 'w', encoding='utf-8') as f:
            for i in range(num_images):
                # เลือกคำสุ่ม
                text = random.choice(self.words)
                
                # สร้างรูปภาพหลายแบบของคำเดียวกัน
                img = self.generate_line_image(text)
                
                # ตั้งชื่อไฟล์
                filename = f"{i:06d}.jpg"
                filepath = os.path.join(self.save_path, 'images', filename)
                
                # บันทึกรูป
                cv2.imwrite(filepath, img)
                
                # บันทึก label
                f.write(f"{filename}\t{text}\n")
                
                if (i + 1) % 100 == 0:
                    print(f"สร้างแล้ว {i + 1}/{num_images} รูป")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--corpus", default="thai_corpus.txt", help="Path to corpus file")
    parser.add_argument("-o", "--output", default="./dataset/", help="Output directory")
    parser.add_argument("-n", "--num", type=int, default=1000, help="Number of images to generate")
    
    args = parser.parse_args()
    
    generator = ThaiTextGenerator(args.corpus, args.output)
    generator.generate_dataset(args.num)
    print("สร้าง dataset เสร็จแล้ว!")