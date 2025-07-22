#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup Dataset for PaddleOCR Training
Quick script to organize dataset for training
"""

import os
import shutil
import glob
from pathlib import Path

def setup_paddleocr_dataset():
    """Setup dataset structure for PaddleOCR training"""
    
    # Paths
    source_dir = "/workspace/thai_dataset_30samples/images"
    base_output = "/workspace/paddle_dataset_30k"
    train_dir = f"{base_output}/recognition/train_images"
    val_dir = f"{base_output}/recognition/val_images"
    
    print("=== Setting up PaddleOCR Dataset Structure ===")
    print(f"Source: {source_dir}")
    print(f"Output: {base_output}")
    
    # Create directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Get all image files
    image_files = sorted(glob.glob(f"{source_dir}/*.jpg"))
    total_images = len(image_files)
    train_count = int(total_images * 0.8)  # 80% for training
    
    print(f"Total images: {total_images}")
    print(f"Training images: {train_count}")
    print(f"Validation images: {total_images - train_count}")
    
    # Copy training images
    print("Copying training images...")
    for i, img_path in enumerate(image_files[:train_count]):
        img_name = os.path.basename(img_path)
        shutil.copy2(img_path, f"{train_dir}/{img_name}")
        if (i + 1) % 500 == 0:
            print(f"  Copied {i + 1}/{train_count} training images")
    
    # Copy validation images
    print("Copying validation images...")
    for i, img_path in enumerate(image_files[train_count:]):
        img_name = os.path.basename(img_path)
        shutil.copy2(img_path, f"{val_dir}/{img_name}")
        if (i + 1) % 100 == 0:
            print(f"  Copied {i + 1}/{total_images - train_count} validation images")
    
    # Create labels files
    print("Creating label files...")
    
    # Read original labels
    labels_file = "/workspace/thai_dataset_30samples/labels.txt"
    with open(labels_file, 'r', encoding='utf-8') as f:
        all_labels = f.readlines()
    
    # Create train labels
    train_labels_file = f"{base_output}/recognition/train_list.txt"
    with open(train_labels_file, 'w', encoding='utf-8') as f:
        for img_path in image_files[:train_count]:
            img_name = os.path.basename(img_path)
            # Find corresponding label
            for label_line in all_labels:
                if img_name in label_line:
                    # Convert path
                    new_line = label_line.replace("images/", "train_images/")
                    f.write(new_line)
                    break
    
    # Create val labels
    val_labels_file = f"{base_output}/recognition/val_list.txt"
    with open(val_labels_file, 'w', encoding='utf-8') as f:
        for img_path in image_files[train_count:]:
            img_name = os.path.basename(img_path)
            # Find corresponding label
            for label_line in all_labels:
                if img_name in label_line:
                    # Convert path
                    new_line = label_line.replace("images/", "val_images/")
                    f.write(new_line)
                    break
    
    # Copy dictionary
    dict_source = "/workspace/thai-letters/th_dict_utf8.txt"
    dict_dest = f"{base_output}/thai_dict.txt"
    if os.path.exists(dict_source):
        shutil.copy2(dict_source, dict_dest)
        print(f"Dictionary copied: {dict_dest}")
    
    # Final verification
    train_images = len(glob.glob(f"{train_dir}/*.jpg"))
    val_images = len(glob.glob(f"{val_dir}/*.jpg"))
    
    print("\n=== Dataset Setup Complete ===")
    print(f"✅ Train images: {train_images}")
    print(f"✅ Val images: {val_images}")
    print(f"✅ Train labels: {train_labels_file}")
    print(f"✅ Val labels: {val_labels_file}")
    print(f"✅ Dictionary: {dict_dest}")
    print(f"📁 Ready for training: {base_output}")
    
    return True

if __name__ == "__main__":
    setup_paddleocr_dataset()
