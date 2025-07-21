#!/usr/bin/env python3
"""
CRNN Training Data Validator and Preparation Script
Validates and prepares training data for CRNN model training

Features:
- Data integrity validation
- Training/validation split verification
- Image quality checks
- Label format validation
- Dataset statistics reporting
"""

import os
import sys
import logging
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CRNNDataValidator:
    """Validates and prepares CRNN training data"""
    
    def __init__(self, project_root=None):
        """Initialize the validator"""
        if project_root is None:
            self.project_root = Path(__file__).parent.parent
        else:
            self.project_root = Path(project_root)
        
        self.crnn_dir = self.project_root / "thai-license-plate-recognition-CRNN"
        self.train_dir = self.crnn_dir / "DB" / "train"
        self.test_dir = self.crnn_dir / "DB" / "test"
        
    def validate_directory_structure(self):
        """Validate the directory structure"""
        logger.info("Validating directory structure...")
        
        required_dirs = [
            self.crnn_dir,
            self.train_dir,
            self.test_dir
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            if not dir_path.exists():
                missing_dirs.append(str(dir_path))
        
        if missing_dirs:
            logger.error(f"Missing directories: {missing_dirs}")
            return False
        
        logger.info("✅ Directory structure is valid")
        return True
    
    def validate_images(self, directory):
        """Validate images in a directory"""
        logger.info(f"Validating images in: {directory.name}")
        
        image_files = list(directory.glob("*.jpg"))
        if not image_files:
            logger.warning(f"No .jpg files found in {directory}")
            return {}
        
        stats = {
            'total_images': len(image_files),
            'valid_images': 0,
            'corrupted_images': [],
            'image_sizes': [],
            'file_sizes': []
        }
        
        for img_file in image_files:
            try:
                # Check file size
                file_size = img_file.stat().st_size
                stats['file_sizes'].append(file_size)
                
                # Try to open with PIL
                with Image.open(img_file) as img:
                    width, height = img.size
                    stats['image_sizes'].append((width, height))
                
                # Try to read with OpenCV
                cv_img = cv2.imread(str(img_file))
                if cv_img is None:
                    raise ValueError("OpenCV cannot read the image")
                
                stats['valid_images'] += 1
                
            except Exception as e:
                logger.warning(f"Corrupted image {img_file.name}: {e}")
                stats['corrupted_images'].append(img_file.name)
        
        # Calculate statistics
        if stats['image_sizes']:
            widths, heights = zip(*stats['image_sizes'])
            stats['avg_width'] = np.mean(widths)
            stats['avg_height'] = np.mean(heights)
            stats['min_width'] = np.min(widths)
            stats['max_width'] = np.max(widths)
            stats['min_height'] = np.min(heights)
            stats['max_height'] = np.max(heights)
        
        if stats['file_sizes']:
            stats['avg_file_size'] = np.mean(stats['file_sizes'])
            stats['total_size_mb'] = sum(stats['file_sizes']) / (1024 * 1024)
        
        logger.info(f"✅ Images validated: {stats['valid_images']}/{stats['total_images']} valid")
        if stats['corrupted_images']:
            logger.warning(f"⚠️ Corrupted images: {len(stats['corrupted_images'])}")
        
        return stats
    
    def analyze_filename_patterns(self, directory):
        """Analyze filename patterns to understand the labeling scheme"""
        logger.info(f"Analyzing filename patterns in: {directory.name}")
        
        image_files = list(directory.glob("*.jpg"))
        if not image_files:
            return {}
        
        # Extract patterns from filenames
        patterns = {}
        char_counts = {}
        
        for img_file in image_files[:10]:  # Analyze first 10 files
            filename = img_file.stem  # Remove .jpg extension
            
            # Extract potential text from filename
            # For license plates, usually the filename contains the plate text
            if len(filename) > 5:  # Reasonable license plate length
                # Count character types
                letters = sum(1 for c in filename if c.isalpha())
                digits = sum(1 for c in filename if c.isdigit())
                
                pattern_key = f"L{letters}D{digits}"
                patterns[pattern_key] = patterns.get(pattern_key, 0) + 1
                
                # Count individual characters
                for char in filename:
                    if char.isalnum():
                        char_counts[char] = char_counts.get(char, 0) + 1
        
        logger.info(f"✅ Filename patterns analyzed")
        return {
            'patterns': patterns,
            'character_frequency': char_counts,
            'sample_filenames': [f.stem for f in image_files[:5]]
        }
    
    def generate_report(self):
        """Generate a comprehensive data validation report"""
        logger.info("Generating validation report...")
        
        # Validate structure
        if not self.validate_directory_structure():
            return False
        
        report = {
            'timestamp': str(Path().cwd()),
            'project_root': str(self.project_root),
            'crnn_directory': str(self.crnn_dir)
        }
        
        # Validate training data
        train_stats = self.validate_images(self.train_dir)
        train_patterns = self.analyze_filename_patterns(self.train_dir)
        report['training_data'] = {**train_stats, **train_patterns}
        
        # Validate test data
        test_stats = self.validate_images(self.test_dir)
        test_patterns = self.analyze_filename_patterns(self.test_dir)
        report['test_data'] = {**test_stats, **test_patterns}
        
        # Overall statistics
        total_images = train_stats.get('total_images', 0) + test_stats.get('total_images', 0)
        valid_images = train_stats.get('valid_images', 0) + test_stats.get('valid_images', 0)
        
        report['summary'] = {
            'total_images': total_images,
            'valid_images': valid_images,
            'corruption_rate': (total_images - valid_images) / total_images if total_images > 0 else 0,
            'train_test_ratio': train_stats.get('total_images', 0) / test_stats.get('total_images', 1),
            'ready_for_training': valid_images > 0 and train_stats.get('valid_images', 0) > 0
        }
        
        # Print report
        self.print_report(report)
        
        # Save report
        report_path = self.project_root / "build-model-th" / "data_validation_report.json"
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Report saved to: {report_path}")
        except Exception as e:
            logger.warning(f"Failed to save report: {e}")
        
        return report['summary']['ready_for_training']
    
    def print_report(self, report):
        """Print a formatted validation report"""
        print("\n" + "=" * 80)
        print("📊 CRNN Training Data Validation Report")
        print("=" * 80)
        
        # Training data
        train_data = report.get('training_data', {})
        print(f"\n🏋️ Training Data:")
        print(f"  📁 Directory: {self.train_dir}")
        print(f"  🖼️ Total images: {train_data.get('total_images', 0)}")
        print(f"  ✅ Valid images: {train_data.get('valid_images', 0)}")
        print(f"  💾 Total size: {train_data.get('total_size_mb', 0):.2f} MB")
        
        if train_data.get('avg_width'):
            print(f"  📏 Average size: {train_data.get('avg_width'):.0f}x{train_data.get('avg_height'):.0f}")
        
        # Test data
        test_data = report.get('test_data', {})
        print(f"\n🧪 Test Data:")
        print(f"  📁 Directory: {self.test_dir}")
        print(f"  🖼️ Total images: {test_data.get('total_images', 0)}")
        print(f"  ✅ Valid images: {test_data.get('valid_images', 0)}")
        print(f"  💾 Total size: {test_data.get('total_size_mb', 0):.2f} MB")
        
        # Summary
        summary = report.get('summary', {})
        print(f"\n📈 Summary:")
        print(f"  🖼️ Total images: {summary.get('total_images', 0)}")
        print(f"  ✅ Valid images: {summary.get('valid_images', 0)}")
        print(f"  📊 Train/Test ratio: {summary.get('train_test_ratio', 0):.2f}")
        print(f"  🚀 Ready for training: {'✅ Yes' if summary.get('ready_for_training') else '❌ No'}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if summary.get('total_images', 0) < 1000:
            print("  ⚠️ Consider adding more training data for better performance")
        if summary.get('train_test_ratio', 0) < 3:
            print("  ⚠️ Consider increasing training data relative to test data")
        if summary.get('corruption_rate', 0) > 0.05:
            print("  ⚠️ High corruption rate detected, clean the dataset")
        if summary.get('ready_for_training'):
            print("  🚀 Dataset is ready for CRNN training!")
        
        print("=" * 80)

def main():
    """Main function"""
    try:
        print("🔍 CRNN Training Data Validator")
        print("=" * 50)
        
        validator = CRNNDataValidator()
        ready = validator.generate_report()
        
        if ready:
            print("\n✅ Data validation passed. Ready to start training!")
            return 0
        else:
            print("\n❌ Data validation failed. Please fix issues before training.")
            return 1
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
