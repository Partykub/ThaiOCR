#!/usr/bin/env python3
"""
Download PaddleOCR Pretrained Models
===================================

Download and prepare pretrained models for Thai OCR training
Supports both detection and recognition models

Author: Thai OCR Development Team
Date: July 22, 2025
"""

import os
import sys
import requests
import tarfile
import zipfile
import shutil
from pathlib import Path
import logging
from typing import Dict, List
from urllib.parse import urlparse
import hashlib
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PretrainedModelDownloader:
    """Download and manage pretrained models for PaddleOCR"""
    
    def __init__(self, models_dir: str = "./pretrain_models"):
        """Initialize downloader"""
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model URLs and info
        self.models = {
            # Recognition models (for Thai fine-tuning)
            "ch_PP-OCRv3_rec": {
                "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar",
                "desc": "Chinese PP-OCRv3 Recognition Model (for Thai fine-tuning)",
                "type": "recognition",
                "extract_dir": "ch_PP-OCRv3_rec_train",
                "required": True
            },
            "ch_PP-OCRv4_rec": {
                "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_train.tar",
                "desc": "Chinese PP-OCRv4 Recognition Model (latest)",
                "type": "recognition", 
                "extract_dir": "ch_PP-OCRv4_rec_train",
                "required": False
            },
            "en_PP-OCRv3_rec": {
                "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/en_PP-OCRv3_rec_train.tar",
                "desc": "English PP-OCRv3 Recognition Model",
                "type": "recognition",
                "extract_dir": "en_PP-OCRv3_rec_train",
                "required": False
            },
            
            # Detection models (pre-trained, ready to use)
            "ch_PP-OCRv3_det": {
                "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar",
                "desc": "Chinese PP-OCRv3 Detection Model (inference)",
                "type": "detection",
                "extract_dir": "ch_PP-OCRv3_det_infer",
                "required": True
            },
            "ch_PP-OCRv4_det": {
                "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar",
                "desc": "Chinese PP-OCRv4 Detection Model (latest)",
                "type": "detection",
                "extract_dir": "ch_PP-OCRv4_det_infer", 
                "required": False
            },
            
            # SVTR models (state-of-the-art recognition)
            "ch_PP-OCRv4_rec_SVTR": {
                "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_SVTR_train.tar",
                "desc": "SVTR Recognition Model (best accuracy)",
                "type": "recognition",
                "extract_dir": "ch_PP-OCRv4_rec_SVTR_train",
                "required": False
            },
            
            # Angle classifier
            "ch_ppocr_mobile_v2.0_cls": {
                "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar",
                "desc": "Text Angle Classifier",
                "type": "classifier",
                "extract_dir": "ch_ppocr_mobile_v2.0_cls_infer",
                "required": True
            }
        }
        
        # Download status file
        self.status_file = self.models_dir / "download_status.json"
        self.load_status()
    
    def load_status(self):
        """Load download status"""
        if self.status_file.exists():
            with open(self.status_file, 'r', encoding='utf-8') as f:
                self.status = json.load(f)
        else:
            self.status = {}
    
    def save_status(self):
        """Save download status"""
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump(self.status, f, indent=2)
    
    def calculate_md5(self, file_path: Path) -> str:
        """Calculate MD5 hash of file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def download_file(self, url: str, output_path: Path, desc: str = "") -> bool:
        """Download file with progress bar"""
        try:
            logger.info(f"📥 Downloading {desc}...")
            logger.info(f"    URL: {url}")
            logger.info(f"    Output: {output_path}")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Progress indicator
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\r    Progress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='')
                        else:
                            print(f"\r    Downloaded: {downloaded} bytes", end='')
            
            print()  # New line after progress
            logger.info(f"✅ Download completed: {output_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            if output_path.exists():
                output_path.unlink()  # Remove partial file
            return False
    
    def extract_archive(self, archive_path: Path, extract_to: Path) -> bool:
        """Extract tar/zip archive"""
        try:
            logger.info(f"📦 Extracting {archive_path.name}...")
            
            if archive_path.suffix.lower() in ['.tar', '.gz']:
                with tarfile.open(archive_path, 'r:*') as tar:
                    tar.extractall(extract_to)
            elif archive_path.suffix.lower() == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            else:
                logger.error(f"❌ Unsupported archive format: {archive_path.suffix}")
                return False
            
            logger.info(f"✅ Extracted to: {extract_to}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Extraction failed: {e}")
            return False
    
    def download_model(self, model_name: str, force: bool = False) -> bool:
        """Download and extract a specific model"""
        if model_name not in self.models:
            logger.error(f"❌ Unknown model: {model_name}")
            return False
        
        model_info = self.models[model_name]
        model_dir = self.models_dir / model_name
        
        # Check if already downloaded and extracted
        expected_dir = self.models_dir / model_info["extract_dir"]
        if not force and expected_dir.exists() and model_name in self.status:
            logger.info(f"✅ Model already downloaded: {model_name}")
            return True
        
        # Download archive
        url = model_info["url"]
        filename = Path(urlparse(url).path).name
        archive_path = self.models_dir / filename
        
        if not archive_path.exists() or force:
            if not self.download_file(url, archive_path, model_info["desc"]):
                return False
        else:
            logger.info(f"📁 Archive already exists: {archive_path.name}")
        
        # Extract archive
        if not self.extract_archive(archive_path, self.models_dir):
            return False
        
        # Verify extraction
        if not expected_dir.exists():
            logger.error(f"❌ Expected directory not found: {expected_dir}")
            return False
        
        # Update status
        self.status[model_name] = {
            "downloaded": True,
            "path": str(expected_dir),
            "type": model_info["type"],
            "desc": model_info["desc"]
        }
        self.save_status()
        
        # Clean up archive (optional)
        if archive_path.exists():
            try:
                archive_path.unlink()
                logger.info(f"🗑️ Cleaned up archive: {archive_path.name}")
            except:
                pass
        
        logger.info(f"✅ Model ready: {model_name}")
        return True
    
    def download_required_models(self, force: bool = False) -> bool:
        """Download all required models for Thai OCR training"""
        logger.info("📚 DOWNLOADING REQUIRED MODELS FOR THAI OCR...")
        
        required_models = [name for name, info in self.models.items() if info.get("required", False)]
        
        logger.info(f"📋 Required models: {', '.join(required_models)}")
        
        success_count = 0
        for model_name in required_models:
            if self.download_model(model_name, force):
                success_count += 1
            else:
                logger.error(f"❌ Failed to download required model: {model_name}")
        
        if success_count == len(required_models):
            logger.info("✅ All required models downloaded successfully!")
            return True
        else:
            logger.error(f"❌ Only {success_count}/{len(required_models)} required models downloaded")
            return False
    
    def download_all_models(self, force: bool = False) -> bool:
        """Download all available models"""
        logger.info("📚 DOWNLOADING ALL MODELS...")
        
        success_count = 0
        for model_name in self.models.keys():
            if self.download_model(model_name, force):
                success_count += 1
        
        logger.info(f"✅ Downloaded {success_count}/{len(self.models)} models")
        return success_count > 0
    
    def list_models(self, show_downloaded: bool = True):
        """List available models"""
        logger.info("📋 AVAILABLE MODELS:")
        
        for model_name, model_info in self.models.items():
            status_icon = "✅" if model_name in self.status else "⬜"
            required_icon = "🔴" if model_info.get("required", False) else "🔵"
            
            if show_downloaded or model_name not in self.status:
                logger.info(f"  {status_icon} {required_icon} {model_name}")
                logger.info(f"      Type: {model_info['type']}")
                logger.info(f"      Desc: {model_info['desc']}")
                
                if model_name in self.status:
                    logger.info(f"      Path: {self.status[model_name]['path']}")
                logger.info("")
        
        logger.info("Legend: ✅=Downloaded, ⬜=Not Downloaded, 🔴=Required, 🔵=Optional")
    
    def verify_models(self) -> Dict[str, bool]:
        """Verify downloaded models"""
        logger.info("🔍 VERIFYING DOWNLOADED MODELS...")
        
        verification_results = {}
        
        for model_name, status_info in self.status.items():
            model_path = Path(status_info["path"])
            
            if model_path.exists():
                # Check for key files
                key_files = ["inference.pdmodel", "inference.pdiparams", "inference.pdiparams.info"]
                has_inference = any((model_path / f).exists() for f in key_files)
                
                # Check for training files  
                train_files = ["model.pdmodel", "model.pdiparams", "model.pdopt"]
                has_training = any((model_path / f).exists() for f in train_files)
                
                if has_inference or has_training:
                    verification_results[model_name] = True
                    logger.info(f"✅ {model_name}: Valid")
                else:
                    verification_results[model_name] = False
                    logger.warning(f"⚠️ {model_name}: No valid model files found")
            else:
                verification_results[model_name] = False
                logger.error(f"❌ {model_name}: Directory not found")
        
        valid_count = sum(verification_results.values())
        total_count = len(verification_results)
        
        logger.info(f"📊 Verification: {valid_count}/{total_count} models valid")
        return verification_results
    
    def create_model_configs(self):
        """Create model configuration for easy access"""
        logger.info("⚙️ CREATING MODEL CONFIGURATIONS...")
        
        config = {
            "pretrained_models": {},
            "download_time": str(Path(__file__).stat().st_mtime),
            "total_models": len(self.status)
        }
        
        for model_name, status_info in self.status.items():
            config["pretrained_models"][model_name] = {
                "path": status_info["path"],
                "type": status_info["type"],
                "desc": status_info["desc"],
                "available": Path(status_info["path"]).exists()
            }
        
        config_file = self.models_dir / "models_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Model config created: {config_file}")
        
        # Create symlinks for easy access (if supported)
        try:
            inference_dir = self.models_dir / "inference"
            training_dir = self.models_dir / "training"
            
            inference_dir.mkdir(exist_ok=True)
            training_dir.mkdir(exist_ok=True)
            
            for model_name, status_info in self.status.items():
                model_path = Path(status_info["path"])
                model_type = status_info["type"]
                
                if model_path.exists():
                    # Inference models
                    if any((model_path / f).exists() for f in ["inference.pdmodel", "inference.pdiparams"]):
                        link_path = inference_dir / model_name
                        if not link_path.exists():
                            try:
                                os.symlink(model_path, link_path)
                            except:
                                shutil.copytree(model_path, link_path, dirs_exist_ok=True)
                    
                    # Training models
                    if any((model_path / f).exists() for f in ["model.pdmodel", "model.pdiparams"]):
                        link_path = training_dir / model_name
                        if not link_path.exists():
                            try:
                                os.symlink(model_path, link_path)
                            except:
                                shutil.copytree(model_path, link_path, dirs_exist_ok=True)
        
        except Exception as e:
            logger.warning(f"⚠️ Could not create symlinks: {e}")
        
        return str(config_file)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download PaddleOCR pretrained models')
    parser.add_argument('--models_dir', type=str, default='./pretrain_models',
                       help='Directory to store models (default: ./pretrain_models)')
    parser.add_argument('--all', action='store_true',
                       help='Download all models (not just required ones)')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download even if models exist')
    parser.add_argument('--list', action='store_true',
                       help='List available models')
    parser.add_argument('--verify', action='store_true',
                       help='Verify downloaded models')
    parser.add_argument('--model', type=str,
                       help='Download specific model by name')
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = PretrainedModelDownloader(args.models_dir)
    
    try:
        if args.list:
            downloader.list_models()
            return
        
        if args.verify:
            downloader.verify_models()
            return
        
        if args.model:
            # Download specific model
            success = downloader.download_model(args.model, args.force)
            if success:
                print(f"✅ Model {args.model} downloaded successfully!")
            else:
                print(f"❌ Failed to download model {args.model}")
                sys.exit(1)
        elif args.all:
            # Download all models
            success = downloader.download_all_models(args.force)
        else:
            # Download required models only
            success = downloader.download_required_models(args.force)
        
        if success:
            # Create configurations
            config_file = downloader.create_model_configs()
            
            print(f"\n🎉 SUCCESS!")
            print(f"📁 Models directory: {args.models_dir}")
            print(f"⚙️ Config file: {config_file}")
            print(f"🚀 Ready for Task 7: PaddleOCR Thai Model Training!")
            
            # Show next steps
            print(f"\n📋 NEXT STEPS:")
            print(f"1. Prepare dataset: python src/data/prepare_paddle_dataset.py")
            print(f"2. Start training: python tools/train.py -c configs/rec/thai_svtr_tiny.yml")
        else:
            print(f"❌ Failed to download models")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
