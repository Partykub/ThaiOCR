#!/usr/bin/env python3
"""
Task 7 Phase 1 Progress Tracker
==============================

Track and manage Phase 1: Environment & Dataset Preparation progress
Provides automated checks and next steps guidance

Author: Thai OCR Development Team
Date: July 22, 2025
"""

import os
import sys
import json
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import cv2

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Phase1ProgressTracker:
    """Track Phase 1 implementation progress"""
    
    def __init__(self):
        """Initialize progress tracker"""
        self.progress_file = Path("logs/task7_phase1_progress.json")
        self.progress_file.parent.mkdir(exist_ok=True)
        
        # Phase 1 tasks checklist
        self.tasks = {
            "rtx5090_environment": {
                "name": "RTX 5090 Environment Setup",
                "status": "completed",  # Already done
                "weight": 10
            },
            "paddlepaddle_installation": {
                "name": "PaddlePaddle GPU Installation (v2.6.2)",
                "status": "completed",  # Already done
                "weight": 10
            },
            "dataset_analysis": {
                "name": "Dataset Analysis (14,672 images)",
                "status": "completed",  # Already done
                "weight": 10
            },
            "build_paddle_sm120": {
                "name": "Build PaddlePaddle with SM_120 support",
                "status": "pending",
                "weight": 25,
                "script": "src/utils/build_paddle_sm120.py"
            },
            "dataset_converter": {
                "name": "Create PaddleOCR dataset format converter",
                "status": "pending",
                "weight": 20,
                "script": "src/data/prepare_paddle_dataset.py"
            },
            "train_val_split": {
                "name": "Setup training/validation split (80:20)",
                "status": "pending",
                "weight": 10
            },
            "download_pretrained": {
                "name": "Download pretrained models",
                "status": "pending",
                "weight": 15,
                "script": "src/utils/download_pretrained.py"
            }
        }
        
        self.load_progress()
    
    def load_progress(self):
        """Load existing progress"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                saved_progress = json.load(f)
                # Update tasks with saved status
                for task_id, task_data in saved_progress.get('tasks', {}).items():
                    if task_id in self.tasks:
                        self.tasks[task_id]['status'] = task_data.get('status', 'pending')
    
    def save_progress(self):
        """Save current progress"""
        progress_data = {
            "tasks": self.tasks,
            "last_updated": str(Path(__file__).stat().st_mtime),
            "completion_percentage": self.get_completion_percentage()
        }
        
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=2)
    
    def get_completion_percentage(self) -> float:
        """Calculate completion percentage"""
        total_weight = sum(task['weight'] for task in self.tasks.values())
        completed_weight = sum(
            task['weight'] for task in self.tasks.values() 
            if task['status'] == 'completed'
        )
        return (completed_weight / total_weight) * 100
    
    def check_gpu_environment(self) -> bool:
        """Check RTX 5090 GPU environment"""
        logger.info("🎮 CHECKING RTX 5090 ENVIRONMENT...")
        
        try:
            # Check NVIDIA-SMI
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0 and 'RTX 5090' in result.stdout:
                logger.info("✅ RTX 5090 detected via nvidia-smi")
                return True
            
            # Check CUDA availability
            test_script = '''
import paddle
if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
    print("GPU_AVAILABLE")
else:
    print("GPU_NOT_AVAILABLE")
'''
            result = subprocess.run([sys.executable, '-c', test_script], 
                                  capture_output=True, text=True)
            
            if 'GPU_AVAILABLE' in result.stdout:
                logger.info("✅ GPU available via PaddlePaddle")
                return True
            else:
                logger.warning("⚠️ GPU not available via PaddlePaddle")
                return False
                
        except Exception as e:
            logger.error(f"❌ GPU check failed: {e}")
            return False
    
    def check_paddlepaddle_installation(self) -> bool:
        """Check PaddlePaddle installation"""
        logger.info("🌊 CHECKING PADDLEPADDLE INSTALLATION...")
        
        try:
            import paddle
            import paddleocr
            
            logger.info(f"✅ PaddlePaddle: {paddle.__version__}")
            logger.info(f"✅ PaddleOCR: {paddleocr.__version__}")
            
            if paddle.device.is_compiled_with_cuda():
                logger.info("✅ CUDA support: Available")
                return True
            else:
                logger.warning("⚠️ CUDA support: Not available")
                return False
                
        except ImportError as e:
            logger.error(f"❌ PaddlePaddle/PaddleOCR not installed: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ PaddlePaddle check failed: {e}")
            return False
    
    def check_dataset_availability(self) -> Tuple[bool, Dict]:
        """Check dataset availability and statistics"""
        logger.info("📊 CHECKING DATASET AVAILABILITY...")
        
        dataset_info = {
            "thai_ocr_dataset": {"path": "thai-letters/thai_ocr_dataset", "images": 0},
            "full_corpus_dataset": {"path": "thai-letters/full_corpus_dataset", "images": 0},
            "total_images": 0,
            "dict_file": "thai-letters/th_dict.txt"
        }
        
        try:
            # Check main datasets
            for dataset_name, info in dataset_info.items():
                if dataset_name in ["thai_ocr_dataset", "full_corpus_dataset"]:
                    dataset_path = Path(info["path"])
                    images_dir = dataset_path / "images"
                    labels_file = dataset_path / "labels.txt"
                    
                    if images_dir.exists() and labels_file.exists():
                        # Count images
                        image_count = len([f for f in images_dir.glob("*.jpg")])
                        info["images"] = image_count
                        dataset_info["total_images"] += image_count
                        logger.info(f"✅ {dataset_name}: {image_count} images")
                    else:
                        logger.warning(f"⚠️ {dataset_name}: Missing files")
            
            # Check dictionary
            dict_path = Path(dataset_info["dict_file"])
            if dict_path.exists():
                with open(dict_path, 'r', encoding='utf-8') as f:
                    char_count = len(f.readlines())
                logger.info(f"✅ Dictionary: {char_count} characters")
                dataset_info["characters"] = char_count
            else:
                logger.warning("⚠️ Dictionary file missing")
                dataset_info["characters"] = 0
            
            logger.info(f"📊 Total dataset: {dataset_info['total_images']} images")
            
            return dataset_info["total_images"] > 1000, dataset_info
            
        except Exception as e:
            logger.error(f"❌ Dataset check failed: {e}")
            return False, dataset_info
    
    def check_file_exists(self, file_path: str) -> bool:
        """Check if a file exists"""
        return Path(file_path).exists()
    
    def run_script_check(self, script_path: str) -> bool:
        """Check if a script can be run successfully"""
        try:
            # Check if file exists
            if not Path(script_path).exists():
                return False
            
            # Try to import/parse the script
            result = subprocess.run([
                sys.executable, '-c', f'import py_compile; py_compile.compile("{script_path}", doraise=True)'
            ], capture_output=True)
            
            return result.returncode == 0
        except:
            return False
    
    def update_task_status(self, task_id: str, status: str):
        """Update task status"""
        if task_id in self.tasks:
            self.tasks[task_id]['status'] = status
            self.save_progress()
            logger.info(f"✅ Updated {task_id}: {status}")
    
    def run_automatic_checks(self):
        """Run automatic checks for all tasks"""
        logger.info("🔍 RUNNING AUTOMATIC CHECKS...")
        
        # Check RTX 5090 environment
        if self.check_gpu_environment():
            self.update_task_status('rtx5090_environment', 'completed')
        
        # Check PaddlePaddle installation
        if self.check_paddlepaddle_installation():
            self.update_task_status('paddlepaddle_installation', 'completed')
        
        # Check dataset
        dataset_ok, dataset_info = self.check_dataset_availability()
        if dataset_ok:
            self.update_task_status('dataset_analysis', 'completed')
        
        # Check script files
        script_checks = {
            'build_paddle_sm120': 'src/utils/build_paddle_sm120.py',
            'dataset_converter': 'src/data/prepare_paddle_dataset.py',
            'download_pretrained': 'src/utils/download_pretrained.py'
        }
        
        for task_id, script_path in script_checks.items():
            if self.run_script_check(script_path):
                if self.tasks[task_id]['status'] == 'pending':
                    self.update_task_status(task_id, 'ready')
            else:
                logger.warning(f"⚠️ Script not ready: {script_path}")
        
        # Check if pretrained models are downloaded
        if Path('pretrain_models').exists() and list(Path('pretrain_models').glob('*')):
            self.update_task_status('download_pretrained', 'completed')
        
        # Check if dataset is converted
        if Path('paddle_dataset').exists() and Path('paddle_dataset/recognition').exists():
            self.update_task_status('dataset_converter', 'completed')
            self.update_task_status('train_val_split', 'completed')
    
    def show_progress_report(self):
        """Show detailed progress report"""
        logger.info("📋 TASK 7 - PHASE 1 PROGRESS REPORT")
        logger.info("=" * 50)
        
        completion = self.get_completion_percentage()
        logger.info(f"📊 Overall Progress: {completion:.1f}%")
        logger.info("")
        
        # Progress bar
        bar_length = 30
        filled_length = int(bar_length * completion / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        logger.info(f"Progress: |{bar}| {completion:.1f}%")
        logger.info("")
        
        # Task breakdown
        logger.info("📝 Task Status:")
        for task_id, task in self.tasks.items():
            status_icon = {
                'completed': '✅',
                'ready': '🔄',
                'pending': '⏳',
                'failed': '❌'
            }.get(task['status'], '❓')
            
            logger.info(f"  {status_icon} {task['name']} ({task['weight']}%)")
            
            if 'script' in task and task['status'] in ['ready', 'pending']:
                logger.info(f"      Script: {task['script']}")
        
        logger.info("")
        
        # Next steps
        self.show_next_steps()
    
    def show_next_steps(self):
        """Show next steps based on current progress"""
        logger.info("🚀 NEXT STEPS:")
        
        pending_tasks = [
            (task_id, task) for task_id, task in self.tasks.items()
            if task['status'] in ['pending', 'ready']
        ]
        
        if not pending_tasks:
            logger.info("🎉 Phase 1 COMPLETED! Ready for Phase 2: Recognition Model Training")
            logger.info("")
            logger.info("📋 Phase 2 Commands:")
            logger.info("1. Configure training: vim configs/rec/thai_svtr_tiny.yml")
            logger.info("2. Start training: python tools/train.py -c configs/rec/thai_svtr_tiny.yml")
            return
        
        # Sort by priority (highest weight first)
        pending_tasks.sort(key=lambda x: x[1]['weight'], reverse=True)
        
        for i, (task_id, task) in enumerate(pending_tasks[:3], 1):  # Show top 3
            logger.info(f"{i}. {task['name']}")
            
            if 'script' in task:
                logger.info(f"   Command: python {task['script']}")
            
            # Specific instructions
            if task_id == 'build_paddle_sm120':
                logger.info("   Note: This will take 30-60 minutes to compile")
            elif task_id == 'dataset_converter':
                logger.info("   Args: --input_dir thai-letters/thai_ocr_dataset --output_dir ./paddle_dataset")
            elif task_id == 'download_pretrained':
                logger.info("   Note: Downloads ~2GB of pretrained models")
        
        logger.info("")
        logger.info("💡 Tip: Run tasks in order of priority (highest weight first)")
    
    def run_phase1_summary(self):
        """Run complete Phase 1 summary and checks"""
        logger.info("🔍 TASK 7 - PHASE 1 ENVIRONMENT & DATASET PREPARATION")
        logger.info("=" * 60)
        
        # Run automatic checks
        self.run_automatic_checks()
        
        # Show progress report
        self.show_progress_report()
        
        # Save progress
        self.save_progress()
        
        completion = self.get_completion_percentage()
        
        if completion >= 100:
            logger.info("🎉 PHASE 1 COMPLETED SUCCESSFULLY!")
            logger.info("🚀 Ready to proceed to Phase 2: Recognition Model Training")
            return True
        else:
            logger.info(f"⏳ Phase 1 is {completion:.1f}% complete")
            logger.info("🔧 Please complete remaining tasks before proceeding to Phase 2")
            return False

def main():
    """Main function"""
    tracker = Phase1ProgressTracker()
    
    try:
        success = tracker.run_phase1_summary()
        
        if success:
            print("\n🎉 SUCCESS! Phase 1 completed!")
            print("📋 Ready for Phase 2: Recognition Model Training")
            sys.exit(0)
        else:
            print("\n⏳ Phase 1 in progress...")
            print("🔧 Complete remaining tasks to proceed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Phase 1 check failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
