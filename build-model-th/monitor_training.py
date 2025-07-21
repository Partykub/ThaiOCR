#!/usr/bin/env python3
"""
CRNN Training Monitor
Real-time monitoring of CRNN training progress
"""

import os
import time
import json
from pathlib import Path
import psutil

def monitor_training():
    """Monitor CRNN training progress"""
    project_root = Path(__file__).parent.parent
    log_file = project_root / "build-model-th" / "crnn_training.log"
    checkpoint_dir = project_root / "build-model-th" / "checkpoints"
    
    print("=" * 80)
    print("📊 CRNN Training Monitor - RTX 5090")
    print("=" * 80)
    
    # Monitor system resources
    print(f"\n💻 System Status:")
    print(f"  CPU Usage: {psutil.cpu_percent():.1f}%")
    print(f"  Memory Usage: {psutil.virtual_memory().percent:.1f}%")
    
    # Check GPU
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            print(f"  🎮 GPU: {gpu.name}")
            print(f"  GPU Memory: {gpu.memoryUtil*100:.1f}%")
            print(f"  GPU Temperature: {gpu.temperature}°C")
    except:
        print("  🎮 GPU monitoring not available")
    
    # Check training log
    if log_file.exists():
        print(f"\n📝 Training Log (last 10 lines):")
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[-10:]:
                print(f"  {line.strip()}")
    else:
        print(f"\n⚠️ Training log not found: {log_file}")
    
    # Check checkpoints
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.h5"))
        print(f"\n💾 Checkpoints ({len(checkpoints)}):")
        for checkpoint in sorted(checkpoints, key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
            mtime = time.ctime(checkpoint.stat().st_mtime)
            size_mb = checkpoint.stat().st_size / (1024 * 1024)
            print(f"  📁 {checkpoint.name} ({size_mb:.1f} MB) - {mtime}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    monitor_training()
