#!/usr/bin/env python3
"""
CRNN Compatibility Fixer
Fix import issues for newer TensorFlow/Keras versions
"""

import os
import shutil
from pathlib import Path

def fix_crnn_compatibility(project_root=None):
    """Fix CRNN model imports for TensorFlow 2.x compatibility"""
    
    if project_root is None:
        project_root = Path(__file__).parent.parent
    else:
        project_root = Path(project_root)
    
    crnn_dir = project_root / "thai-license-plate-recognition-CRNN"
    
    print("🔧 Fixing CRNN compatibility for TensorFlow 2.x")
    print("=" * 60)
    
    # Fix Model.py
    model_file = crnn_dir / "Model.py"
    if model_file.exists():
        print(f"📝 Fixing {model_file}")
        
        # Read original content
        with open(model_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create backup
        backup_file = model_file.with_suffix('.py.backup')
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"💾 Backup created: {backup_file}")
        
        # Apply fixes
        fixes = [
            # Fix imports for TensorFlow 2.x
            ('from keras.layers.merge import add, concatenate', 'from keras.layers import add, concatenate'),
            ('from keras.layers.recurrent import LSTM', 'from keras.layers import LSTM'),
            ('from keras import backend as K', 'from tensorflow.keras import backend as K'),
            ('from keras.layers import', 'from tensorflow.keras.layers import'),
            ('from keras.models import', 'from tensorflow.keras.models import'),
            ('from keras.optimizers import', 'from tensorflow.keras.optimizers import'),
            ('from keras.callbacks import', 'from tensorflow.keras.callbacks import'),
            ('K.set_learning_phase(0)', '# K.set_learning_phase(0)  # Not needed in TF 2.x'),
        ]
        
        for old, new in fixes:
            if old in content:
                content = content.replace(old, new)
                print(f"  ✅ Fixed: {old[:50]}...")
        
        # Write fixed content
        with open(model_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Fixed {model_file}")
    
    # Fix training.py
    training_file = crnn_dir / "training.py"
    if training_file.exists():
        print(f"📝 Fixing {training_file}")
        
        # Read original content
        with open(training_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create backup
        backup_file = training_file.with_suffix('.py.backup')
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"💾 Backup created: {backup_file}")
        
        # Apply fixes
        fixes = [
            ('from keras import backend as K', 'from tensorflow.keras import backend as K'),
            ('from keras.optimizers import', 'from tensorflow.keras.optimizers import'),
            ('from keras.callbacks import', 'from tensorflow.keras.callbacks import'),
            ('K.set_learning_phase(0)', '# K.set_learning_phase(0)  # Not needed in TF 2.x'),
        ]
        
        for old, new in fixes:
            if old in content:
                content = content.replace(old, new)
                print(f"  ✅ Fixed: {old[:50]}...")
        
        # Write fixed content
        with open(training_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Fixed {training_file}")
    
    # Fix Image_Generator.py
    generator_file = crnn_dir / "Image_Generator.py"
    if generator_file.exists():
        print(f"📝 Fixing {generator_file}")
        
        # Read original content
        with open(generator_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create backup
        backup_file = generator_file.with_suffix('.py.backup')
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"💾 Backup created: {backup_file}")
        
        # Apply fixes for imports
        fixes = [
            ('from keras import backend as K', 'from tensorflow.keras import backend as K'),
            ('from keras.preprocessing import image', 'from tensorflow.keras.preprocessing import image'),
            ('from keras.applications.vgg16 import preprocess_input', 'from tensorflow.keras.applications.vgg16 import preprocess_input'),
        ]
        
        for old, new in fixes:
            if old in content:
                content = content.replace(old, new)
                print(f"  ✅ Fixed: {old[:50]}...")
        
        # Write fixed content
        with open(generator_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Fixed {generator_file}")
    
    print("\n🎉 CRNN compatibility fixes completed!")
    print("📝 Backup files created with .backup extension")
    print("🚀 Ready to start training with TensorFlow 2.x")

if __name__ == "__main__":
    fix_crnn_compatibility()
