# Thai Dataset Generator

🎯 **Complete Thai Character Dataset Generation System**

A comprehensive toolkit for generating high-quality Thai character datasets for OCR training.

## ⚡ Quick Start

```bash
# Generate standard dataset (recommended)
python thai_dataset_quick.py 10

# Small test dataset
python thai_dataset_quick.py 5

# Large production dataset
python thai_dataset_quick.py 30
```

## 📦 What's Included

### 🎯 Main Generators
- **`thai_dataset_generator.py`** - Optimized generator (8 obstacles, 99.8% success)
- **`thai_dataset_generator_advanced.py`** - Advanced generator (15 obstacles)
- **`thai_dataset_quick.py`** - Easy-to-use helper

### 🛠️ Helper Tools
- **`thai_generator_helper.py`** - Interactive command builder
- **`thai_dataset_quick.bat`** - Windows batch menu

### 📚 Documentation
- **`THAI_DATASET_GUIDE.md`** - Complete user guide
- **`THAI_DATASET_ADVANCED_GUIDE.md`** - Advanced features guide

### 📊 Data Files
- **`th_dict.txt`** - 879 Thai characters dictionary
- **`thai_corpus.txt`** - Thai text corpus
- **`thai_dataset_sample/`** - Sample generated dataset

## 🎨 Features

### ✅ Optimized Obstacles (8 types)
- **Rotation**: ±2 degrees (gentle)
- **Brightness**: 0.8-1.2 (readable)
- **Contrast**: 0.8-1.2 (clear)
- **Blur**: 0-0.4 (minimal)
- **Noise**: 0-0.05 (low)
- **Position**: 3 variants (centered)
- **Padding**: 15-25 pixels
- **Compression**: 85-100% quality

### 📊 High Success Rate
- **99.8% success rate** (almost no errors)
- **Character visibility enhanced**
- **Suitable for OCR training**

### 🚀 Easy Usage
- **Command line interface**
- **Auto-generated output names**
- **Statistics and JSON output**
- **Cross-platform support**

## 🎯 Usage Examples

### Basic Usage
```bash
python thai_dataset_generator.py 15
```

### Advanced Usage
```bash
python thai_dataset_generator.py 20 -d th_dict.txt -o my_custom_dataset
```

### Quick Generation
```bash
# Interactive menu
python thai_dataset_quick.py 10

# Windows batch file
thai_dataset_quick.bat
```

## 📁 Output Structure

```
thai_dataset_standard_10samples_0722_1234/
├── images/
│   ├── 000_00.jpg    # Character 1, Sample 1
│   ├── 000_01.jpg    # Character 1, Sample 2
│   └── ...
├── labels.txt        # Image-to-character mapping
└── optimized_dataset_details.json  # Statistics & config
```

## 🎨 Dataset Categories

| Samples | Category | Use Case | Generation Time |
|---------|----------|----------|----------------|
| 5 | Test | Quick testing | 2-3 minutes |
| 10-15 | Standard | OCR training | 5-8 minutes |
| 20-30 | Large | High quality | 10-15 minutes |
| 50+ | Production | Professional | 20+ minutes |

## 🔧 Requirements

```bash
pip install pillow opencv-python numpy
```

## 📊 Comparison

| Generator | Obstacles | Success Rate | Character Visibility | Use Case |
|-----------|-----------|--------------|---------------------|----------|
| **Main** | 8 types | 99.8% | Excellent | Production |
| **Advanced** | 15 types | 94.6% | Good | Research |

## 🎉 Why Choose This Generator?

1. **🎯 Optimized for OCR** - Perfect balance of variation and readability
2. **⚡ Fast & Reliable** - 99.8% success rate with minimal errors
3. **🔧 Easy to Use** - Simple command line interface
4. **📊 Complete Output** - Images, labels, and statistics included
5. **🌐 Cross-platform** - Works on Windows, Mac, and Linux
6. **🎨 Flexible** - Multiple generators for different needs

## 🚀 Get Started

1. **Clone the repository**
2. **Install dependencies**: `pip install pillow opencv-python numpy`
3. **Generate your first dataset**: `python thai_dataset_quick.py 10`
4. **Check the results** in the generated folder

Perfect for OCR researchers, AI developers, and anyone working with Thai text recognition!

---

**⭐ Star this project if it helps you create better Thai OCR models!**
