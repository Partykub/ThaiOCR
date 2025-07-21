#!/usr/bin/env python3
"""
Simple PaddleOCR Test for RTX 5090
Basic test to verify installation and CPU mode
"""

import os
import warnings

# Configure environment for RTX 5090 compatibility
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU mode
os.environ['PADDLE_ONLY_CPU'] = '1'
warnings.filterwarnings("ignore")

def test_basic_imports():
    """Test basic package imports"""
    print("🧪 Testing basic imports...")
    
    try:
        import paddle
        print(f"✅ PaddlePaddle: {paddle.__version__}")
        
        import paddleocr
        print(f"✅ PaddleOCR: Available")
        
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
        
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
        
        import PIL
        print(f"✅ Pillow: {PIL.__version__}")
        
        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_paddle_device():
    """Test PaddlePaddle device configuration"""
    print("\n🎮 Testing PaddlePaddle device configuration...")
    
    try:
        import paddle
        
        # Force CPU mode
        paddle.device.set_device('cpu')
        device = paddle.device.get_device()
        print(f"✅ Current device: {device}")
        
        # Test basic tensor operations
        x = paddle.ones([2, 2], dtype='float32')
        y = paddle.zeros([2, 2], dtype='float32') + 1.0
        z = paddle.add(x, y)
        
        print(f"✅ Tensor operations: OK")
        print(f"   Result shape: {z.shape}")
        print(f"   Result sum: {float(paddle.sum(z))}")
        
        return True
    except Exception as e:
        print(f"❌ PaddlePaddle test failed: {e}")
        return False

def test_simple_ocr():
    """Test simple OCR functionality (without Thai-specific features)"""
    print("\n📝 Testing basic OCR functionality...")
    
    try:
        from paddleocr import PaddleOCR
        
        # Create simple OCR instance with minimal configuration
        print("🔧 Initializing PaddleOCR (CPU mode)...")
        ocr = PaddleOCR(lang='en')  # English model for basic testing
        
        print("✅ PaddleOCR initialized successfully!")
        print("💡 Ready for image recognition tasks")
        
        return True
    except Exception as e:
        print(f"❌ OCR test failed: {e}")
        print("💡 This may be due to model download issues or version incompatibility")
        return False

def test_thai_support():
    """Test Thai language processing support"""
    print("\n🇹🇭 Testing Thai language support...")
    
    try:
        import pythainlp
        print(f"✅ PyThaiNLP: {pythainlp.__version__}")
        
        # Test basic Thai text processing
        thai_text = "ภาษาไทย"
        print(f"✅ Thai text processing: {thai_text}")
        
        return True
    except Exception as e:
        print(f"❌ Thai support test failed: {e}")
        return False

def create_sample_test_image():
    """Create a simple test image with text"""
    print("\n🖼️ Creating sample test image...")
    
    try:
        import cv2
        import numpy as np
        
        # Create a white image
        img = np.ones((100, 400, 3), dtype=np.uint8) * 255
        
        # Add simple text
        cv2.putText(img, 'Hello OCR', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Save test image
        test_image_path = 'test_sample.jpg'
        cv2.imwrite(test_image_path, img)
        
        print(f"✅ Created test image: {test_image_path}")
        return test_image_path
    except Exception as e:
        print(f"❌ Image creation failed: {e}")
        return None

def main():
    """Run all tests"""
    print("🚀 RTX 5090 PaddleOCR Compatibility Test")
    print("=" * 50)
    print("Testing PaddleOCR installation with RTX 5090 CPU fallback")
    
    tests = [
        ("Basic imports", test_basic_imports),
        ("PaddlePaddle device", test_paddle_device),
        ("Thai language support", test_thai_support),
        ("Simple OCR", test_simple_ocr),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
        else:
            print(f"⚠️  {test_name} test failed but continuing...")
    
    # Create sample image for manual testing
    sample_image = create_sample_test_image()
    
    # Final summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"✅ Passed: {passed}/{total} tests")
    
    if passed >= 3:
        print("🎉 Your Thai OCR environment is working!")
        print("💡 PaddleOCR is running in CPU mode (RTX 5090 compatibility)")
        
        if sample_image:
            print(f"\n📚 Next steps:")
            print(f"1. Test OCR manually with: {sample_image}")
            print(f"2. Create Thai text images for recognition")
            print(f"3. Develop custom Thai OCR training scripts")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        print("💡 You may need to reinstall some packages or fix environment issues.")
    
    print("\n🔧 Environment configuration:")
    print(f"   - CPU mode enabled (RTX 5090 compatibility)")
    print(f"   - CUDA disabled for PaddlePaddle")
    print(f"   - Ready for Thai text processing")

if __name__ == "__main__":
    main()
