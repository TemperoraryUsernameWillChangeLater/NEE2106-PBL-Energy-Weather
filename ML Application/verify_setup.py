#!/usr/bin/env python3
"""
Quick TensorFlow Verification Script
Run this to verify TensorFlow is working properly
"""

def test_tensorflow():
    """Test TensorFlow installation and basic functionality"""
    print("🔧 Testing TensorFlow Installation...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} installed successfully!")
        
        # Test basic tensor operations
        x = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
        y = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)
        z = tf.matmul(x, y)
        
        print(f"✅ Basic tensor operations working!")
        
        # Test model creation
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, LSTM
        
        model = Sequential([
            Dense(10, activation='relu', input_shape=(5,)),
            Dense(1)
        ])
        
        print(f"✅ Model creation working!")
        print(f"📊 Available physical devices: {len(tf.config.list_physical_devices())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 TENSORFLOW VERIFICATION")
    print("=" * 50)
    
    success = test_tensorflow()
    
    if success:
        print("\n🎉 TensorFlow is ready for your ML applications!")
        print("💡 You can now run ML.py for energy forecasting")
    else:
        print("\n❌ TensorFlow installation needs attention")
        print("💡 Try: pip install tensorflow")
    
    print("=" * 50)
