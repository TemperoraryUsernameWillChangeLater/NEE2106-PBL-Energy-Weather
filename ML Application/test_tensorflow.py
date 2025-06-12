#!/usr/bin/env python3
"""
Test TensorFlow Installation
"""

print("🧪 Testing TensorFlow Installation...")
print("-" * 40)

try:
    import tensorflow as tf
    print(f"✅ TensorFlow imported successfully!")
    print(f"📦 TensorFlow version: {tf.__version__}")
    
    # Test basic operations
    print("\n🔧 Testing basic TensorFlow operations...")
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    c = tf.matmul(a, b)
    print(f"✅ Matrix multiplication test passed!")
    print(f"Result: \n{c.numpy()}")
    
    # Test GPU availability
    print(f"\n🖥️  GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    # Test other required packages
    print("\n📚 Testing other required packages...")
    
    import pandas as pd
    print(f"✅ Pandas: {pd.__version__}")
    
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
    
    import matplotlib.pyplot as plt
    print(f"✅ Matplotlib: {plt.matplotlib.__version__}")
    
    import seaborn as sns
    print(f"✅ Seaborn: {sns.__version__}")
    
    import sklearn
    print(f"✅ Scikit-learn: {sklearn.__version__}")
    
    # Test LSTM layer creation
    print("\n🧠 Testing LSTM layer creation...")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    
    model = Sequential([
        LSTM(10, return_sequences=True, input_shape=(5, 1)),
        Dropout(0.2),
        LSTM(10, return_sequences=False),
        Dense(1)
    ])
    
    print(f"✅ LSTM model created successfully!")
    print(f"📊 Model summary:")
    model.summary()
    
    print("\n🎉 ALL TESTS PASSED!")
    print("🚀 TensorFlow is ready for your ML application!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Try installing missing packages with: pip install tensorflow pandas numpy matplotlib seaborn scikit-learn")
    
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    print("💡 Please check your Python environment")

print("\n" + "=" * 50)
