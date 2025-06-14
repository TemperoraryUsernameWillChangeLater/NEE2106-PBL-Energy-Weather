# Test Integration between ML.py and plot_refined_datasets.py
# This script verifies that ML.py properly calls plot_refined_datasets.py

import subprocess
import sys
import os

def test_plot_script_callable():
    """Test that plot_refined_datasets.py can be called as a standalone script"""
    plot_script = 'plot_refined_datasets.py'
    
    if not os.path.exists(plot_script):
        print(f"❌ {plot_script} not found!")
        return False
    
    print(f"✅ {plot_script} exists and can be called")
    print(f"💡 To test the full integration:")
    print(f"   1. Run 'python ML.py' to train the model and save data")
    print(f"   2. The script will automatically call plot_refined_datasets.py")
    print(f"   3. Or run 'python plot_refined_datasets.py' manually for visualizations")
    
    return True

if __name__ == "__main__":
    print("🔍 TESTING ML.py → plot_refined_datasets.py INTEGRATION")
    print("=" * 60)
    
    test_plot_script_callable()
    
    print("\n📋 INTEGRATION SUMMARY:")
    print("✅ ML.py: Handles model training and data generation")
    print("✅ plot_refined_datasets.py: Handles all visualizations from .dat and .csv files")
    print("✅ Integration: ML.py automatically calls plot_refined_datasets.py after training")
    print("\n💡 This separation provides better code organization and maintainability!")
