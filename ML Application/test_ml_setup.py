# Quick Test Script for ML Application
# Tests basic functionality before running full ML pipeline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def test_data_loading():
    """Test if we can load the energy dataset"""
    print("🔍 Testing Data Loading...")
    
    data_path = r"C:\Users\gabri\Documents\Python\(NEE2106) Computer Programming For Electrical Engineers\Session 5\Integrated Energy Management and Forecasting Dataset.csv"
    
    try:
        data = pd.read_csv(data_path)
        print(f"✅ Data loaded successfully!")
        print(f"   Shape: {data.shape}")
        print(f"   Columns: {list(data.columns)}")
        print(f"\n📊 First 5 rows:")
        print(data.head())
        
        # Check for target variable
        if 'Energy_Demand' in data.columns:
            print(f"\n🎯 Target variable found: Energy_Demand")
            print(f"   Range: {data['Energy_Demand'].min():.2f} - {data['Energy_Demand'].max():.2f}")
            print(f"   Mean: {data['Energy_Demand'].mean():.2f}")
        
        # Check for weather features
        weather_features = ['Temperature', 'Weather_Condition_x', 'Weather_Condition_y']
        available_weather = [f for f in weather_features if f in data.columns]
        print(f"\n🌤️ Available weather features: {available_weather}")
        
        return True, data
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False, None

def test_basic_visualization(data):
    """Create basic visualizations to understand the data"""
    print(f"\n📈 Creating basic visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Energy demand over time
    axes[0, 0].plot(data['Energy_Demand'][:100], color='blue', linewidth=2)
    axes[0, 0].set_title('Energy Demand Over Time (First 100 Points)')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Energy Demand')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Energy demand distribution
    axes[0, 1].hist(data['Energy_Demand'], bins=30, color='skyblue', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Energy Demand Distribution')
    axes[0, 1].set_xlabel('Energy Demand')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Temperature vs Energy Demand
    if 'Temperature' in data.columns:
        axes[1, 0].scatter(data['Temperature'], data['Energy_Demand'], alpha=0.5, s=20)
        axes[1, 0].set_title('Temperature vs Energy Demand')
        axes[1, 0].set_xlabel('Temperature')
        axes[1, 0].set_ylabel('Energy Demand')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Energy Supply vs Demand
    if 'Energy_Supply' in data.columns:
        axes[1, 1].scatter(data['Energy_Supply'], data['Energy_Demand'], alpha=0.5, s=20, color='green')
        axes[1, 1].set_title('Energy Supply vs Energy Demand')
        axes[1, 1].set_xlabel('Energy Supply')
        axes[1, 1].set_ylabel('Energy Demand')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Basic visualizations created and saved as 'data_exploration.png'")

def test_correlation_analysis(data):
    """Analyze correlations between features and target"""
    print(f"\n🔍 Correlation Analysis...")
    
    # Select numerical columns
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    correlation_with_target = data[numerical_cols].corrwith(data['Energy_Demand'])
    
    print("📊 Correlations with Energy Demand:")
    for feature, corr in correlation_with_target.sort_values(ascending=False).items():
        if feature != 'Energy_Demand':
            print(f"   {feature}: {corr:.3f}")
    
    return correlation_with_target

def main():
    """Main test function"""
    print("🧪 ML APPLICATION - QUICK TEST")
    print("=" * 40)
    
    # Test 1: Data Loading
    success, data = test_data_loading()
    if not success:
        print("❌ Cannot proceed without data. Please check the file path.")
        return
    
    # Test 2: Basic Statistics
    print(f"\n📊 BASIC STATISTICS")
    print(f"   Total records: {len(data):,}")
    print(f"   Missing values: {data.isnull().sum().sum()}")
    print(f"   Duplicate rows: {data.duplicated().sum()}")
    
    # Test 3: Visualizations
    test_basic_visualization(data)
    
    # Test 4: Correlation Analysis
    correlations = test_correlation_analysis(data)
    
    print(f"\n✅ QUICK TEST COMPLETED!")
    print(f"📋 Data is ready for ML training")
    print(f"🚀 Run the full ML.py script to train models")

if __name__ == "__main__":
    main()
