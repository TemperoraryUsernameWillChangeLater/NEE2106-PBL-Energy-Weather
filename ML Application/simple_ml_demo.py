# Simple ML Demo - Works with basic packages only
# This version demonstrates the concept without requiring TensorFlow

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns

class SimpleEnergyForecastingDemo:
    """
    Simplified Energy Forecasting Demo using basic ML models
    This version works with standard packages and demonstrates the concept
    """
    
    def __init__(self):
        self.data = None
        self.models = {}
        self.results = []
        
    def load_and_prepare_data(self, data_path):
        """Load and prepare the energy dataset"""
        print("🔄 Loading Energy Dataset...")
        
        try:
            self.data = pd.read_csv(data_path)
            print(f"✅ Data loaded successfully: {self.data.shape[0]} records, {self.data.shape[1]} features")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
            
        # Display basic info
        print("\n📊 Dataset Overview:")
        print(self.data.head())
        print(f"\nColumns: {list(self.data.columns)}")
        
        # Data preprocessing
        print("\n🔧 Preprocessing data...")
        self.preprocess_data()
        
        return True
    
    def preprocess_data(self):
        """Preprocess the data for ML model training"""
        # Convert timestamp to datetime
        self.data['Timestamp'] = pd.to_datetime(self.data['Timestamp'])
        
        # Create time-based features
        self.data['Hour'] = self.data['Timestamp'].dt.hour
        self.data['DayOfWeek'] = self.data['Timestamp'].dt.dayofweek
        self.data['Month'] = self.data['Timestamp'].dt.month
        
        # Handle weather conditions (encode categorical data)
        le_weather_x = LabelEncoder()
        le_weather_y = LabelEncoder()
        
        self.data['Weather_Condition_x_encoded'] = le_weather_x.fit_transform(self.data['Weather_Condition_x'].fillna('Unknown'))
        self.data['Weather_Condition_y_encoded'] = le_weather_y.fit_transform(self.data['Weather_Condition_y'].fillna('Unknown'))
        
        # Fill missing values
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_columns] = self.data[numeric_columns].fillna(self.data[numeric_columns].mean())
        
        print(f"✅ Data preprocessed: {self.data.shape[0]} records ready for modeling")
    
    def train_single_factor_model(self, factor='Temperature'):
        """Train model with single weather factor"""
        print(f"\n🤖 Training Single Factor Model: {factor}")
        
        # Prepare features
        feature_cols = [factor, 'Hour', 'DayOfWeek', 'Month']
        X = self.data[feature_cols]
        y = self.data['Energy_Demand']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        best_model = None
        best_score = -float('inf')
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            print(f"   {name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}")
            
            if r2 > best_score:
                best_score = r2
                best_model = model
                best_predictions = y_pred
        
        # Store results
        result = {
            'Model': f'Single Factor - {factor}',
            'Factors': [factor],
            'MAE': mean_absolute_error(y_test, best_predictions),
            'RMSE': np.sqrt(mean_squared_error(y_test, best_predictions)),
            'R²': r2_score(y_test, best_predictions),
            'Predictions': best_predictions,
            'Actual': y_test.values
        }
        
        self.results.append(result)
        self.models[f'single_{factor.lower()}'] = best_model
        
        print(f"✅ Best model for {factor}: R²={best_score:.3f}")
        return result
    
    def train_two_factor_model(self, factor1='Temperature', factor2='Weather_Condition_x_encoded'):
        """Train model with two weather factors"""
        print(f"\n🤖 Training Two Factor Model: {factor1} + {factor2}")
        
        # Prepare features
        feature_cols = [factor1, factor2, 'Hour', 'DayOfWeek', 'Month']
        X = self.data[feature_cols]
        y = self.data['Energy_Demand']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest (usually performs well)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        result = {
            'Model': f'Two Factor - {factor1} + {factor2}',
            'Factors': [factor1, factor2],
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'Predictions': y_pred,
            'Actual': y_test.values
        }
        
        self.results.append(result)
        self.models[f'two_{factor1.lower()}_{factor2.lower()}'] = model
        
        print(f"✅ Two Factor Model: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}")
        return result
    
    def train_multi_factor_model(self):
        """Train model with multiple factors"""
        print(f"\n🤖 Training Multi-Factor Model")
        
        # Prepare features - use all relevant numerical features
        feature_cols = [
            'Temperature', 'Weather_Condition_x_encoded', 'Weather_Condition_y_encoded',
            'Energy_Supply', 'Grid_Load', 'Renewable_Source_Output', 
            'NonRenewable_Source_Output', 'Energy_Price',
            'Hour', 'DayOfWeek', 'Month'
        ]
        
        X = self.data[feature_cols]
        y = self.data['Energy_Demand']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest with more trees for complex model
        model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=15)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        result = {
            'Model': 'Multi-Factor (All Features)',
            'Factors': feature_cols,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'Predictions': y_pred,
            'Actual': y_test.values
        }
        
        self.results.append(result)
        self.models['multi_factor'] = model
        
        print(f"✅ Multi-Factor Model: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}")
        return result
    
    def create_performance_summary(self):
        """Create performance summary table"""
        print("\n📊 MODEL PERFORMANCE SUMMARY")
        print("=" * 80)
        
        summary_data = []
        for result in self.results:
            summary_data.append({
                'Model': result['Model'],
                'Factors Count': len(result['Factors']),
                'Primary Factors': ', '.join(result['Factors'][:2]) if len(result['Factors']) > 2 else ', '.join(result['Factors']),
                'MAE': round(result['MAE'], 2),
                'RMSE': round(result['RMSE'], 2),
                'R²': round(result['R²'], 3)
            })
        
        performance_df = pd.DataFrame(summary_data)
        performance_df = performance_df.sort_values('R²', ascending=False)
        
        print(performance_df.to_string(index=False))
        
        # Find best model
        best_model = performance_df.iloc[0]
        print(f"\n🏆 BEST PERFORMING MODEL:")
        print(f"   Model: {best_model['Model']}")
        print(f"   R² Score: {best_model['R²']}")
        print(f"   RMSE: {best_model['RMSE']}")
        print(f"   MAE: {best_model['MAE']}")
        
        return performance_df
    
    def visualize_results(self):
        """Create visualizations of model performance"""
        print("\n📈 Creating visualizations...")
        
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Model Performance Comparison
        plt.subplot(2, 3, 1)
        models = [result['Model'] for result in self.results]
        r2_scores = [result['R²'] for result in self.results]
        
        bars = plt.bar(range(len(models)), r2_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        plt.xlabel('Model')
        plt.ylabel('R² Score')
        plt.title('Model Performance Comparison (R² Score)')
        plt.xticks(range(len(models)), [m.split(' - ')[0] for m in models], rotation=45)
        
        for bar, score in zip(bars, r2_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 2. MAE Comparison
        plt.subplot(2, 3, 2)
        mae_scores = [result['MAE'] for result in self.results]
        plt.bar(range(len(models)), mae_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        plt.xlabel('Model')
        plt.ylabel('MAE')
        plt.title('Mean Absolute Error Comparison')
        plt.xticks(range(len(models)), [m.split(' - ')[0] for m in models], rotation=45)
        
        # 3. RMSE Comparison
        plt.subplot(2, 3, 3)
        rmse_scores = [result['RMSE'] for result in self.results]
        plt.bar(range(len(models)), rmse_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        plt.xlabel('Model')
        plt.ylabel('RMSE')
        plt.title('Root Mean Square Error Comparison')
        plt.xticks(range(len(models)), [m.split(' - ')[0] for m in models], rotation=45)
        
        # 4-6. Prediction vs Actual for each model
        for i, result in enumerate(self.results):
            plt.subplot(2, 3, 4 + i)
            actual = result['Actual'][:50]  # Show first 50 predictions
            predicted = result['Predictions'][:50]
            
            plt.plot(actual, label='Actual', alpha=0.7, linewidth=2)
            plt.plot(predicted, label='Predicted', alpha=0.7, linewidth=2)
            plt.xlabel('Time Steps')
            plt.ylabel('Energy Demand')
            plt.title(f'{result["Model"]}\nPrediction vs Actual')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('simple_ml_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations created and saved as 'simple_ml_results.png'")

def main():
    """Main execution function"""
    print("🚀 SIMPLE ENERGY FORECASTING ML DEMO")
    print("=" * 50)
    print("📝 Note: This is a simplified version using basic ML models")
    print("🔧 For full RNN/LSTM implementation, install TensorFlow and run ML.py")
    print("")
    
    # Initialize model
    forecaster = SimpleEnergyForecastingDemo()
    
    # Load data
    data_path = r"C:\Users\gabri\Documents\Python\(NEE2106) Computer Programming For Electrical Engineers\Session 5\Integrated Energy Management and Forecasting Dataset.csv"
    
    if not forecaster.load_and_prepare_data(data_path):
        print("❌ Failed to load data. Exiting...")
        return
    
    # Train models
    print("\n🎯 TRAINING PHASE - Multiple Model Analysis")
    print("-" * 50)
    
    # 1. Single Factor Analysis - Temperature
    forecaster.train_single_factor_model('Temperature')
    
    # 2. Two Factor Analysis - Temperature + Weather
    forecaster.train_two_factor_model('Temperature', 'Weather_Condition_x_encoded')
    
    # 3. Multi-Factor Analysis
    forecaster.train_multi_factor_model()
    
    # Performance Analysis
    print("\n📈 ANALYSIS PHASE")
    print("-" * 50)
    
    # Create performance summary
    performance_df = forecaster.create_performance_summary()
    
    # Generate visualizations
    forecaster.visualize_results()
    
    # Save results
    performance_df.to_csv('simple_ml_performance.csv', index=False)
    print(f"\n💾 Performance summary saved as 'simple_ml_performance.csv'")
    
    print("\n✅ SIMPLE ML DEMO COMPLETE!")
    print("📊 Check 'simple_ml_results.png' for visualizations")
    print("🚀 For advanced RNN models, install TensorFlow and run: python ML.py")

if __name__ == "__main__":
    main()
