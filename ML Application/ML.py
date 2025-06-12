# PBL Project Part 2: Machine Learning Implementation
# Predictive Modelling of Household Energy Consumption Based on Weather Pattern

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class EnergyForecastingModel:
    """
    Comprehensive Energy Forecasting Model using RNN/LSTM
    Implements single-factor, two-factor, and multi-factor analysis
    """
    
    def __init__(self, data_path=None):
        """Initialize the forecasting model"""
        self.data_path = data_path
        self.data = None
        self.models = {}
        self.results = []
        self.scalers = {}
        
    def load_and_prepare_data(self, data_path=None):
        """Load and prepare the energy dataset"""
        if data_path:
            self.data_path = data_path
            
        print("🔄 Loading Energy Dataset...")
        
        # Load the integrated energy dataset
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"✅ Data loaded successfully: {self.data.shape[0]} records, {self.data.shape[1]} features")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
            
        # Display basic info
        print("\n📊 Dataset Overview:")
        print(self.data.head())
        print(f"\nColumns: {list(self.data.columns)}")
        print(f"\nData types:\n{self.data.dtypes}")
        print(f"\nMissing values:\n{self.data.isnull().sum()}")
        
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
        
        # Remove any remaining rows with missing target variable
        self.data = self.data.dropna(subset=['Energy_Demand'])
        
        print(f"✅ Data preprocessed: {self.data.shape[0]} records ready for modeling")
    
    def create_sequences(self, data, target_col, feature_cols, sequence_length=24):
        """Create sequences for LSTM training"""
        X, y = [], []
        
        for i in range(len(data) - sequence_length):
            # Features sequence
            X.append(data[feature_cols].iloc[i:(i + sequence_length)].values)
            # Target (next value)
            y.append(data[target_col].iloc[i + sequence_length])
            
        return np.array(X), np.array(y)
    
    def build_rnn_model(self, input_shape, model_name="Basic"):
        """Build RNN/LSTM model architecture"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_single_factor_model(self, factor='Temperature'):
        """Train RNN model with single weather factor"""
        print(f"\n🤖 Training Single Factor Model: {factor}")
        
        # Prepare features
        feature_cols = [factor, 'Hour', 'DayOfWeek', 'Month']
        target_col = 'Energy_Demand'
        
        # Create sequences
        X, y = self.create_sequences(self.data, target_col, feature_cols)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = self._scale_sequences(X_train, scaler, fit=True)
        X_test_scaled = self._scale_sequences(X_test, scaler, fit=False)
        
        # Scale target
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        # Build and train model
        model = self.build_rnn_model(X_train_scaled.shape[1:], f"Single_{factor}")
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = model.fit(
            X_train_scaled, y_train_scaled,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Make predictions
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        result = {
            'Model': f'Single Factor - {factor}',
            'Factors': [factor],
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'Training_History': history,
            'Predictions': y_pred,
            'Actual': y_test
        }
        
        self.results.append(result)
        self.models[f'single_{factor.lower()}'] = model
        self.scalers[f'single_{factor.lower()}'] = (scaler, target_scaler)
        
        print(f"✅ Single Factor Model ({factor}) trained successfully!")
        print(f"   MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")
        
        return result
    
    def train_two_factor_model(self, factor1='Temperature', factor2='Weather_Condition_x_encoded'):
        """Train RNN model with two weather factors"""
        print(f"\n🤖 Training Two Factor Model: {factor1} + {factor2}")
        
        # Prepare features
        feature_cols = [factor1, factor2, 'Hour', 'DayOfWeek', 'Month']
        target_col = 'Energy_Demand'
        
        # Create sequences
        X, y = self.create_sequences(self.data, target_col, feature_cols)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = self._scale_sequences(X_train, scaler, fit=True)
        X_test_scaled = self._scale_sequences(X_test, scaler, fit=False)
        
        # Scale target
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        # Build and train model
        model = self.build_rnn_model(X_train_scaled.shape[1:], f"Two_{factor1}_{factor2}")
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = model.fit(
            X_train_scaled, y_train_scaled,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Make predictions
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
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
            'Training_History': history,
            'Predictions': y_pred,
            'Actual': y_test
        }
        
        self.results.append(result)
        self.models[f'two_{factor1.lower()}_{factor2.lower()}'] = model
        self.scalers[f'two_{factor1.lower()}_{factor2.lower()}'] = (scaler, target_scaler)
        
        print(f"✅ Two Factor Model ({factor1} + {factor2}) trained successfully!")
        print(f"   MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")
        
        return result
    
    def train_multi_factor_model(self):
        """Train RNN model with multiple factors"""
        print(f"\n🤖 Training Multi-Factor Model (All available features)")
        
        # Prepare features - use all relevant numerical features
        feature_cols = [
            'Temperature', 'Weather_Condition_x_encoded', 'Weather_Condition_y_encoded',
            'Energy_Supply', 'Grid_Load', 'Renewable_Source_Output', 
            'NonRenewable_Source_Output', 'Energy_Price',
            'Hour', 'DayOfWeek', 'Month'
        ]
        target_col = 'Energy_Demand'
        
        # Create sequences
        X, y = self.create_sequences(self.data, target_col, feature_cols)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = self._scale_sequences(X_train, scaler, fit=True)
        X_test_scaled = self._scale_sequences(X_test, scaler, fit=False)
        
        # Scale target
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        # Build and train model (larger architecture for multi-factor)
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=X_train_scaled.shape[1:]),
            Dropout(0.3),
            LSTM(50, return_sequences=True),
            Dropout(0.3),
            LSTM(25, return_sequences=False),
            Dropout(0.2),
            Dense(50),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        
        history = model.fit(
            X_train_scaled, y_train_scaled,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Make predictions
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
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
            'Training_History': history,
            'Predictions': y_pred,
            'Actual': y_test
        }
        
        self.results.append(result)
        self.models['multi_factor'] = model
        self.scalers['multi_factor'] = (scaler, target_scaler)
        
        print(f"✅ Multi-Factor Model trained successfully!")
        print(f"   MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")
        
        return result
    
    def _scale_sequences(self, sequences, scaler, fit=False):
        """Helper function to scale sequence data"""
        # Reshape for scaling
        n_samples, n_timesteps, n_features = sequences.shape
        sequences_reshaped = sequences.reshape(-1, n_features)
        
        # Scale
        if fit:
            sequences_scaled = scaler.fit_transform(sequences_reshaped)
        else:
            sequences_scaled = scaler.transform(sequences_reshaped)
        
        # Reshape back
        return sequences_scaled.reshape(n_samples, n_timesteps, n_features)
    
    def create_performance_summary(self):
        """Create comprehensive performance summary table"""
        print("\n📊 MODEL PERFORMANCE SUMMARY")
        print("=" * 80)
        
        # Create DataFrame with results
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
        
        self.performance_df = pd.DataFrame(summary_data)
        self.performance_df = self.performance_df.sort_values('R²', ascending=False)
        
        print(self.performance_df.to_string(index=False))
        
        # Find best model
        best_model = self.performance_df.iloc[0]
        print(f"\n🏆 BEST PERFORMING MODEL:")
        print(f"   Model: {best_model['Model']}")
        print(f"   R² Score: {best_model['R²']}")
        print(f"   RMSE: {best_model['RMSE']}")
        print(f"   MAE: {best_model['MAE']}")
        
        return self.performance_df
    
    def visualize_results(self):
        """Create comprehensive visualizations of model performance"""
        print("\n📈 Creating visualizations...")
        
        # Set up the plotting
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Model Performance Comparison
        plt.subplot(3, 3, 1)
        models = [result['Model'] for result in self.results]
        r2_scores = [result['R²'] for result in self.results]
        
        bars = plt.bar(range(len(models)), r2_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        plt.xlabel('Model')
        plt.ylabel('R² Score')
        plt.title('Model Performance Comparison (R² Score)')
        plt.xticks(range(len(models)), [m.split(' - ')[0] for m in models], rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, r2_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 2. MAE Comparison
        plt.subplot(3, 3, 2)
        mae_scores = [result['MAE'] for result in self.results]
        plt.bar(range(len(models)), mae_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        plt.xlabel('Model')
        plt.ylabel('MAE')
        plt.title('Mean Absolute Error Comparison')
        plt.xticks(range(len(models)), [m.split(' - ')[0] for m in models], rotation=45)
        
        # 3. RMSE Comparison
        plt.subplot(3, 3, 3)
        rmse_scores = [result['RMSE'] for result in self.results]
        plt.bar(range(len(models)), rmse_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        plt.xlabel('Model')
        plt.ylabel('RMSE')
        plt.title('Root Mean Square Error Comparison')
        plt.xticks(range(len(models)), [m.split(' - ')[0] for m in models], rotation=45)
        
        # 4-6. Prediction vs Actual for each model
        for i, result in enumerate(self.results[:3]):  # Show first 3 models
            plt.subplot(3, 3, 4 + i)
            actual = result['Actual'][:100]  # Show first 100 predictions
            predicted = result['Predictions'][:100]
            
            plt.plot(actual, label='Actual', alpha=0.7, linewidth=2)
            plt.plot(predicted, label='Predicted', alpha=0.7, linewidth=2)
            plt.xlabel('Time Steps')
            plt.ylabel('Energy Demand')
            plt.title(f'{result["Model"]}\nPrediction vs Actual')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 7. Training History for best model
        if self.results:
            best_result = max(self.results, key=lambda x: x['R²'])
            plt.subplot(3, 3, 7)
            history = best_result['Training_History']
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Training History - {best_result["Model"]}')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 8. Residual Analysis for best model
        if self.results:
            plt.subplot(3, 3, 8)
            best_result = max(self.results, key=lambda x: x['R²'])
            residuals = best_result['Actual'] - best_result['Predictions']
            plt.scatter(best_result['Predictions'], residuals, alpha=0.6, s=20)
            plt.axhline(y=0, color='red', linestyle='--')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title(f'Residual Plot - {best_result["Model"]}')
            plt.grid(True, alpha=0.3)
        
        # 9. Feature Importance (correlation with target)
        plt.subplot(3, 3, 9)
        correlations = self.data[['Temperature', 'Energy_Supply', 'Grid_Load', 
                                 'Renewable_Source_Output', 'Energy_Price']].corrwith(self.data['Energy_Demand'])
        correlations.abs().sort_values(ascending=True).plot(kind='barh', color='skyblue')
        plt.xlabel('Absolute Correlation with Energy Demand')
        plt.title('Feature Importance (Correlation)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('energy_forecasting_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations created and saved as 'energy_forecasting_results.png'")
    
    def generate_insights_and_recommendations(self):
        """Generate insights and optimization recommendations"""
        print("\n💡 INSIGHTS AND RECOMMENDATIONS")
        print("=" * 80)
        
        # Analyze feature correlations
        correlations = self.data[['Temperature', 'Energy_Supply', 'Grid_Load', 
                                 'Renewable_Source_Output', 'Energy_Price']].corrwith(self.data['Energy_Demand'])
        
        print("🔍 KEY INSIGHTS:")
        print(f"1. Strongest correlation with energy demand: {correlations.abs().idxmax()} ({correlations.abs().max():.3f})")
        print(f"2. Best performing model: {max(self.results, key=lambda x: x['R²'])['Model']}")
        print(f"3. Average energy demand: {self.data['Energy_Demand'].mean():.2f} units")
        print(f"4. Peak energy demand: {self.data['Energy_Demand'].max():.2f} units")
        
        # Time-based patterns
        hourly_avg = self.data.groupby('Hour')['Energy_Demand'].mean()
        peak_hour = hourly_avg.idxmax()
        low_hour = hourly_avg.idxmin()
        
        print(f"5. Peak demand hour: {peak_hour}:00 ({hourly_avg[peak_hour]:.2f} units)")
        print(f"6. Lowest demand hour: {low_hour}:00 ({hourly_avg[low_hour]:.2f} units)")
        
        print("\n⚡ OPTIMIZATION RECOMMENDATIONS:")
        print("1. 🌡️  Temperature Management: Monitor temperature forecasts for demand planning")
        print("2. 🔋 Energy Storage: Store excess renewable energy during low demand hours")
        print(f"3. ⏰ Load Scheduling: Schedule non-critical loads outside peak hour ({peak_hour}:00)")
        print("4. 📊 Real-time Monitoring: Implement the multi-factor model for continuous forecasting")
        print("5. 🌿 Renewable Integration: Increase renewable sources during predicted high demand")
        
        return correlations

def main():
    """Main execution function"""
    print("🚀 ENERGY FORECASTING ML APPLICATION")
    print("=" * 50)
    
    # Initialize model
    forecaster = EnergyForecastingModel()
    
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
    
    # 2. Single Factor Analysis - Weather Condition
    forecaster.train_single_factor_model('Weather_Condition_x_encoded')
    
    # 3. Two Factor Analysis - Temperature + Weather
    forecaster.train_two_factor_model('Temperature', 'Weather_Condition_x_encoded')
    
    # 4. Multi-Factor Analysis
    forecaster.train_multi_factor_model()
    
    # Performance Analysis
    print("\n📈 ANALYSIS PHASE")
    print("-" * 50)
    
    # Create performance summary
    performance_df = forecaster.create_performance_summary()
    
    # Generate visualizations
    forecaster.visualize_results()
    
    # Generate insights and recommendations
    forecaster.generate_insights_and_recommendations()
    
    # Save performance summary
    performance_df.to_csv('model_performance_summary.csv', index=False)
    print(f"\n💾 Performance summary saved as 'model_performance_summary.csv'")
    
    print("\n✅ ANALYSIS COMPLETE!")
    print("📊 Check 'energy_forecasting_results.png' for visualizations")
    print("📋 Check 'model_performance_summary.csv' for detailed metrics")

if __name__ == "__main__":
    main()