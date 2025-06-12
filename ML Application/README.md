# Machine Learning Application - Energy Forecasting

## Overview
This ML application implements **Part 2** of the PBL Project: Machine Learning Implementation for predicting household energy consumption based on weather patterns using Recurrent Neural Networks (RNN/LSTM).

## Project Requirements Implemented

### ✅ Dataset Preparation
- Loads integrated energy dataset from Session 5
- Splits data into training and testing sets
- Implements proper data preprocessing and scaling

### ✅ RNN Model Development
- **Single Factor Analysis**: Temperature only
- **Single Factor Analysis**: Weather conditions only  
- **Two Factor Analysis**: Temperature + Weather conditions
- **Multi-Factor Analysis**: All available features

### ✅ Model Performance Evaluation
- **Error Metrics**: MAE (Mean Absolute Error), RMSE (Root Mean Square Error)
- **Accuracy Measures**: R² Score for model comparison
- **Visual Comparison**: Predicted vs Actual results with pattern analysis

### ✅ Comprehensive Analysis
- Performance comparison table for all models
- Feature importance analysis
- Training history visualization
- Residual analysis for model validation

### ✅ Optimization Recommendations
- Energy usage optimization techniques
- Load scheduling recommendations
- Renewable energy integration strategies

## Files Structure

```
ML Application/
├── ML.py                    # Main ML application (Complete implementation)
├── test_ml_setup.py         # Quick test script to verify setup
├── requirements.txt         # Required Python packages
└── README.md               # This file
```

## Installation & Setup

### 1. Install Required Packages
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### 2. Verify Setup
Run the test script first to ensure everything works:
```bash
python test_ml_setup.py
```

### 3. Run Full ML Analysis
Execute the complete ML pipeline:
```bash
python ML.py
```

## Expected Outputs

### 📊 Performance Summary Table
- Comparison of all model types
- MAE, RMSE, and R² scores for each model
- Factor count and primary features used

### 📈 Visualizations (`energy_forecasting_results.png`)
1. **Model Performance Comparison**: R², MAE, RMSE charts
2. **Prediction vs Actual**: Time series comparison for each model
3. **Training History**: Loss curves during model training
4. **Residual Analysis**: Error pattern analysis
5. **Feature Importance**: Correlation analysis with target variable

### 📋 CSV Export (`model_performance_summary.csv`)
- Detailed performance metrics for all models
- Ready for report inclusion and further analysis

### 💡 Insights and Recommendations
- Key findings about energy consumption patterns
- Optimization strategies for energy management
- Peak hour identification and load scheduling advice

## Model Architecture

### Single/Two Factor Models
```
LSTM(50) → Dropout(0.2) → LSTM(50) → Dropout(0.2) → Dense(25) → Dense(1)
```

### Multi-Factor Model
```
LSTM(100) → Dropout(0.3) → LSTM(50) → Dropout(0.3) → LSTM(25) → Dropout(0.2) 
→ Dense(50) → Dense(25) → Dense(1)
```

## Key Features

### 🔄 Data Preprocessing
- Timestamp conversion and time-based feature extraction
- Weather condition encoding (categorical → numerical)
- Missing value handling and data cleaning
- Sequence creation for LSTM training (24-hour windows)

### 🧠 Model Training
- Early stopping to prevent overfitting
- Validation split for model tuning
- Feature scaling for optimal performance
- Multiple model comparison framework

### 📊 Comprehensive Evaluation
- Multiple error metrics (MAE, RMSE, R²)
- Visual comparison of predictions vs actual values
- Training history analysis
- Residual plotting for model validation

### ⚡ Optimization Insights
- Peak demand hour identification
- Feature correlation analysis
- Energy usage optimization recommendations
- Renewable energy integration strategies

## Usage Example

```python
# Initialize the forecasting model
forecaster = EnergyForecastingModel()

# Load and prepare data
forecaster.load_and_prepare_data("path/to/energy_dataset.csv")

# Train different model types
forecaster.train_single_factor_model('Temperature')
forecaster.train_two_factor_model('Temperature', 'Weather_Condition_x_encoded')
forecaster.train_multi_factor_model()

# Generate performance summary and visualizations
forecaster.create_performance_summary()
forecaster.visualize_results()
forecaster.generate_insights_and_recommendations()
```

## Assessment Criteria Addressed

### ✅ Dataset Preparation
- ✅ Training/testing split implemented
- ✅ RNN model development and training
- ✅ Model validation with testing data

### ✅ Model Performance Evaluation
- ✅ MAE and RMSE error handling
- ✅ Visual comparison of predicted vs actual results
- ✅ Pattern discrepancy analysis

### ✅ Single Factor Analysis
- ✅ Temperature factor analysis
- ✅ Weather condition factor analysis
- ✅ Performance recording and comparison

### ✅ Two Factor Analysis
- ✅ Combined temperature and weather analysis
- ✅ Performance comparison with single-factor models
- ✅ Multi-factor effect examination

### ✅ Model Performance Summary
- ✅ Comprehensive performance table (DataFrame)
- ✅ Analysis of findings and insights
- ✅ Energy optimization techniques based on predictions

## Troubleshooting

### Common Issues:
1. **TensorFlow Installation**: Ensure you have Python 3.8-3.11
2. **Memory Issues**: Reduce batch size or sequence length if needed
3. **Data Path**: Verify the correct path to the energy dataset
4. **Missing Packages**: Install all requirements from requirements.txt

### Performance Tips:
- Run `test_ml_setup.py` first to verify data accessibility
- Use GPU acceleration if available (CUDA-enabled TensorFlow)
- Adjust sequence length based on available memory
- Monitor training progress and stop early if overfitting occurs

## Expected Runtime
- **Quick Test**: ~30 seconds
- **Full ML Pipeline**: ~5-10 minutes (depending on hardware)
- **Visualization Generation**: ~1-2 minutes

---

🎯 **This implementation fully satisfies Part 2 requirements of the PBL Project and provides comprehensive analysis for energy consumption forecasting based on weather patterns.**
