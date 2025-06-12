# ML Application: Energy Demand and Supply Prediction using RNN
# Based on PBL Project - Energy and Weather Part 2
# Predicts Energy Demand and Supply using Temperature and Weather Condition

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns

# Load dataset
csv_path = r"c:\Users\gabri\Documents\Python\(NEE2106) Computer Programming For Electrical Engineers\PBL Project - Energy and Weather\ML Application\Integrated Energy Management and Forecasting Dataset.csv"
data = pd.read_csv(csv_path)

print("Dataset Info:")
print(f"Shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")
print("\nFirst few rows:")
print(data.head())

# Data preprocessing
# Use Weather_Condition_x as the primary weather condition
data_clean = data.dropna()

# Encode weather conditions
le = LabelEncoder()
data_clean['Weather_Encoded'] = le.fit_transform(data_clean['Weather_Condition_x'])

print(f"\nWeather conditions mapping:")
for i, condition in enumerate(le.classes_):
    print(f"{condition}: {i}")

# Prepare features and targets
features = ['Temperature', 'Weather_Encoded']
targets = ['Energy_Demand', 'Energy_Supply']

X = data_clean[features].values
y = data_clean[targets].values

# Scale features
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Reshape for RNN (samples, timesteps, features)
# For this case, we'll use timesteps=1 since we don't have sequential data
X_train_rnn = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_rnn = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

print(f"\nData shapes:")
print(f"X_train_rnn: {X_train_rnn.shape}")
print(f"X_test_rnn: {X_test_rnn.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")

# Build RNN Model
def create_rnn_model(input_shape, output_dim):
    model = keras.Sequential([
        layers.SimpleRNN(64, activation='relu', input_shape=input_shape, return_sequences=True),
        layers.Dropout(0.2),
        layers.SimpleRNN(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(output_dim)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Create and train the model
model = create_rnn_model((1, 2), 2)  # (timesteps, features), output_dim
print("\nModel Architecture:")
model.summary()

# Train model
print("\nTraining RNN model...")
history = model.fit(
    X_train_rnn, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Make predictions
y_pred_scaled = model.predict(X_test_rnn)

# Inverse transform predictions and actual values
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test)

# Calculate performance metrics
mse = mean_squared_error(y_test_actual, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_actual, y_pred)
r2 = r2_score(y_test_actual, y_pred)

print(f"\n=== RNN Model Performance (Two Factors: Temperature + Weather) ===")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# Individual metrics for Energy Demand and Supply
demand_mse = mean_squared_error(y_test_actual[:, 0], y_pred[:, 0])
supply_mse = mean_squared_error(y_test_actual[:, 1], y_pred[:, 1])
demand_r2 = r2_score(y_test_actual[:, 0], y_pred[:, 0])
supply_r2 = r2_score(y_test_actual[:, 1], y_pred[:, 1])

print(f"\nEnergy Demand - MSE: {demand_mse:.2f}, R²: {demand_r2:.4f}")
print(f"Energy Supply - MSE: {supply_mse:.2f}, R²: {supply_r2:.4f}")

# Visualizations
plt.figure(figsize=(15, 10))

# Training history
plt.subplot(2, 3, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Energy Demand Predictions
plt.subplot(2, 3, 2)
X_test_original = scaler_X.inverse_transform(X_test)
plt.scatter(X_test_original[:, 0], y_test_actual[:, 0], color='blue', label='Actual Demand', alpha=0.6)
sorted_indices = np.argsort(X_test_original[:, 0])
plt.plot(X_test_original[sorted_indices, 0], y_pred[sorted_indices, 0], color='red', label='Predicted Demand', linewidth=2)
plt.xlabel('Temperature (°C)')
plt.ylabel('Energy Demand')
plt.title('Energy Demand Prediction')
plt.legend()

# Energy Supply Predictions
plt.subplot(2, 3, 3)
plt.scatter(X_test_original[:, 0], y_test_actual[:, 1], color='green', label='Actual Supply', alpha=0.6)
plt.plot(X_test_original[sorted_indices, 0], y_pred[sorted_indices, 1], color='orange', label='Predicted Supply', linewidth=2)
plt.xlabel('Temperature (°C)')
plt.ylabel('Energy Supply')
plt.title('Energy Supply Prediction')
plt.legend()

# Actual vs Predicted for Energy Demand
plt.subplot(2, 3, 4)
plt.scatter(y_test_actual[:, 0], y_pred[:, 0], alpha=0.6)
plt.plot([y_test_actual[:, 0].min(), y_test_actual[:, 0].max()], 
         [y_test_actual[:, 0].min(), y_test_actual[:, 0].max()], 'r--', lw=2)
plt.xlabel('Actual Energy Demand')
plt.ylabel('Predicted Energy Demand')
plt.title('Actual vs Predicted: Energy Demand')

# Actual vs Predicted for Energy Supply
plt.subplot(2, 3, 5)
plt.scatter(y_test_actual[:, 1], y_pred[:, 1], alpha=0.6)
plt.plot([y_test_actual[:, 1].min(), y_test_actual[:, 1].max()], 
         [y_test_actual[:, 1].min(), y_test_actual[:, 1].max()], 'r--', lw=2)
plt.xlabel('Actual Energy Supply')
plt.ylabel('Predicted Energy Supply')
plt.title('Actual vs Predicted: Energy Supply')

# Weather condition effect
plt.subplot(2, 3, 6)
weather_conditions = le.classes_
demand_by_weather = []
supply_by_weather = []

for i, condition in enumerate(weather_conditions):
    mask = X_test_original[:, 1] == i
    if np.any(mask):
        demand_by_weather.append(np.mean(y_test_actual[mask, 0]))
        supply_by_weather.append(np.mean(y_test_actual[mask, 1]))
    else:
        demand_by_weather.append(0)
        supply_by_weather.append(0)

x_pos = np.arange(len(weather_conditions))
width = 0.35

plt.bar(x_pos - width/2, demand_by_weather, width, label='Energy Demand', alpha=0.8)
plt.bar(x_pos + width/2, supply_by_weather, width, label='Energy Supply', alpha=0.8)
plt.xlabel('Weather Condition')
plt.ylabel('Average Energy')
plt.title('Energy by Weather Condition')
plt.xticks(x_pos, weather_conditions, rotation=45)
plt.legend()

plt.tight_layout()
plt.show()

# Save model performance summary
performance_summary = pd.DataFrame({
    'Model': ['RNN (Temperature + Weather)'],
    'Features': ['Temperature, Weather_Condition'],
    'MSE': [mse],
    'RMSE': [rmse],
    'MAE': [mae],
    'R2_Score': [r2],
    'Demand_MSE': [demand_mse],
    'Supply_MSE': [supply_mse],
    'Demand_R2': [demand_r2],
    'Supply_R2': [supply_r2]
})

print("\n=== Model Performance Summary ===")
print(performance_summary)

# Save to CSV
performance_summary.to_csv(r'c:\Users\gabri\Documents\Python\(NEE2106) Computer Programming For Electrical Engineers\model_performance_summary.csv', index=False)
print(f"\nPerformance summary saved to model_performance_summary.csv")

# Feature importance analysis
print(f"\n=== Feature Analysis ===")
print(f"Temperature range: {data_clean['Temperature'].min():.1f}°C to {data_clean['Temperature'].max():.1f}°C")
print(f"Weather conditions: {', '.join(le.classes_)}")

# Correlation analysis
correlation_matrix = data_clean[['Temperature', 'Weather_Encoded', 'Energy_Demand', 'Energy_Supply']].corr()
print(f"\nCorrelation Matrix:")
print(correlation_matrix)