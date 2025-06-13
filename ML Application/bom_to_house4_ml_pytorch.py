# PyTorch ML Application - BOM Weather to House 4 Energy Prediction
# Converted from TensorFlow version for better GPU support
# Uses temperature from BOM data to predict energy consumption in House 4

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
import math
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# ====================
# CUDA CONFIGURATION
# ====================
def configure_cuda():
    """Configure CUDA for optimal GPU performance"""
    print("🚀 CUDA Configuration Starting...")
    
    # Check PyTorch CUDA support
    cuda_available = torch.cuda.is_available()
    print(f"   • PyTorch CUDA available: {cuda_available}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"   • Found {device_count} GPU(s):")
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            print(f"     GPU {i}: {device_name}")
            
        # Set the device
        device = torch.device("cuda:0")
        print(f"   ✅ Primary GPU set: {torch.cuda.get_device_name(0)}")
        
        # Print CUDA version
        print(f"   • CUDA Version: {torch.version.cuda}")
        
        # Print memory info
        print(f"   • GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        return device
    else:
        print("   ❌ No GPU available. Using CPU.")
        return torch.device("cpu")

# Define the device globally
device = configure_cuda()

# ====================
# DATA LOADING
# ====================
def load_and_preprocess_data(bom_path, house_path):
    """Load and preprocess BOM weather and House energy data"""
    print("\n📊 Loading and preprocessing data...")
    
    # Load BOM weather data
    print(f"   • Loading BOM data from: {bom_path}")
    try:
        # Read BOM data with proper parsing of dates
        bom_data = pd.read_csv(bom_path, parse_dates=['Date'])
        print(f"   ✅ BOM data loaded with {bom_data.shape[0]} rows and {bom_data.shape[1]} columns")
    except Exception as e:
        print(f"   ❌ Error loading BOM data: {e}")
        raise
    
    # Load House energy data
    print(f"   • Loading House data from: {house_path}")
    try:
        # The house data has no header and contains timestamp and energy consumption
        house_data = pd.read_csv(house_path, header=None, names=['Timestamp', 'Energy'])
        
        # Convert timestamps to datetime
        house_data['Timestamp'] = pd.to_datetime(house_data['Timestamp'], format='%d-%m-%y %H:%M', errors='coerce')
        
        # Add date column for merging with BOM data
        house_data['Date'] = house_data['Timestamp'].dt.date
        
        print(f"   ✅ House data loaded with {house_data.shape[0]} rows and {house_data.shape[1]} columns")
    except Exception as e:
        print(f"   ❌ Error loading House data: {e}")
        raise
    
    # Preprocess BOM data
    print("   • Preprocessing BOM data...")
    
    # Rename columns for easier access
    bom_data = bom_data.rename(columns={
        'MinimumTemperature__C_': 'min_temp',
        'MaximumTemperature__C_': 'max_temp',
        'x9amTemperature__C_': 'temp_9am',
        'x3pmTemperature__C_': 'temp_3pm',
        'Rainfall_mm_': 'rainfall'
    })
    
    # Extract date as string for later merging
    bom_data['DateStr'] = bom_data['Date'].dt.date
    
    # Preprocess House data
    print("   • Preprocessing House data...")
    
    # Aggregate energy consumption by date (sum for each day)
    daily_energy = house_data.groupby('Date')['Energy'].sum().reset_index()
    daily_energy['DateStr'] = daily_energy['Date']
    
    # Merge BOM and House data on date
    print("   • Merging datasets...")
    merged_data = pd.merge(bom_data, daily_energy, left_on='DateStr', right_on='DateStr', how='inner')
    
    print(f"   ✅ Merged data has {merged_data.shape[0]} rows")
    
    # Display sample data
    print("\n   • BOM data sample:")
    print(bom_data[['Date', 'min_temp', 'max_temp', 'temp_9am', 'temp_3pm']].head(3))
    print("\n   • House data sample (daily aggregated):")
    print(daily_energy.head(3))
    print("\n   • Merged data sample:")
    print(merged_data[['Date_x', 'min_temp', 'max_temp', 'temp_9am', 'temp_3pm', 'Energy']].head(3))
    
    print("   ✅ Data loaded and preprocessed successfully")
    
    return merged_data, merged_data

# ====================
# MODEL DEFINITION
# ====================
class EnergyPredictionModel(nn.Module):
    """PyTorch model for energy prediction based on weather data"""
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(EnergyPredictionModel, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)

# ====================
# TRAINING FUNCTIONS
# ====================
def train_model(model, train_loader, val_loader, epochs=50, learning_rate=0.001):
    """Train the PyTorch model"""
    print("\n🏋️ Training model...")
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                # Move data to device
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        # Calculate average validation loss
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"   • Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    print("   ✅ Model training completed")
    return train_losses, val_losses

# ====================
# EVALUATION FUNCTIONS
# ====================
def evaluate_model(model, test_loader):
    """Evaluate the trained model on test data"""
    print("\n📏 Evaluating model...")
    
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0.0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            # Store predictions and actual values
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())
    
    # Calculate average test loss
    test_loss /= len(test_loader)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    
    print(f"   • Test Loss: {test_loss:.4f}")
    print(f"   • RMSE: {rmse:.4f}")
    
    return test_loss, predictions, actuals

# ====================
# PLOTTING FUNCTIONS
# ====================
def plot_results(train_losses, val_losses, predictions, actuals):
    """Plot training history and prediction results"""
    print("\n📈 Plotting results...")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot training history
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot predictions vs actuals
    ax2.scatter(actuals, predictions, alpha=0.5)
    ax2.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--')
    ax2.set_title('Predictions vs Actuals')
    ax2.set_xlabel('Actual Energy Consumption')
    ax2.set_ylabel('Predicted Energy Consumption')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('energy_prediction_results.png')
    plt.show()
    
    print("   ✅ Results plotted and saved")

# ====================
# MAIN FUNCTION
# ====================
def main():
    """Main function to run the energy prediction pipeline"""
    print("\n🔄 Starting Energy Prediction Pipeline...")
    
    # File paths
    bom_path = "Datasets/BOM_year.csv"
    house_path = "Datasets/House 4_Melb West.csv"
    
    # Load and preprocess data
    merged_data, _ = load_and_preprocess_data(bom_path, house_path)
    
    # Convert data to tensors and prepare for PyTorch
    print("\n🔧 Preparing data for model...")
    
    # Extract temperature features from merged data
    try:
        # Use the four temperature columns as features
        features = merged_data[['min_temp', 'max_temp', 'temp_9am', 'temp_3pm']].values
        target = merged_data['Energy'].values
        
        print(f"   • Using temperature features: min_temp, max_temp, temp_9am, temp_3pm")
        print(f"   • Feature shape: {features.shape}")
        print(f"   • Target shape: {target.shape}")
    except Exception as e:
        print(f"   ❌ Error extracting features: {e}")
        raise
    
    # Split data into train, validation, and test sets
    data_size = len(features)
    train_size = int(0.7 * data_size)
    val_size = int(0.15 * data_size)
    
    # Create random permutation for shuffling
    indices = np.random.permutation(data_size)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    # Create tensor datasets
    X_train = torch.tensor(features[train_indices], dtype=torch.float32)
    y_train = torch.tensor(target[train_indices].reshape(-1, 1), dtype=torch.float32)
    
    X_val = torch.tensor(features[val_indices], dtype=torch.float32)
    y_val = torch.tensor(target[val_indices].reshape(-1, 1), dtype=torch.float32)
    
    X_test = torch.tensor(features[test_indices], dtype=torch.float32)
    y_test = torch.tensor(target[test_indices].reshape(-1, 1), dtype=torch.float32)
    
    print(f"   • Training samples: {len(X_train)}")
    print(f"   • Validation samples: {len(X_val)}")
    print(f"   • Test samples: {len(X_test)}")
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # Create model
    input_dim = X_train.shape[1]  # Number of temperature features (4)
    model = EnergyPredictionModel(input_dim=input_dim, hidden_dim=64, output_dim=1)
    model = model.to(device)
    print(f"   • Model created with {input_dim} input features")
    
    # Train model
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=100)
    
    # Evaluate model
    test_loss, predictions, actuals = evaluate_model(model, test_loader)
    
    # Plot results
    plot_results(train_losses, val_losses, predictions, actuals)
    
    # Save model
    torch.save(model.state_dict(), 'energy_prediction_model.pth')
    print("   ✅ Model saved to energy_prediction_model.pth")
    
    print("\n✅ Energy Prediction Pipeline completed successfully")

if __name__ == "__main__":
    main()
