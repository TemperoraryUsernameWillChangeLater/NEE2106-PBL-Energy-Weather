# Plot Data from .dat Files
# This script loads and visualizes the preprocessed data from bom.dat and house4.dat
# Created from the main bom_to_house4_ml.py script

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import seaborn as sns

# Set up paths
script_dir = os.path.dirname(os.path.abspath(__file__))
refined_datasets_dir = os.path.join(script_dir, 'Refined Datasets')
bomfile = os.path.join(refined_datasets_dir, 'bom.dat')
house4file = os.path.join(refined_datasets_dir, 'house4.dat')

def load_dat_files():
    """Load data from .dat files"""
    print("Loading data from .dat files...")
    
    # Check if files exist
    if not os.path.exists(bomfile):
        print(f"❌ {bomfile} not found! Run the main script first to create it.")
        return None, None
    
    if not os.path.exists(house4file):
        print(f"❌ {house4file} not found! Run the main script first to create it.")
        return None, None
    
    # Load BOM weather data
    with open(bomfile, 'rb') as file:
        bom_data = pickle.load(file)
    print(f"✅ BOM data loaded: {len(bom_data)} records")
    
    # Load House 4 energy data
    with open(house4file, 'rb') as file:
        house4_data = pickle.load(file)
    print(f"✅ House 4 data loaded: {len(house4_data)} records")
    
    return bom_data, house4_data

def convert_to_dataframes(bom_data, house4_data):
    """Convert dictionary data to pandas DataFrames for easier plotting"""
    print("Converting to DataFrames...")
    
    # Convert BOM data
    bom_list = []
    for date_key, temps in bom_data.items():
        # Parse date key (format: YYYYMMDD)
        try:
            year = int(date_key[:4])
            month = int(date_key[4:6])
            day = int(date_key[6:8])
            date_obj = datetime(year, month, day)
            
            bom_list.append({
                'Date': date_obj,
                'DateKey': date_key,
                'MinTemp': temps[0],
                'MaxTemp': temps[1],
                'Temp9am': temps[2],
                'Temp3pm': temps[3],
                'AvgTemp': np.mean(temps)
            })
        except:
            continue
    
    bom_df = pd.DataFrame(bom_list).sort_values('Date')
    
    # Convert House 4 data
    house4_list = []
    for time_key, power in house4_data.items():
        # Parse time key (format: YYYYMMDDXX where XX is hour)
        try:
            date_part = time_key[:8]
            hour_part = time_key[8:]
            
            year = int(date_part[:4])
            month = int(date_part[4:6])
            day = int(date_part[6:8])
            hour = int(hour_part)
            
            timestamp = datetime(year, month, day, hour)
            
            house4_list.append({
                'DateTime': timestamp,
                'DateKey': date_part,
                'TimeKey': time_key,
                'Power_kW': power
            })
        except:
            continue
    
    house4_df = pd.DataFrame(house4_list).sort_values('DateTime')
    
    print(f"✅ BOM DataFrame: {len(bom_df)} records")
    print(f"✅ House 4 DataFrame: {len(house4_df)} records")
    
    return bom_df, house4_df

def plot_bom_weather_data(bom_df):
    """Plot BOM weather data"""
    print("Creating BOM weather plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('BOM Weather Data Analysis', fontsize=16, fontweight='bold')
    
    # Temperature trends over time
    axes[0, 0].plot(bom_df['Date'], bom_df['MinTemp'], 'b-', label='Min Temp', alpha=0.7)
    axes[0, 0].plot(bom_df['Date'], bom_df['MaxTemp'], 'r-', label='Max Temp', alpha=0.7)
    axes[0, 0].plot(bom_df['Date'], bom_df['AvgTemp'], 'g-', label='Avg Temp', linewidth=2)
    axes[0, 0].set_title('Temperature Trends Over Time')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Temperature (°C)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Temperature distribution
    axes[0, 1].hist(bom_df['MinTemp'], bins=30, alpha=0.7, label='Min Temp', color='blue')
    axes[0, 1].hist(bom_df['MaxTemp'], bins=30, alpha=0.7, label='Max Temp', color='red')
    axes[0, 1].set_title('Temperature Distribution')
    axes[0, 1].set_xlabel('Temperature (°C)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Daily temperature range
    bom_df['TempRange'] = bom_df['MaxTemp'] - bom_df['MinTemp']
    axes[1, 0].plot(bom_df['Date'], bom_df['TempRange'], 'purple', alpha=0.7)
    axes[1, 0].set_title('Daily Temperature Range')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Temperature Range (°C)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 9am vs 3pm temperatures
    axes[1, 1].scatter(bom_df['Temp9am'], bom_df['Temp3pm'], alpha=0.6, color='orange')
    axes[1, 1].plot([bom_df['Temp9am'].min(), bom_df['Temp9am'].max()], 
                    [bom_df['Temp9am'].min(), bom_df['Temp9am'].max()], 'k--', alpha=0.5)
    axes[1, 1].set_title('9am vs 3pm Temperature Correlation')
    axes[1, 1].set_xlabel('9am Temperature (°C)')
    axes[1, 1].set_ylabel('3pm Temperature (°C)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=False)

def plot_house4_energy_data(house4_df):
    """Plot House 4 energy consumption data"""
    print("Creating House 4 energy plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('House 4 Energy Consumption Analysis', fontsize=16, fontweight='bold')
    
    # Energy consumption over time
    axes[0, 0].plot(house4_df['DateTime'], house4_df['Power_kW'], 'b-', alpha=0.7, linewidth=0.5)
    axes[0, 0].set_title('Energy Consumption Over Time')
    axes[0, 0].set_xlabel('Date/Time')
    axes[0, 0].set_ylabel('Power (kW)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Power distribution
    axes[0, 1].hist(house4_df['Power_kW'], bins=50, alpha=0.7, color='green')
    axes[0, 1].set_title('Power Consumption Distribution')
    axes[0, 1].set_xlabel('Power (kW)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Daily average consumption
    house4_df['Date'] = house4_df['DateTime'].dt.date
    daily_avg = house4_df.groupby('Date')['Power_kW'].mean().reset_index()
    daily_avg['Date'] = pd.to_datetime(daily_avg['Date'])
    
    axes[1, 0].plot(daily_avg['Date'], daily_avg['Power_kW'], 'r-', linewidth=2)
    axes[1, 0].set_title('Daily Average Power Consumption')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Average Power (kW)')
    axes[1, 0].grid(True, alpha=0.3)
      # Hourly consumption pattern
    house4_df['Hour'] = house4_df['DateTime'].dt.hour
    hourly_avg = house4_df.groupby('Hour')['Power_kW'].mean()
    
    axes[1, 1].bar(hourly_avg.index, hourly_avg.values, alpha=0.7, color='orange')
    axes[1, 1].set_title('Average Hourly Power Consumption')
    axes[1, 1].set_xlabel('Hour of Day')
    axes[1, 1].set_ylabel('Average Power (kW)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=False)

def plot_correlation_analysis(bom_df, house4_df):
    """Plot correlation between weather and energy data"""
    print("Creating correlation analysis...")
    
    # Merge data on date
    bom_df['DateKey'] = bom_df['DateKey'].astype(str)
    house4_df['DateKey'] = house4_df['DateKey'].astype(str)
    
    # Get daily average power consumption
    daily_power = house4_df.groupby('DateKey')['Power_kW'].mean().reset_index()
    daily_power.columns = ['DateKey', 'DailyAvgPower']
    
    # Merge with weather data
    merged_df = pd.merge(bom_df, daily_power, on='DateKey', how='inner')
    
    if len(merged_df) == 0:
        print("❌ No matching dates found between weather and energy data")
        return
    
    print(f"✅ Found {len(merged_df)} matching date records")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Weather vs Energy Consumption Correlation Analysis', fontsize=16, fontweight='bold')
    
    # Temperature vs Power scatter plots
    axes[0, 0].scatter(merged_df['MinTemp'], merged_df['DailyAvgPower'], alpha=0.6, color='blue')
    axes[0, 0].set_title('Min Temperature vs Daily Average Power')
    axes[0, 0].set_xlabel('Min Temperature (°C)')
    axes[0, 0].set_ylabel('Daily Avg Power (kW)')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].scatter(merged_df['MaxTemp'], merged_df['DailyAvgPower'], alpha=0.6, color='red')
    axes[0, 1].set_title('Max Temperature vs Daily Average Power')
    axes[0, 1].set_xlabel('Max Temperature (°C)')
    axes[0, 1].set_ylabel('Daily Avg Power (kW)')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].scatter(merged_df['AvgTemp'], merged_df['DailyAvgPower'], alpha=0.6, color='green')
    axes[1, 0].set_title('Average Temperature vs Daily Average Power')
    axes[1, 0].set_xlabel('Average Temperature (°C)')
    axes[1, 0].set_ylabel('Daily Avg Power (kW)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Correlation heatmap
    corr_data = merged_df[['MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm', 'AvgTemp', 'DailyAvgPower']].corr()
    im = axes[1, 1].imshow(corr_data.values, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 1].set_xticks(range(len(corr_data.columns)))
    axes[1, 1].set_yticks(range(len(corr_data.columns)))
    axes[1, 1].set_xticklabels(corr_data.columns, rotation=45)
    axes[1, 1].set_yticklabels(corr_data.columns)
    axes[1, 1].set_title('Correlation Matrix')
      # Add correlation values to heatmap
    for i in range(len(corr_data.columns)):
        for j in range(len(corr_data.columns)):
            axes[1, 1].text(j, i, f'{corr_data.iloc[i, j]:.2f}', 
                           ha='center', va='center', color='black', fontsize=8)
    
    plt.colorbar(im, ax=axes[1, 1])
    plt.tight_layout()
    plt.show(block=False)

def print_data_summary(bom_df, house4_df):
    """Print summary statistics of the data"""
    print("\n" + "="*60)
    print("📊 DATA SUMMARY FROM .DAT FILES")
    print("="*60)
    
    print("\n🌡️  BOM WEATHER DATA:")
    print(f"   • Records: {len(bom_df)}")
    print(f"   • Date range: {bom_df['Date'].min().strftime('%Y-%m-%d')} to {bom_df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"   • Min Temperature: {bom_df['MinTemp'].min():.1f}°C to {bom_df['MinTemp'].max():.1f}°C")
    print(f"   • Max Temperature: {bom_df['MaxTemp'].min():.1f}°C to {bom_df['MaxTemp'].max():.1f}°C")
    print(f"   • Average Temperature: {bom_df['AvgTemp'].mean():.1f}°C")
    
    print("\n⚡ HOUSE 4 ENERGY DATA:")
    print(f"   • Records: {len(house4_df)}")
    print(f"   • Date range: {house4_df['DateTime'].min().strftime('%Y-%m-%d %H:%M')} to {house4_df['DateTime'].max().strftime('%Y-%m-%d %H:%M')}")
    print(f"   • Power range: {house4_df['Power_kW'].min():.3f} kW to {house4_df['Power_kW'].max():.3f} kW")
    print(f"   • Average power: {house4_df['Power_kW'].mean():.3f} kW")
    print(f"   • Total energy: {house4_df['Power_kW'].sum():.1f} kWh")

def main():
    """Main function to run all plotting functions"""
    print("🔍 DAT FILE VISUALIZATION TOOL")
    print("="*50)
    
    # Load data from .dat files
    bom_data, house4_data = load_dat_files()
    
    if bom_data is None or house4_data is None:
        print("❌ Cannot proceed without data files. Run bom_to_house4_ml.py first!")
        return
    
    # Convert to DataFrames
    bom_df, house4_df = convert_to_dataframes(bom_data, house4_data)
      # Print summary
    print_data_summary(bom_df, house4_df)
    
    print("\n📈 Creating visualizations...")
    
    # Create plots
    plot_bom_weather_data(bom_df)
    plot_house4_energy_data(house4_df)
    plot_correlation_analysis(bom_df, house4_df)
    
    # Try to load and display CSV files if they exist
    plot_csv_results()
    
    print("\n✅ All visualizations completed!")
    print("📁 All files are saved in the Refined Datasets directory:")
    print("   • bom.dat / house4.dat (processed data)")
    print("   • incremental_epoch_results.csv (if available)")
    print("   • epoch_differences_results.csv (if available)")
    
    # Keep all plot windows open
    print("\n📊 All plots are now displayed simultaneously!")
    print("💡 Close any plot window or press Enter to exit...")
    input()  # Wait for user input before closing all plots

def plot_csv_results():
    """Plot results from CSV files if they exist"""
    epoch_results_file = os.path.join(refined_datasets_dir, 'incremental_epoch_results.csv')
    differences_file = os.path.join(refined_datasets_dir, 'epoch_differences_results.csv')
    
    if os.path.exists(epoch_results_file):
        print("📊 Loading epoch results from CSV...")
        try:
            epoch_df = pd.read_csv(epoch_results_file)
            plot_epoch_results(epoch_df)
        except Exception as e:
            print(f"❌ Error loading epoch results: {e}")
    
    if os.path.exists(differences_file):
        print("📊 Loading epoch differences from CSV...")
        try:
            diff_df = pd.read_csv(differences_file)
            plot_epoch_differences(diff_df)
        except Exception as e:
            print(f"❌ Error loading epoch differences: {e}")

def plot_epoch_results(epoch_df):
    """Plot epoch training results"""
    unique_epochs = sorted(epoch_df['Epoch'].unique())
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('ML Training Results - Epoch Progression', fontsize=16, fontweight='bold')
    
    # MSE progression
    epoch_mse = epoch_df.groupby('Epoch')['MSE'].first()
    axes[0, 0].plot(epoch_mse.index, epoch_mse.values, 'bo-', linewidth=2, markersize=6)
    axes[0, 0].set_title('MSE vs Epochs')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Mean Squared Error')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Error rate progression
    epoch_error_rate = epoch_df.groupby('Epoch')['Error_Rate_%'].first()
    axes[0, 1].plot(epoch_error_rate.index, epoch_error_rate.values, 'ro-', linewidth=2, markersize=6)
    axes[0, 1].set_title('Error Rate vs Epochs')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Error Rate (%)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Prediction accuracy scatter for final epoch
    final_epoch_data = epoch_df[epoch_df['Epoch'] == unique_epochs[-1]]
    axes[0, 2].scatter(final_epoch_data['Actual_kW'], final_epoch_data['Predicted_kW'], alpha=0.6)
    axes[0, 2].plot([final_epoch_data['Actual_kW'].min(), final_epoch_data['Actual_kW'].max()],
                    [final_epoch_data['Actual_kW'].min(), final_epoch_data['Actual_kW'].max()], 'r--', alpha=0.8)
    axes[0, 2].set_title(f'Actual vs Predicted (Epoch {unique_epochs[-1]})')
    axes[0, 2].set_xlabel('Actual Power (kW)')
    axes[0, 2].set_ylabel('Predicted Power (kW)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Error distribution for final epoch
    axes[1, 0].hist(final_epoch_data['Error_kW'], bins=30, alpha=0.7, color='purple')
    axes[1, 0].set_title(f'Error Distribution (Epoch {unique_epochs[-1]})')
    axes[1, 0].set_xlabel('Prediction Error (kW)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Average error per epoch
    avg_abs_error = epoch_df.groupby('Epoch')['Error_kW'].apply(lambda x: np.abs(x).mean())
    axes[1, 1].plot(avg_abs_error.index, avg_abs_error.values, 'go-', linewidth=2, markersize=6)
    axes[1, 1].set_title('Average Absolute Error vs Epochs')
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].set_ylabel('Average Absolute Error (kW)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Prediction spread by epoch
    epoch_std = epoch_df.groupby('Epoch')['Predicted_kW'].std()
    axes[1, 2].plot(epoch_std.index, epoch_std.values, 'mo-', linewidth=2, markersize=6)
    axes[1, 2].set_title('Prediction Variability vs Epochs')
    axes[1, 2].set_xlabel('Epochs')
    axes[1, 2].set_ylabel('Prediction Std Deviation (kW)')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=False)

def plot_epoch_differences(diff_df):
    """Plot epoch difference analysis"""
    unique_transitions = diff_df[['From_Epoch', 'To_Epoch']].drop_duplicates()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Epoch-to-Epoch Prediction Changes Analysis', fontsize=16, fontweight='bold')
    
    # Mean differences across epoch transitions
    mean_diffs = diff_df.groupby(['From_Epoch', 'To_Epoch'])['Mean_Difference'].first()
    transition_labels = [f"{int(idx[0])}-{int(idx[1])}" for idx in mean_diffs.index]
    
    axes[0, 0].bar(range(len(mean_diffs)), mean_diffs.values, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Mean Prediction Changes Between Epochs')
    axes[0, 0].set_xlabel('Epoch Transitions')
    axes[0, 0].set_ylabel('Mean Difference (kW)')
    axes[0, 0].set_xticks(range(len(transition_labels)))
    axes[0, 0].set_xticklabels(transition_labels, rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Standard deviation of differences
    std_diffs = diff_df.groupby(['From_Epoch', 'To_Epoch'])['Std_Difference'].first()
    axes[0, 1].bar(range(len(std_diffs)), std_diffs.values, alpha=0.7, color='lightcoral')
    axes[0, 1].set_title('Prediction Change Variability Between Epochs')
    axes[0, 1].set_xlabel('Epoch Transitions')
    axes[0, 1].set_ylabel('Std Deviation (kW)')
    axes[0, 1].set_xticks(range(len(transition_labels)))
    axes[0, 1].set_xticklabels(transition_labels, rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Distribution of all prediction differences
    axes[1, 0].hist(diff_df['Prediction_Difference'], bins=50, alpha=0.7, color='lightgreen')
    axes[1, 0].set_title('Distribution of All Prediction Changes')
    axes[1, 0].set_xlabel('Prediction Difference (kW)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Box plot of differences by transition
    transition_data = []
    labels = []
    for _, group in diff_df.groupby(['From_Epoch', 'To_Epoch']):
        transition_data.append(group['Prediction_Difference'].values)
        labels.append(f"{int(group['From_Epoch'].iloc[0])}-{int(group['To_Epoch'].iloc[0])}")
    
    axes[1, 1].boxplot(transition_data, labels=labels)
    axes[1, 1].set_title('Prediction Change Distribution by Transition')
    axes[1, 1].set_xlabel('Epoch Transitions')
    axes[1, 1].set_ylabel('Prediction Difference (kW)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=False)

if __name__ == "__main__":
    main()
