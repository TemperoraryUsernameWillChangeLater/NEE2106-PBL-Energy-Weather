# Plot Data from .dat Files
# This script loads and visualizes the preprocessed data from bom.dat and house4.dat
# Created from the main bom_to_house4_ml.py script

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

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
    """Plot BOM weather data with A4-vertical layout"""
    print("Creating BOM weather plots...")
    
    # More vertical layout for A4 compatibility (3 rows, 2 columns)
    fig, axes = plt.subplots(3, 2, figsize=(8.5, 12))  # A4-like proportions
    fig.suptitle('BOM Weather Data Analysis', fontsize=14, fontweight='bold')
      # Temperature trends over time
    axes[0, 0].plot(bom_df['Date'], bom_df['MinTemp'], 'b-', label='Min Temp', alpha=0.7)
    axes[0, 0].plot(bom_df['Date'], bom_df['MaxTemp'], 'r-', label='Max Temp', alpha=0.7)
    axes[0, 0].plot(bom_df['Date'], bom_df['AvgTemp'], 'g-', label='Avg Temp', linewidth=2)
    axes[0, 0].set_title('Temperature Trends Over Time', fontsize=12)
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Temperature (°C)')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Temperature distribution
    axes[0, 1].hist(bom_df['MinTemp'], bins=30, alpha=0.7, label='Min Temp', color='blue')
    axes[0, 1].hist(bom_df['MaxTemp'], bins=30, alpha=0.7, label='Max Temp', color='red')
    axes[0, 1].set_title('Temperature Distribution', fontsize=12)
    axes[0, 1].set_xlabel('Temperature (°C)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
      # Daily temperature range
    bom_df['TempRange'] = bom_df['MaxTemp'] - bom_df['MinTemp']
    axes[1, 0].plot(bom_df['Date'], bom_df['TempRange'], 'purple', alpha=0.7)
    axes[1, 0].set_title('Daily Temperature Range', fontsize=12)
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Temperature Range (°C)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 9am vs 3pm temperatures
    axes[1, 1].scatter(bom_df['Temp9am'], bom_df['Temp3pm'], alpha=0.6, color='orange')
    axes[1, 1].plot([bom_df['Temp9am'].min(), bom_df['Temp9am'].max()], 
                    [bom_df['Temp9am'].min(), bom_df['Temp9am'].max()], 'k--', alpha=0.5)
    axes[1, 1].set_title('9am vs 3pm Temperature Correlation', fontsize=12)
    axes[1, 1].set_xlabel('9am Temperature (°C)')
    axes[1, 1].set_ylabel('3pm Temperature (°C)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Monthly temperature averages
    bom_df['Month'] = bom_df['Date'].dt.month
    monthly_temp = bom_df.groupby('Month')[['MinTemp', 'MaxTemp', 'AvgTemp']].mean()
    axes[2, 0].plot(monthly_temp.index, monthly_temp['MinTemp'], 'b-o', label='Min Temp')
    axes[2, 0].plot(monthly_temp.index, monthly_temp['MaxTemp'], 'r-o', label='Max Temp') 
    axes[2, 0].plot(monthly_temp.index, monthly_temp['AvgTemp'], 'g-o', label='Avg Temp')
    axes[2, 0].set_title('Monthly Temperature Averages', fontsize=12)
    axes[2, 0].set_xlabel('Month')
    axes[2, 0].set_ylabel('Temperature (°C)')
    axes[2, 0].legend(fontsize=9)
    axes[2, 0].grid(True, alpha=0.3)
    
    # Temperature variability by month
    monthly_std = bom_df.groupby('Month')['AvgTemp'].std()
    axes[2, 1].bar(monthly_std.index, monthly_std.values, alpha=0.7, color='teal')
    axes[2, 1].set_title('Temperature Variability by Month', fontsize=12)
    axes[2, 1].set_xlabel('Month')
    axes[2, 1].set_ylabel('Temperature Std Dev (°C)')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=False)

def plot_house4_energy_data(house4_df):
    """Plot House 4 energy consumption data with A4-vertical layout"""
    print("Creating House 4 energy plots...")
    
    # More vertical layout for A4 compatibility (3 rows, 2 columns)
    fig, axes = plt.subplots(3, 2, figsize=(8.5, 12))  # A4-like proportions
    fig.suptitle('House 4 Energy Consumption Analysis', fontsize=14, fontweight='bold')
      # Energy consumption over time
    axes[0, 0].plot(house4_df['DateTime'], house4_df['Power_kW'], 'b-', alpha=0.7, linewidth=0.5)
    axes[0, 0].set_title('Energy Consumption Over Time', fontsize=12)
    axes[0, 0].set_xlabel('Date/Time')
    axes[0, 0].set_ylabel('Power (kW)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Power distribution
    axes[0, 1].hist(house4_df['Power_kW'], bins=50, alpha=0.7, color='green')
    axes[0, 1].set_title('Power Consumption Distribution', fontsize=12)
    axes[0, 1].set_xlabel('Power (kW)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
      # Daily average consumption
    house4_df['Date'] = house4_df['DateTime'].dt.date
    daily_avg = house4_df.groupby('Date')['Power_kW'].mean().reset_index()
    daily_avg['Date'] = pd.to_datetime(daily_avg['Date'])
    
    axes[1, 0].plot(daily_avg['Date'], daily_avg['Power_kW'], 'r-', linewidth=2)
    axes[1, 0].set_title('Daily Average Power Consumption', fontsize=12)
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Average Power (kW)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Daily power consumption change (day-to-day difference)
    house4_df['Date'] = house4_df['DateTime'].dt.date
    daily_avg = house4_df.groupby('Date')['Power_kW'].mean().reset_index()
    daily_avg['Date'] = pd.to_datetime(daily_avg['Date'])
    daily_avg = daily_avg.sort_values('Date')
    
    # Calculate day-to-day change
    daily_avg['Daily_Change'] = daily_avg['Power_kW'].diff()
    
    # Create bar chart with positive changes in green, negative in red
    colors = ['red' if x < 0 else 'green' for x in daily_avg['Daily_Change'].fillna(0)]
    axes[1, 1].bar(range(len(daily_avg)), daily_avg['Daily_Change'].fillna(0), 
                   alpha=0.7, color=colors, width=0.8)
    axes[1, 1].set_title('Daily Power Consumption Change', fontsize=12)
    axes[1, 1].set_xlabel('Days (Sequential)')
    axes[1, 1].set_ylabel('Power Change (kW)')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Weekly consumption pattern
    house4_df['Weekday'] = house4_df['DateTime'].dt.dayofweek
    weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    weekly_avg = house4_df.groupby('Weekday')['Power_kW'].mean()
    
    axes[2, 0].bar(range(7), weekly_avg.values, alpha=0.7, color='purple')
    axes[2, 0].set_title('Average Power by Day of Week', fontsize=12)
    axes[2, 0].set_xlabel('Day of Week')
    axes[2, 0].set_ylabel('Average Power (kW)')
    axes[2, 0].set_xticks(range(7))
    axes[2, 0].set_xticklabels(weekday_names)
    axes[2, 0].grid(True, alpha=0.3)
    
    # Monthly consumption pattern
    house4_df['Month'] = house4_df['DateTime'].dt.month
    monthly_avg = house4_df.groupby('Month')['Power_kW'].mean()
    
    axes[2, 1].plot(monthly_avg.index, monthly_avg.values, 'mo-', linewidth=2, markersize=6)
    axes[2, 1].set_title('Average Power by Month', fontsize=12)
    axes[2, 1].set_xlabel('Month')
    axes[2, 1].set_ylabel('Average Power (kW)')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=False)

def plot_correlation_analysis(bom_df, house4_df):
    """Plot correlation between weather and energy data with A4-vertical layout"""
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
    
    # More vertical layout for A4 compatibility (3 rows, 2 columns)
    fig, axes = plt.subplots(3, 2, figsize=(8.5, 12))  # A4-like proportions
    fig.suptitle('Weather vs Energy Consumption Correlation Analysis', fontsize=14, fontweight='bold')
    
    # Temperature vs Power scatter plots
    axes[0, 0].scatter(merged_df['MinTemp'], merged_df['DailyAvgPower'], alpha=0.6, color='blue')
    axes[0, 0].set_title('Min Temperature vs Daily Avg Power', fontsize=12)
    axes[0, 0].set_xlabel('Min Temperature (°C)')
    axes[0, 0].set_ylabel('Daily Avg Power (kW)')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].scatter(merged_df['MaxTemp'], merged_df['DailyAvgPower'], alpha=0.6, color='red')
    axes[0, 1].set_title('Max Temperature vs Daily Avg Power', fontsize=12)
    axes[0, 1].set_xlabel('Max Temperature (°C)')
    axes[0, 1].set_ylabel('Daily Avg Power (kW)')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].scatter(merged_df['AvgTemp'], merged_df['DailyAvgPower'], alpha=0.6, color='green')
    axes[1, 0].set_title('Average Temperature vs Daily Avg Power', fontsize=12)
    axes[1, 0].set_xlabel('Average Temperature (°C)')
    axes[1, 0].set_ylabel('Daily Avg Power (kW)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 9am and 3pm temperature vs power
    axes[1, 1].scatter(merged_df['Temp9am'], merged_df['DailyAvgPower'], alpha=0.6, color='orange', label='9am Temp')
    axes[1, 1].scatter(merged_df['Temp3pm'], merged_df['DailyAvgPower'], alpha=0.6, color='purple', label='3pm Temp')
    axes[1, 1].set_title('9am & 3pm Temperature vs Power', fontsize=12)
    axes[1, 1].set_xlabel('Temperature (°C)')
    axes[1, 1].set_ylabel('Daily Avg Power (kW)')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Correlation heatmap
    corr_data = merged_df[['MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm', 'AvgTemp', 'DailyAvgPower']].corr()
    im = axes[2, 0].imshow(corr_data.values, cmap='coolwarm', vmin=-1, vmax=1)
    axes[2, 0].set_xticks(range(len(corr_data.columns)))
    axes[2, 0].set_yticks(range(len(corr_data.columns)))
    axes[2, 0].set_xticklabels(corr_data.columns, rotation=45, fontsize=9)
    axes[2, 0].set_yticklabels(corr_data.columns, fontsize=9)
    axes[2, 0].set_title('Correlation Matrix', fontsize=12)
    # Add correlation values to heatmap
    for i in range(len(corr_data.columns)):
        for j in range(len(corr_data.columns)):
            axes[2, 0].text(j, i, f'{corr_data.iloc[i, j]:.2f}', 
                           ha='center', va='center', color='black', fontsize=7)
    
    # Temperature range vs power
    merged_df['TempRange'] = merged_df['MaxTemp'] - merged_df['MinTemp']
    axes[2, 1].scatter(merged_df['TempRange'], merged_df['DailyAvgPower'], alpha=0.6, color='teal')
    axes[2, 1].set_title('Temperature Range vs Daily Avg Power', fontsize=12)
    axes[2, 1].set_xlabel('Daily Temperature Range (°C)')
    axes[2, 1].set_ylabel('Daily Avg Power (kW)')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.colorbar(im, ax=axes[2, 0], shrink=0.6)
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
    print("🔍 COMPREHENSIVE DATA VISUALIZATION TOOL")
    print("="*60)
    print("📊 This script creates multiple visualization sets:")
    print("   1. Weather data analysis from .dat files")
    print("   2. Energy consumption analysis from .dat files") 
    print("   3. Weather-energy correlation analysis")
    print("   4. ML training results analysis from CSV files")
    print("   5. ML training progression (reproduces ML.py plots from CSV data)")
    print("   6. Prediction differences between epochs")
    print("   7. Actual vs predicted comparisons")
    print("="*60)
    
    # Load data from .dat files
    bom_data, house4_data = load_dat_files()
    
    if bom_data is None or house4_data is None:
        print("❌ Cannot proceed without data files. Run ML.py first!")
        return
    
    # Convert to DataFrames
    bom_df, house4_df = convert_to_dataframes(bom_data, house4_data)
    
    # Print summary
    print_data_summary(bom_df, house4_df)
    
    print("\n📈 Creating visualizations...")
    
    # Create plots from .dat files
    plot_bom_weather_data(bom_df)
    plot_house4_energy_data(house4_df)
    plot_correlation_analysis(bom_df, house4_df)
    
    # Try to load and display CSV files if they exist
    plot_csv_results()
      # Create new 5x2 grid plots for ML results (reproduces ML.py plots from CSV data)    plot_actual_vs_predicted_5x2()
    plot_iteration_differences_5x2()
    plot_ml_training_results_5x2()
    # Removed plot_ml_differences_5x2() - was duplicate of Figure 7
    
    print("\n✅ All visualizations completed!")
    print("📁 Data sources:")
    print("   • bom.dat / house4.dat (processed weather & energy data)")
    print("   • incremental_epoch_results.csv (ML training results)")
    print("   • epoch_differences_results.csv (epoch comparison data)")
    print("\n📊 Generated plot types:")
    print("   • Weather data analysis (temperature, distributions, correlations)")
    print("   • Energy consumption patterns and trends")
    print("   • Weather-energy correlation analysis")
    print("   • ML training progression with enhanced delta statistics")
    print("   • Prediction differences between training epochs")
    print("   • Actual vs predicted comparison grids")
    print("   • Statistical analysis and performance metrics")
    print("   ⚠️  Note: Removed duplicate Figure 9 - enhanced Figure 7 with delta stats instead")
    
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
    """Plot epoch training results with enhanced A4-vertical layout"""
    unique_epochs = sorted(epoch_df['Epoch'].unique())
    
    # More vertical layout for A4 compatibility (3 rows, 2 columns)
    fig, axes = plt.subplots(3, 2, figsize=(8.5, 12))  # A4-like proportions
    fig.suptitle('ML Training Results - Epoch Progression', fontsize=14, fontweight='bold')
    
    # MSE progression
    epoch_mse = epoch_df.groupby('Epoch')['MSE'].first()
    axes[0, 0].plot(epoch_mse.index, epoch_mse.values, 'bo-', linewidth=2, markersize=6)
    axes[0, 0].set_title('MSE vs Epochs', fontsize=12)
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Mean Squared Error')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Error rate progression
    epoch_error_rate = epoch_df.groupby('Epoch')['Error_Rate_%'].first()
    axes[0, 1].plot(epoch_error_rate.index, epoch_error_rate.values, 'ro-', linewidth=2, markersize=6)
    axes[0, 1].set_title('Error Rate vs Epochs', fontsize=12)
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Error Rate (%)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Actual vs Predicted for ALL epochs with distinct colors and legends
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, epoch in enumerate(unique_epochs):
        epoch_data = epoch_df[epoch_df['Epoch'] == epoch]
        color = colors[i % len(colors)]
        axes[1, 0].scatter(epoch_data['Actual_kW'], epoch_data['Predicted_kW'], 
                          alpha=0.7, color=color, label=f'{epoch} epochs', s=25)
    
    # Add perfect prediction line
    min_val = epoch_df['Actual_kW'].min()
    max_val = epoch_df['Actual_kW'].max()
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2, label='Perfect Prediction')
    axes[1, 0].set_title('Actual vs Predicted (All Epochs)', fontsize=12)
    axes[1, 0].set_xlabel('Actual Power (kW)')
    axes[1, 0].set_ylabel('Predicted Power (kW)')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error distribution for final epoch
    final_epoch_data = epoch_df[epoch_df['Epoch'] == unique_epochs[-1]]
    axes[1, 1].hist(final_epoch_data['Error_kW'], bins=30, alpha=0.7, color='purple')
    axes[1, 1].set_title(f'Error Distribution (Epoch {unique_epochs[-1]})', fontsize=12)
    axes[1, 1].set_xlabel('Prediction Error (kW)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Average error per epoch
    avg_abs_error = epoch_df.groupby('Epoch')['Error_kW'].apply(lambda x: np.abs(x).mean())
    axes[2, 0].plot(avg_abs_error.index, avg_abs_error.values, 'go-', linewidth=2, markersize=6)
    axes[2, 0].set_title('Average Absolute Error vs Epochs', fontsize=12)
    axes[2, 0].set_xlabel('Epochs')
    axes[2, 0].set_ylabel('Average Absolute Error (kW)')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Prediction spread by epoch
    epoch_std = epoch_df.groupby('Epoch')['Predicted_kW'].std()
    axes[2, 1].plot(epoch_std.index, epoch_std.values, 'mo-', linewidth=2, markersize=6)
    axes[2, 1].set_title('Prediction Variability vs Epochs', fontsize=12)
    axes[2, 1].set_xlabel('Epochs')
    axes[2, 1].set_ylabel('Prediction Std Deviation (kW)')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=False)

def plot_epoch_differences(diff_df):
    """Plot epoch difference analysis with A4-vertical layout"""
    unique_transitions = diff_df[['From_Epoch', 'To_Epoch']].drop_duplicates()
    
    # More vertical layout for A4 compatibility (3 rows, 2 columns)
    fig, axes = plt.subplots(3, 2, figsize=(8.5, 12))  # A4-like proportions
    fig.suptitle('Epoch-to-Epoch Prediction Changes Analysis', fontsize=14, fontweight='bold')
    
    # Mean differences across epoch transitions
    mean_diffs = diff_df.groupby(['From_Epoch', 'To_Epoch'])['Mean_Difference'].first()
    transition_labels = [f"{int(idx[0])}-{int(idx[1])}" for idx in mean_diffs.index]
    
    axes[0, 0].bar(range(len(mean_diffs)), mean_diffs.values, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Mean Prediction Differences', fontsize=12)
    axes[0, 0].set_xlabel('Epoch Transition')
    axes[0, 0].set_ylabel('Mean Difference (kW)')
    axes[0, 0].set_xticks(range(len(transition_labels)))
    axes[0, 0].set_xticklabels(transition_labels, rotation=45, fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Standard deviations of differences
    std_diffs = diff_df.groupby(['From_Epoch', 'To_Epoch'])['Std_Difference'].first()
    
    axes[0, 1].bar(range(len(std_diffs)), std_diffs.values, alpha=0.7, color='lightcoral')
    axes[0, 1].set_title('Std Dev of Changes', fontsize=12)
    axes[0, 1].set_xlabel('Epoch Transition')
    axes[0, 1].set_ylabel('Std Deviation (kW)')
    axes[0, 1].set_xticks(range(len(transition_labels)))
    axes[0, 1].set_xticklabels(transition_labels, rotation=45, fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Distribution of prediction differences for each transition
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
    
    for i, (from_epoch, to_epoch) in enumerate(unique_transitions.values):
        transition_data = diff_df[(diff_df['From_Epoch'] == from_epoch) & 
                                 (diff_df['To_Epoch'] == to_epoch)]
        color = colors[i % len(colors)]
        axes[1, 0].hist(transition_data['Prediction_Difference'], bins=20, alpha=0.6, 
                       label=f'{int(from_epoch)}-{int(to_epoch)}', color=color)
    
    axes[1, 0].set_title('Distribution of Individual Changes', fontsize=12)
    axes[1, 0].set_xlabel('Prediction Difference (kW)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Convergence pattern - showing how differences decrease over time
    mean_abs_diffs = diff_df.groupby(['From_Epoch', 'To_Epoch'])['Prediction_Difference'].apply(lambda x: np.abs(x).mean())
    
    axes[1, 1].plot(range(len(mean_abs_diffs)), mean_abs_diffs.values, 'go-', linewidth=2, markersize=6)
    axes[1, 1].set_title('Average Absolute Change', fontsize=12)
    axes[1, 1].set_xlabel('Epoch Transition')
    axes[1, 1].set_ylabel('Average Absolute Change (kW)')
    axes[1, 1].set_xticks(range(len(transition_labels)))
    axes[1, 1].set_xticklabels(transition_labels, rotation=45, fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Prediction stability (difference variance analysis)
    var_diffs = diff_df.groupby(['From_Epoch', 'To_Epoch'])['Prediction_Difference'].var()
    
    axes[2, 0].plot(range(len(var_diffs)), var_diffs.values, 'mo-', linewidth=2, markersize=6)
    axes[2, 0].set_title('Prediction Change Variance', fontsize=12)
    axes[2, 0].set_xlabel('Epoch Transition')
    axes[2, 0].set_ylabel('Variance (kW²)')
    axes[2, 0].set_xticks(range(len(transition_labels)))
    axes[2, 0].set_xticklabels(transition_labels, rotation=45, fontsize=9)
    axes[2, 0].grid(True, alpha=0.3)
    
    # Summary statistics table visualization
    summary_stats = []
    for from_epoch, to_epoch in unique_transitions.values:
        transition_data = diff_df[(diff_df['From_Epoch'] == from_epoch) & 
                                 (diff_df['To_Epoch'] == to_epoch)]
        stats = {
            'Transition': f'{int(from_epoch)}-{int(to_epoch)}',
            'Mean': transition_data['Prediction_Difference'].mean(),
            'Std': transition_data['Prediction_Difference'].std(),
            'Min': transition_data['Prediction_Difference'].min(),
            'Max': transition_data['Prediction_Difference'].max()
        }
        summary_stats.append(stats)
    
    # Create text summary in the final subplot
    axes[2, 1].axis('off')
    axes[2, 1].set_title('Summary Statistics', fontsize=12)    
    table_text = "Transition | Mean | Std | Min | Max\n" + "-" * 35 + "\n"
    for stat in summary_stats:
        table_text += f"{stat['Transition']:>8} | {stat['Mean']:>5.3f} | {stat['Std']:>4.3f} | {stat['Min']:>5.3f} | {stat['Max']:>5.3f}\n"
    
    axes[2, 1].text(0.05, 0.95, table_text, transform=axes[2, 1].transAxes, fontsize=9, 
                   verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show(block=False)

def plot_actual_vs_predicted_5x2():
    """Plot 5x2 grid of actual vs predicted values for each epoch with delta statistics"""
    print("📊 Creating 5x2 Actual vs Predicted comparison plots...")
    
    results_file = os.path.join(refined_datasets_dir, 'incremental_epoch_results.csv')
    differences_file = os.path.join(refined_datasets_dir, 'epoch_differences_results.csv')
    
    if not os.path.exists(results_file):
        print(f"❌ {results_file} not found! Run the ML script first.")
        return
    
    # Load results data
    results_df = pd.read_csv(results_file)
    epochs = sorted(results_df['Epoch'].unique())
    
    # Load differences data if available for enhanced titles
    delta_stats = {}
    if os.path.exists(differences_file):
        diff_df = pd.read_csv(differences_file)
        # Calculate mean and std for each epoch transition
        for _, group in diff_df.groupby(['From_Epoch', 'To_Epoch']):
            from_epoch = int(group['From_Epoch'].iloc[0])
            to_epoch = int(group['To_Epoch'].iloc[0])
            mean_delta = group['Mean_Difference'].iloc[0]
            std_delta = group['Std_Difference'].iloc[0]
            delta_stats[to_epoch] = {'mean_delta': mean_delta, 'std_delta': std_delta, 'from_epoch': from_epoch}
    
    # Create 5x2 subplot grid with A4-friendly dimensions
    fig, axes = plt.subplots(5, 2, figsize=(8.5, 14))  # Taller and narrower for A4
    
    fig.suptitle('Actual vs Predicted Energy Consumption - Incremental Training Comparison', 
                 fontsize=14, fontweight='bold')
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    # Define distinct colors for each epoch
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Plot each epoch's results
    for i, epoch in enumerate(epochs):
        if i >= 10:  # Only plot first 10 epochs (5x2 grid)
            break
            
        epoch_data = results_df[results_df['Epoch'] == epoch]
        color = colors[i % len(colors)]
        
        # Scatter plot: Actual vs Predicted
        axes_flat[i].scatter(epoch_data['Actual_kW'], epoch_data['Predicted_kW'], 
                           alpha=0.7, s=25, color=color, label=f'{epoch} epochs')
        
        # Perfect prediction line (y=x)
        min_val = min(epoch_data['Actual_kW'].min(), epoch_data['Predicted_kW'].min())
        max_val = max(epoch_data['Actual_kW'].max(), epoch_data['Predicted_kW'].max())
        axes_flat[i].plot([min_val, max_val], [min_val, max_val], 
                         'k--', linewidth=2, alpha=0.8, label='Perfect Prediction')
        
        # Calculate R² and MSE for the subplot
        mse = epoch_data['MSE'].iloc[0]
        correlation = np.corrcoef(epoch_data['Actual_kW'], epoch_data['Predicted_kW'])[0, 1]
        r_squared = correlation ** 2
        
        # Enhanced title with delta statistics if available
        title = f'Epoch {epoch}\nMSE: {mse:.4f}, R²: {r_squared:.3f}'
        if epoch in delta_stats:
            mean_delta = delta_stats[epoch]['mean_delta']
            std_delta = delta_stats[epoch]['std_delta']
            from_epoch = delta_stats[epoch]['from_epoch']
            title += f'\nΔ from {from_epoch}: μ={mean_delta:.4f}, σ={std_delta:.4f}'
        
        axes_flat[i].set_title(title, fontsize=10)
        axes_flat[i].set_xlabel('Actual Energy (kW)', fontsize=9)
        axes_flat[i].set_ylabel('Predicted Energy (kW)', fontsize=9)
        axes_flat[i].grid(True, alpha=0.3)
        axes_flat[i].legend(fontsize=7)
        axes_flat[i].tick_params(labelsize=8)
        
        # Set equal aspect ratio for better comparison
        axes_flat[i].set_aspect('equal', adjustable='box')
    
    # Hide unused subplots if there are fewer than 10 epochs
    for i in range(len(epochs), 10):
        axes_flat[i].set_visible(False)
    
    # Auto-adjust layout with dynamic spacing
    plt.tight_layout()
    plt.show(block=False)
    print("✅ Actual vs Predicted 5x2 plot completed with delta statistics!")

def plot_iteration_differences_5x2():
    """Plot 5x2 grid (9 plots) showing differences between consecutive epoch iterations"""
    print("📊 Creating 5x2 Iteration Differences comparison plots...")
    
    results_file = os.path.join(refined_datasets_dir, 'incremental_epoch_results.csv')
    if not os.path.exists(results_file):
        print(f"❌ {results_file} not found! Run the ML script first.")
        return
    
    # Load results data
    results_df = pd.read_csv(results_file)
    epochs = sorted(results_df['Epoch'].unique())
    
    if len(epochs) < 2:
        print("❌ Need at least 2 epochs to calculate differences!")
        return
    
    # Create 5x2 subplot grid with A4-friendly dimensions
    fig, axes = plt.subplots(5, 2, figsize=(8.5, 14))  # Taller and narrower for A4
    fig.suptitle('Prediction Differences Between Consecutive Training Epochs', 
                 fontsize=14, fontweight='bold')
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    # Calculate and plot differences between consecutive epochs
    plot_count = 0
    for i in range(len(epochs) - 1):
        if plot_count >= 9:  # Maximum 9 plots (leaving one subplot empty for legend/info)
            break
            
        current_epoch = epochs[i]
        next_epoch = epochs[i + 1]
        
        # Get data for both epochs
        current_data = results_df[results_df['Epoch'] == current_epoch].sort_values('DataPoint')
        next_data = results_df[results_df['Epoch'] == next_epoch].sort_values('DataPoint')
        
        # Calculate prediction differences
        prediction_diff = next_data['Predicted_kW'].values - current_data['Predicted_kW'].values
        data_points = current_data['DataPoint'].values
        
        # Create difference plot
        colors = ['red' if diff > 0 else 'blue' for diff in prediction_diff]
        axes_flat[plot_count].bar(data_points, prediction_diff, color=colors, alpha=0.7, width=0.8)
        
        # Statistics
        mean_diff = np.mean(np.abs(prediction_diff))
        max_diff = np.max(np.abs(prediction_diff))
        
        axes_flat[plot_count].set_title(f'Epoch {current_epoch} → {next_epoch}\n' +
                                       f'Mean |Δ|: {mean_diff:.4f}, Max |Δ|: {max_diff:.4f}', fontsize=10)
        axes_flat[plot_count].set_xlabel('Data Point', fontsize=9)
        axes_flat[plot_count].set_ylabel('Prediction Difference (kW)', fontsize=9)
        axes_flat[plot_count].grid(True, alpha=0.3)
        axes_flat[plot_count].axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        axes_flat[plot_count].tick_params(labelsize=8)
        
        plot_count += 1
    
    # Use the last subplot for summary information
    if plot_count < 10:
        summary_ax = axes_flat[9]  # Last subplot
        summary_ax.axis('off')
        
        # Create summary text with adjusted font size for A4
        summary_text = "Summary of Iteration Differences:\n\n"
        summary_text += "🔴 Red bars: Prediction increased\n"
        summary_text += "🔵 Blue bars: Prediction decreased\n\n"
        summary_text += "Interpretation:\n"
        summary_text += "• Large differences indicate model learning\n"
        summary_text += "• Small differences suggest convergence\n"
        summary_text += "• Consistent patterns show stable learning\n\n"
        
        # Calculate overall statistics
        all_diffs = []
        for i in range(len(epochs) - 1):
            current_data = results_df[results_df['Epoch'] == epochs[i]].sort_values('DataPoint')
            next_data = results_df[results_df['Epoch'] == epochs[i + 1]].sort_values('DataPoint')
            diff = next_data['Predicted_kW'].values - current_data['Predicted_kW'].values
            all_diffs.extend(diff)
        
        summary_text += f"Overall Statistics:\n"
        summary_text += f"Mean |Difference|: {np.mean(np.abs(all_diffs)):.4f} kW\n"
        summary_text += f"Std Difference: {np.std(all_diffs):.4f} kW\n"
        summary_text += f"Max |Difference|: {np.max(np.abs(all_diffs)):.4f} kW"
        
        summary_ax.text(0.05, 0.95, summary_text, transform=summary_ax.transAxes, 
                       verticalalignment='top', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Hide unused subplots
    for i in range(plot_count, 9):  # Hide subplots before the summary
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    plt.show(block=False)
    print("✅ Iteration Differences 5x2 plot completed!")

def plot_ml_training_results_5x2():
    """Recreate the ML training 5x2 grid plots using CSV data with delta statistics"""
    print("📊 Creating ML Training Results 5x2 plots from CSV data...")
    
    results_file = os.path.join(refined_datasets_dir, 'incremental_epoch_results.csv')
    if not os.path.exists(results_file):
        print(f"❌ {results_file} not found! Run the ML script first.")
        return
    
    # Load results data
    results_df = pd.read_csv(results_file)
    epochs = sorted(results_df['Epoch'].unique())
    
    if len(epochs) == 0:
        print("❌ No epoch data found in CSV file!")
        return
    
    # Calculate delta statistics between consecutive epochs
    delta_stats = {}
    for i in range(1, len(epochs)):
        current_epoch = epochs[i]
        previous_epoch = epochs[i-1]
        
        current_data = results_df[results_df['Epoch'] == current_epoch].sort_values('DataPoint')
        previous_data = results_df[results_df['Epoch'] == previous_epoch].sort_values('DataPoint')
        
        # Calculate prediction differences
        prediction_diff = current_data['Predicted_kW'].values - previous_data['Predicted_kW'].values
        mean_delta = np.mean(prediction_diff)
        std_delta = np.std(prediction_diff)
        
        delta_stats[current_epoch] = {
            'mean_delta': mean_delta,
            'std_delta': std_delta,
            'from_epoch': previous_epoch
        }
    
    # Create 5x2 subplot grid with A4-friendly dimensions
    fig, axes = plt.subplots(5, 2, figsize=(8.5, 14))  # Taller and narrower for A4
    
    fig.suptitle('Incremental Epoch Comparison: ML Training Results\nBOM Weather → House 4 Energy Prediction (Enhanced with Δ Stats)', 
                fontsize=12, fontweight='bold')
    
    # Plot each epoch's results
    for i, epoch in enumerate(epochs):
        if i >= 10:  # Only plot first 10 epochs (5x2 grid)
            break
            
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        # Get data for this epoch
        epoch_data = results_df[results_df['Epoch'] == epoch].sort_values('DataPoint')
        
        x_range = range(1, len(epoch_data) + 1)
        ax.plot(x_range, epoch_data['Actual_kW'], 'b-', label='Actual', linewidth=2)
        ax.plot(x_range, epoch_data['Predicted_kW'], 'r--', label='Predicted', linewidth=2)
        
        # Get statistics
        mse = epoch_data['MSE'].iloc[0]
        error_rate = epoch_data['Error_Rate_%'].iloc[0]
        
        # Enhanced title with delta statistics
        title = f"Epochs: {epoch}\nMSE: {mse:.4f}, Error: {error_rate:.1f}%"
        if epoch in delta_stats:
            mean_delta = delta_stats[epoch]['mean_delta']
            std_delta = delta_stats[epoch]['std_delta']
            from_epoch = delta_stats[epoch]['from_epoch']
            title += f"\nΔ from {from_epoch}: μ={mean_delta:.4f}, σ={std_delta:.4f}"
        
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Data Point #', fontsize=9)
        ax.set_ylabel('Power (kW)', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
    
    # Hide unused subplots if there are fewer than 10 epochs
    for i in range(len(epochs), 10):
        row = i // 2
        col = i % 2
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.show(block=False)
    
    print("✅ ML Training Results 5x2 plot completed!")

def plot_ml_differences_5x2():
    """Recreate the ML training differences 5x2 grid plots using CSV data"""
    print("📊 Creating ML Training Differences 5x2 plots from CSV data...")
    
    results_file = os.path.join(refined_datasets_dir, 'incremental_epoch_results.csv')
    if not os.path.exists(results_file):
        print(f"❌ {results_file} not found! Run the ML script first.")
        return
    
    # Load results data
    results_df = pd.read_csv(results_file)
    epochs = sorted(results_df['Epoch'].unique())
    
    if len(epochs) < 2:
        print("❌ Need at least 2 epochs to calculate differences!")
        return
      # Set dynamic font scaling for this figure (9 subplots, 16x20 size)
    set_dynamic_font_params(num_subplots=9, figure_size=(16, 20))
    
    # Create 5x2 subplot grid
    fig, axes = plt.subplots(5, 2, figsize=(16, 20))
    
    # Calculate dynamic font sizes
    fonts = calculate_dynamic_font_sizes(fig.get_size_inches())
    
    fig.suptitle('Prediction Differences Between Consecutive Epochs\n(Current Epoch - Previous Epoch) from CSV',
                fontsize=fonts['title'], fontweight='bold')
    
    # Calculate differences between consecutive epochs
    plot_count = 0
    for i in range(len(epochs) - 1):
        if plot_count >= 9:  # Maximum 9 plots
            break
            
        row = plot_count // 2
        col = plot_count % 2
        ax = axes[row, col]
        
        current_epoch = epochs[i + 1]
        previous_epoch = epochs[i]
        
        # Get data for both epochs
        current_data = results_df[results_df['Epoch'] == current_epoch].sort_values('DataPoint')
        previous_data = results_df[results_df['Epoch'] == previous_epoch].sort_values('DataPoint')
          # Calculate prediction differences
        prediction_diff = current_data['Predicted_kW'].values - previous_data['Predicted_kW'].values
        
        x_range = range(1, len(prediction_diff) + 1)
        ax.plot(x_range, prediction_diff, 'g-', linewidth=2, label='Prediction Difference')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Calculate statistics
        mean_diff = np.mean(prediction_diff)
        std_diff = np.std(prediction_diff)
        
        ax.set_title(f"Epochs {previous_epoch} → {current_epoch}\nMean Δ: {mean_diff:.4f}, Std: {std_diff:.4f}", 
                    fontsize=fonts['subtitle'], fontweight='bold')
        ax.set_xlabel('Data Point #', fontsize=fonts['label'])
        ax.set_ylabel('Prediction Difference (kW)', fontsize=fonts['label'])
        ax.legend(fontsize=fonts['legend'])
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=fonts['tick'])
        
        plot_count += 1
    
    # Hide the last subplot since we only have 9 difference plots
    axes[4, 1].set_visible(False)
    
    # Hide any other unused subplots
    for i in range(plot_count, 9):
        row = i // 2
        col = i % 2
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.show(block=False)
    
    # Reset font parameters
    reset_font_params()
    
    print("✅ ML Training Differences 5x2 plot completed!")

# ====================
# DYNAMIC FONT SCALING UTILITIES
# ====================

def calculate_dynamic_font_sizes(figure_size):
    """
    Calculate dynamic font sizes based on figure dimensions
    
    Args:
        figure_size: Tuple of (width, height) in inches
    
    Returns:
        dict: Dictionary of font sizes for different elements
    """
    base_size = min(figure_size)
    return {
        'title': max(8, int(base_size * 0.8)),
        'subtitle': max(6, int(base_size * 0.5)),
        'label': max(5, int(base_size * 0.4)),
        'legend': max(4, int(base_size * 0.35)),
        'tick': max(4, int(base_size * 0.3))
    }

def set_dynamic_font_params(num_subplots=1, figure_size=(12, 8)):
    """
    Configure matplotlib to use dynamic font scaling based on figure complexity
    
    Args:
        num_subplots: Number of subplots in the figure
        figure_size: Tuple of (width, height) in inches
    """
    # Calculate scaling factors
    subplot_density = num_subplots / (figure_size[0] * figure_size[1])
    base_scale = min(figure_size) / 10  # Base scaling factor
    
    # Set matplotlib rcParams for dynamic scaling
    plt.rcParams.update({
        'font.size': max(8, int(12 * base_scale / max(1, subplot_density))),
        'axes.titlesize': max(10, int(14 * base_scale / max(1, subplot_density))),
        'axes.labelsize': max(8, int(11 * base_scale / max(1, subplot_density))),
        'xtick.labelsize': max(7, int(9 * base_scale / max(1, subplot_density))),
        'ytick.labelsize': max(7, int(9 * base_scale / max(1, subplot_density))),
        'legend.fontsize': max(6, int(8 * base_scale / max(1, subplot_density))),
        'figure.titlesize': max(12, int(16 * base_scale / max(1, subplot_density)))
    })

def reset_font_params():
    """Reset matplotlib font parameters to defaults"""
    plt.rcParams.update(plt.rcParamsDefault)

if __name__ == "__main__":
    main()
