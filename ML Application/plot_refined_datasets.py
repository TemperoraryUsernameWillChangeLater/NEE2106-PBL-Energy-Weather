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
    print("   4. ML training results analysis from CSV files (enhanced with Figure 5 content)")
    print("   5. ML training progression (reproduces ML.py plots from CSV data)")  
    print("   6. Prediction differences between epochs")
    print("   7. Actual vs predicted comparisons")
    print("   8. Figure 7: Prediction vs Actual difference analysis")
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
    
    # Create new 5x2 grid plots for ML results (reproduces ML.py plots from CSV data)
    plot_ml_training_results_5x2()
    plot_iteration_differences_5x2()
    plot_figure_7_prediction_vs_actual_differences()
    
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
    print("   • Figure 7: Prediction vs Actual difference analysis by epoch")
    
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
    
    # Note: Figure 5 (plot_epoch_differences) has been removed
    # The key charts from Figure 5 are now integrated into Figure 4

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
    
    # Mean prediction differences (moved from Figure 5)
    # Need to calculate differences between consecutive epochs
    if len(unique_epochs) > 1:
        mean_diffs = []
        transition_labels = []
        for i in range(len(unique_epochs) - 1):
            current_epoch = unique_epochs[i + 1]
            previous_epoch = unique_epochs[i]
            
            current_data = epoch_df[epoch_df['Epoch'] == current_epoch].sort_values('DataPoint')
            previous_data = epoch_df[epoch_df['Epoch'] == previous_epoch].sort_values('DataPoint')
            
            # Calculate prediction differences
            prediction_diff = current_data['Predicted_kW'].values - previous_data['Predicted_kW'].values
            mean_diff = np.mean(prediction_diff)
            mean_diffs.append(mean_diff)
            transition_labels.append(f"{previous_epoch}-{current_epoch}")
        
        axes[2, 0].bar(range(len(mean_diffs)), mean_diffs, alpha=0.7, color='skyblue')
        axes[2, 0].set_title('Mean Prediction Differences', fontsize=12)
        axes[2, 0].set_xlabel('Epoch Transition')
        axes[2, 0].set_ylabel('Mean Difference (kW)')
        axes[2, 0].set_xticks(range(len(transition_labels)))
        axes[2, 0].set_xticklabels(transition_labels, rotation=45, fontsize=9)
        axes[2, 0].grid(True, alpha=0.3)
          # Standard deviations of differences (moved from Figure 5)
        std_diffs = []
        for i in range(len(unique_epochs) - 1):
            current_epoch = unique_epochs[i + 1]
            previous_epoch = unique_epochs[i]
            
            current_data = epoch_df[epoch_df['Epoch'] == current_epoch].sort_values('DataPoint')
            previous_data = epoch_df[epoch_df['Epoch'] == previous_epoch].sort_values('DataPoint')
            
            # Calculate prediction differences
            prediction_diff = current_data['Predicted_kW'].values - previous_data['Predicted_kW'].values
            std_diff = np.std(prediction_diff)
            std_diffs.append(std_diff)
        
        axes[2, 1].bar(range(len(std_diffs)), std_diffs, alpha=0.7, color='lightcoral')
        axes[2, 1].set_title('Std Dev of Changes', fontsize=12)
        axes[2, 1].set_xlabel('Epoch Transition')
        axes[2, 1].set_ylabel('Std Deviation (kW)')
        axes[2, 1].set_xticks(range(len(transition_labels)))
        axes[2, 1].set_xticklabels(transition_labels, rotation=45, fontsize=9)
        axes[2, 1].grid(True, alpha=0.3)
    else:
        # If only one epoch, show placeholder message
        axes[2, 0].text(0.5, 0.5, 'Need at least 2 epochs\nfor difference analysis', 
                       transform=axes[2, 0].transAxes, ha='center', va='center', fontsize=12)
        axes[2, 0].set_title('Mean Prediction Differences', fontsize=12)
        axes[2, 1].text(0.5, 0.5, 'Need at least 2 epochs\nfor difference analysis', 
                       transform=axes[2, 1].transAxes, ha='center', va='center', fontsize=12)
        axes[2, 1].set_title('Std Dev of Changes', fontsize=12)
    
    plt.tight_layout()
    plt.show(block=False)

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
    fig, axes = plt.subplots(5, 2, figsize=(8.5, 12))  # A4-like proportions consistent with other figures
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
    fig, axes = plt.subplots(5, 2, figsize=(8.5, 12))  # A4-like proportions consistent with other figures
    
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
    
    # Hide unused subplots if there are fewer than 10 epochs
    for i in range(len(epochs), 10):
        row = i // 2
        col = i % 2
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.show(block=False)
    
    print("✅ ML Training Results 5x2 plot completed!")

def plot_figure_7_prediction_vs_actual_differences():
    """Figure 7: 5x2 bar plot showing difference (Predicted - Actual) for each epoch interval"""
    print("📊 Creating Figure 7: 5x2 Prediction - Actual Differences by Epoch...")
    
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
    
    # Create 5x2 subplot grid with A4-friendly dimensions
    fig, axes = plt.subplots(5, 2, figsize=(8.5, 12))  # A4-like proportions consistent with other figures
    fig.suptitle('Figure 7: Prediction - Actual Differences (5x2 Bar Plot)', 
                 fontsize=14, fontweight='bold')
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    # Define colors for each epoch
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Plot each epoch's prediction - actual differences
    for i, epoch in enumerate(epochs):
        if i >= 10:  # Only plot first 10 epochs (5x2 grid)
            break
            
        epoch_data = results_df[results_df['Epoch'] == epoch].sort_values('DataPoint')
        
        # Calculate differences: Predicted - Actual
        differences = epoch_data['Predicted_kW'] - epoch_data['Actual_kW']
        data_points = range(1, len(differences) + 1)
        
        # Create bar plot
        color = colors[i % len(colors)]
        bars = axes_flat[i].bar(data_points, differences, alpha=0.7, color=color, width=0.8)
        
        # Color bars based on positive/negative differences
        for j, (bar, diff) in enumerate(zip(bars, differences)):
            if diff > 0:
                bar.set_color('red')  # Over-prediction
                bar.set_alpha(0.7)
            else:
                bar.set_color('blue')  # Under-prediction
                bar.set_alpha(0.7)
        
        # Calculate statistics for title
        mean_diff = np.mean(differences)
        std_diff = np.std(differences)
        max_abs_diff = np.max(np.abs(differences))
        
        # Enhanced title with statistics
        title = f'Epoch {epoch}\nμ={mean_diff:.4f}, σ={std_diff:.4f}\nMax|Δ|={max_abs_diff:.4f}'
        
        axes_flat[i].set_title(title, fontsize=10)
        axes_flat[i].set_xlabel('Data Point', fontsize=9)
        axes_flat[i].set_ylabel('Predicted - Actual (kW)', fontsize=9)
        axes_flat[i].grid(True, alpha=0.3)
        axes_flat[i].axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.8)
        axes_flat[i].tick_params(labelsize=8)
        
        # Add legend only to first subplot
        if i == 0:
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='red', alpha=0.7, label='Over-prediction (Pred > Actual)'),
                             Patch(facecolor='blue', alpha=0.7, label='Under-prediction (Pred < Actual)')]
            axes_flat[i].legend(handles=legend_elements, fontsize=7, loc='upper right')
    
    # Hide unused subplots if there are fewer than 10 epochs
    for i in range(len(epochs), 10):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    plt.show(block=False)
    
    # Print summary statistics
    print("✅ Figure 7 completed!")
    print("📊 Summary of Prediction - Actual Analysis:")
    
    # Calculate overall statistics across all epochs
    all_differences = []
    for epoch in epochs:
        epoch_data = results_df[results_df['Epoch'] == epoch]
        differences = epoch_data['Predicted_kW'] - epoch_data['Actual_kW']
        all_differences.extend(differences)
    
    print(f"   • Overall mean difference: {np.mean(all_differences):.4f} kW")
    print(f"   • Overall std deviation: {np.std(all_differences):.4f} kW")
    print(f"   • Maximum over-prediction: {np.max(all_differences):.4f} kW")
    print(f"   • Maximum under-prediction: {np.min(all_differences):.4f} kW")
    print("   • 🔴 Red bars: Model over-predicts (Predicted > Actual)")
    print("   • 🔵 Blue bars: Model under-predicts (Predicted < Actual)")

if __name__ == "__main__":
    main()