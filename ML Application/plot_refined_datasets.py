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
    print("   4. ML training results analysis from CSV files (80-20 or 95-5 split)")
    print("   5. Actual vs predicted comparisons (80-20 split)")
    print("   6. Prediction differences between epochs (80-20 split)")
    print("   7. Prediction differences between epochs (95-5 split)")
    print("   8. ML training results comprehensive analysis (95-5 split)")
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
    plot_csv_results()  # Figure 4
    
    # Create dual split plots
    plot_actual_vs_predicted_80_20()  # Figure 5 (80-20 split)
    plot_iteration_differences_80_20()  # Figure 6 (80-20 split)
    plot_iteration_differences_95_5()  # Figure 7 (95-5 split)
    plot_ml_training_results_95_5()  # Figure 8 (95-5 split)
    
    # New plotting functions for dual split visualization
    plot_actual_vs_predicted_80_20()
    plot_iteration_differences_80_20()
    plot_iteration_differences_95_5()
    plot_ml_training_results_95_5()
    
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
    # Try to load dual split results first
    epoch_80_file = os.path.join(refined_datasets_dir, 'incremental_epoch_results_80_20.csv')
    epoch_95_file = os.path.join(refined_datasets_dir, 'incremental_epoch_results_95_5.csv')
    
    # Check if dual split files exist
    if os.path.exists(epoch_80_file) and os.path.exists(epoch_95_file):
        print("📊 Loading dual split results from CSV...")
        try:
            epoch_80_df = pd.read_csv(epoch_80_file)
            epoch_95_df = pd.read_csv(epoch_95_file)
            plot_epoch_results(epoch_80_df, "80-20 Split")  # Figure 4
        except Exception as e:
            print(f"❌ Error loading dual split results: {e}")
    else:
        # Fallback to original single file
        epoch_results_file = os.path.join(refined_datasets_dir, 'incremental_epoch_results.csv')
        if os.path.exists(epoch_results_file):
            print("📊 Loading epoch results from CSV...")
            try:
                epoch_df = pd.read_csv(epoch_results_file)
                plot_epoch_results(epoch_df, "95-5 Split")  # Figure 4
            except Exception as e:
                print(f"❌ Error loading epoch results: {e}")
    
    # Note: Figure 5 (plot_epoch_differences) has been removed
    # The key charts from Figure 5 are now integrated into Figure 4

def plot_epoch_results(epoch_df, split_name="Training Results"):
    """Plot epoch training results with enhanced A4-vertical layout"""
    unique_epochs = sorted(epoch_df['Epoch'].unique())
    
    # More vertical layout for A4 compatibility (3 rows, 2 columns)
    fig, axes = plt.subplots(3, 2, figsize=(8.5, 12))  # A4-like proportions
    fig.suptitle(f'ML Training Results - {split_name}', fontsize=14, fontweight='bold')
    
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

def plot_actual_vs_predicted_80_20():
    """Plot actual vs predicted for 80-20 split (Figure 5)"""
    epoch_file = os.path.join(refined_datasets_dir, 'incremental_epoch_results_80_20.csv')
    if os.path.exists(epoch_file):
        try:
            df = pd.read_csv(epoch_file)
            plot_actual_vs_predicted_split(df, "80-20 Split", figure_num=5)
        except Exception as e:
            print(f"❌ Error loading 80-20 split data: {e}")
    else:
        print("⚠️ 80-20 split data not found, skipping Figure 5")

def plot_iteration_differences_80_20():
    """Plot iteration differences for 80-20 split (Figure 6)"""
    differences_file = os.path.join(refined_datasets_dir, 'epoch_differences_results_80_20.csv')
    if os.path.exists(differences_file):
        try:
            df = pd.read_csv(differences_file)
            plot_iteration_differences_split(df, "80-20 Split", figure_num=6)
        except Exception as e:
            print(f"❌ Error loading 80-20 differences data: {e}")
    else:
        print("⚠️ 80-20 differences data not found, skipping Figure 6")

def plot_iteration_differences_95_5():
    """Plot iteration differences for 95-5 split (Figure 7)"""
    differences_file = os.path.join(refined_datasets_dir, 'epoch_differences_results_95_5.csv')
    if os.path.exists(differences_file):
        try:
            df = pd.read_csv(differences_file)
            plot_iteration_differences_split(df, "95-5 Split", figure_num=7)
        except Exception as e:
            print(f"❌ Error loading 95-5 differences data: {e}")
    else:
        print("⚠️ 95-5 differences data not found, skipping Figure 7")

def plot_ml_training_results_95_5():
    """Plot ML training results for 95-5 split (Figure 8)"""
    epoch_file = os.path.join(refined_datasets_dir, 'incremental_epoch_results_95_5.csv')
    if os.path.exists(epoch_file):
        try:
            df = pd.read_csv(epoch_file)
            plot_ml_training_results_split(df, "95-5 Split", figure_num=8)
        except Exception as e:
            print(f"❌ Error loading 95-5 training data: {e}")
    else:
        print("⚠️ 95-5 training data not found, skipping Figure 8")

def plot_actual_vs_predicted_split(df, split_name, figure_num):
    """Generic function to plot actual vs predicted for any split"""
    print(f"📊 Creating Figure {figure_num}: Actual vs Predicted - {split_name}")
    
    unique_epochs = sorted(df['Epoch'].unique())
    selected_epochs = unique_epochs[:10]  # First 10 epochs
    
    fig, axes = plt.subplots(2, 5, figsize=(8.5, 12))
    fig.suptitle(f'Figure {figure_num}: Actual vs Predicted Energy Consumption - {split_name}', fontsize=14, fontweight='bold')
    
    for i, epoch in enumerate(selected_epochs):
        row = i // 5
        col = i % 5
        
        epoch_data = df[df['Epoch'] == epoch]
        
        axes[row, col].scatter(epoch_data['Actual_kW'], epoch_data['Predicted_kW'], alpha=0.6, s=30)
        axes[row, col].plot([epoch_data['Actual_kW'].min(), epoch_data['Actual_kW'].max()], 
                           [epoch_data['Actual_kW'].min(), epoch_data['Actual_kW'].max()], 'r--', alpha=0.8)
        axes[row, col].set_title(f'Epoch {epoch}', fontsize=10)
        axes[row, col].set_xlabel('Actual (kW)', fontsize=8)
        axes[row, col].set_ylabel('Predicted (kW)', fontsize=8)
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].tick_params(labelsize=7)
    
    plt.tight_layout()
    plt.show()

def plot_iteration_differences_split(df, split_name, figure_num):
    """Generic function to plot iteration differences for any split"""
    print(f"📊 Creating Figure {figure_num}: Prediction Differences - {split_name}")
    
    unique_epochs = sorted(df['To_Epoch'].unique())
    selected_epochs = unique_epochs[:10]  # First 10 epoch transitions
    
    fig, axes = plt.subplots(2, 5, figsize=(8.5, 12))
    fig.suptitle(f'Figure {figure_num}: Prediction Differences Between Epochs - {split_name}', fontsize=14, fontweight='bold')
    
    for i, to_epoch in enumerate(selected_epochs):
        row = i // 5
        col = i % 5
        
        epoch_data = df[df['To_Epoch'] == to_epoch]
        from_epoch = epoch_data['From_Epoch'].iloc[0]
        
        axes[row, col].hist(epoch_data['Prediction_Difference_kW'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        axes[row, col].set_title(f'{from_epoch}→{to_epoch} Epochs', fontsize=10)
        axes[row, col].set_xlabel('Prediction Δ (kW)', fontsize=8)
        axes[row, col].set_ylabel('Frequency', fontsize=8)
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].tick_params(labelsize=7)
        
        # Add mean and std annotations
        mean_diff = epoch_data['Mean_Difference_kW'].iloc[0]
        std_diff = epoch_data['Std_Difference_kW'].iloc[0]
        axes[row, col].axvline(mean_diff, color='red', linestyle='--', alpha=0.8)
        axes[row, col].text(0.05, 0.95, f'μ={mean_diff:.3f}', transform=axes[row, col].transAxes, 
                           fontsize=7, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def plot_ml_training_results_split(df, split_name, figure_num):
    """Generic function to plot ML training results for any split"""
    print(f"📊 Creating Figure {figure_num}: ML Training Results - {split_name}")
    
    unique_epochs = sorted(df['Epoch'].unique())
    
    fig, axes = plt.subplots(2, 5, figsize=(8.5, 12))
    fig.suptitle(f'Figure {figure_num}: ML Training Results - {split_name}', fontsize=14, fontweight='bold')
    
    # Plot various metrics across epochs
    epoch_mse = df.groupby('Epoch')['MSE'].first()
    epoch_error_rate = df.groupby('Epoch')['Error_Rate_%'].first()
    
    # MSE progression
    axes[0, 0].plot(epoch_mse.index, epoch_mse.values, 'bo-', linewidth=2, markersize=6)
    axes[0, 0].set_title('MSE vs Epochs', fontsize=10)
    axes[0, 0].set_xlabel('Epochs', fontsize=8)
    axes[0, 0].set_ylabel('MSE', fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(labelsize=7)
    
    # Error rate progression
    axes[0, 1].plot(epoch_error_rate.index, epoch_error_rate.values, 'ro-', linewidth=2, markersize=6)
    axes[0, 1].set_title('Error Rate vs Epochs', fontsize=10)
    axes[0, 1].set_xlabel('Epochs', fontsize=8)
    axes[0, 1].set_ylabel('Error Rate (%)', fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(labelsize=7)
    
    # Error distribution for different epochs
    for i, epoch in enumerate(unique_epochs[:8]):
        row = (i + 2) // 5
        col = (i + 2) % 5
        
        epoch_data = df[df['Epoch'] == epoch]
        axes[row, col].hist(epoch_data['Error_kW'], bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[row, col].set_title(f'Epoch {epoch} Errors', fontsize=10)
        axes[row, col].set_xlabel('Error (kW)', fontsize=8)
        axes[row, col].set_ylabel('Frequency', fontsize=8)
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].tick_params(labelsize=7)
    
    plt.tight_layout()
    plt.show()

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
