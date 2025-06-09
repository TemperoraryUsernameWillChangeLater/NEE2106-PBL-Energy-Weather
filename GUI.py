import tkinter as  tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
import pandas as pd
import os



# Define month mapping for variable names
month_mapping = {
    '05': 'May24', '06': 'Jun24', '07': 'Jul24', '08': 'Aug24',
    '09': 'Sep24', '10': 'Oct24', '11': 'Nov24', '12': 'Dec24',
    '01': 'Jan25', '02': 'Feb25', '03': 'Mar25', '04': 'Apr25'  # Feb25 has 28 days (2025 is not a leap year)
}

# Dictionary to store all datasets
datasets = {}

# Read all dataset files in a loop
dataset_folder = "DataSets"
for filename in os.listdir(dataset_folder):
    if filename.endswith('.csv'):
        # Extract the month-year part from filename (e.g., '24-05' from '24-05.csv')
        file_key = filename.replace('.csv', '')
        
        # Get the month part (e.g., '05' from '24-05')
        month_part = file_key.split('-')[1]
        
        # Get the corresponding variable name (e.g., 'May24' for '05')
        if month_part in month_mapping:
            var_name = month_mapping[month_part]            # Read the CSV file - skip the header rows and use row 7 as column names
            file_path = os.path.join(dataset_folder, filename)
            datasets[var_name] = pd.read_csv(file_path, skiprows=7, encoding='latin-1')
            # Remove the first empty column if it exists
            if datasets[var_name].columns[0] == 'Unnamed: 0':
                datasets[var_name] = datasets[var_name].drop(columns=['Unnamed: 0'])
            
            print(f"Loaded {filename} as {var_name} with {len(datasets[var_name])} rows")

# Create individual variables for easier access
# Now you can use May24, Jun24, Jul24, etc. directly
for var_name, data in datasets.items():
    globals()[var_name] = data

print(f"\nAvailable datasets: {list(datasets.keys())}")
print("You can now use variables like May24, Jun24, Jul24, etc.")

# Example usage after loading all data:
# print(f"May 2024 average temperature: {May24['Maximum temperature (°C)'].mean():.1f}°C")
# print(f"Total rainfall in June 2024: {Jun24['Rainfall (mm)'].sum():.1f}mm")
# 
# # Access all temperature data across months:
# all_temps = []
# for dataset_name, data in datasets.items():
#     all_temps.extend(data['Maximum temperature (°C)'].dropna().tolist())
# print(f"Overall temperature range: {min(all_temps):.1f}°C to {max(all_temps):.1f}°C")

