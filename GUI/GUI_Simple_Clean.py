# Simple Weather Data Visualization GUI
# Using only syntax from example syntax.py file

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import os

# Simple variables to store data
data_dict = {}
months_list = ["24-05", "24-06", "24-07", "24-08", "24-09", "24-10", "24-11", "24-12", "25-01", "25-02", "25-03", "25-04"]

# Get the relative path to the CSV files
csv_folder = os.path.join(os.path.dirname(__file__), "..", "Datasets")

def normalize_column_name(col_name):
    """Normalize column names to handle encoding artifacts"""
    return str(col_name).lower().replace('ï¿½', 'o').replace('°', 'o')

def load_csv_files():
    """Load CSV files"""
    for month in months_list:
        try:
            filename = month + ".csv"
            full_path = os.path.join(csv_folder, filename)
            df = pd.read_csv(full_path, skiprows=7, encoding='latin-1')
            data_dict[month] = df
            print("Loaded:", filename)
        except Exception as e:
            print("Could not load:", filename, "Error:", str(e))

def get_selected_data():
    """Get data based on selection"""
    display_choice = display_var.get()
    
    if display_choice == "Single Month":
        month = month_var.get()
        if month in data_dict:
            return data_dict[month]
        else:
            return pd.DataFrame()
    
    else:  # All Data
        all_data = pd.DataFrame()
        for month in months_list:
            if month in data_dict:
                if all_data.empty:
                    all_data = data_dict[month].copy()
                else:
                    all_data = pd.concat([all_data, data_dict[month]], ignore_index=True)
        return all_data

def create_plot():
    """Create plot based on selections"""
    data = get_selected_data()
    
    if data.empty:
        return
    
    variable = variable_var.get()
    plot_type = plot_var.get()
      # Find column that matches variable
    col_name = None
    
    # Create mapping for more robust column matching
    if variable == "Max Wind Speed":
        # Look for maximum wind gust speed column
        for col in data.columns:
            if "speed of maximum wind gust" in col.lower():
                col_name = col
                break
    elif variable == "Max Temperature":
        # Look for ALL maximum temperature columns and combine their data
        matching_cols = []
        for col in data.columns:
            if "maximum temperature" in normalize_column_name(col):
                matching_cols.append(col)
        
        if matching_cols:
            # Combine data from all matching columns
            all_temp_data = pd.Series(dtype=float)
            for col in matching_cols:
                col_data = pd.to_numeric(data[col], errors='coerce').dropna()
                all_temp_data = pd.concat([all_temp_data, col_data], ignore_index=True)
            
            if len(all_temp_data) > 0:
                clean_data = all_temp_data
                
                # Clear previous plot
                ax.clear()
                
                if plot_type == "Line Plot":
                    ax.plot(range(len(clean_data)), clean_data, marker='o')
                elif plot_type == "Bar Chart":
                    ax.bar(range(len(clean_data)), clean_data)
                elif plot_type == "Histogram":
                    ax.hist(clean_data, bins=15)
                elif plot_type == "Scatter Plot":
                    ax.scatter(range(len(clean_data)), clean_data)
                
                ax.set_title(variable + " - " + plot_type)
                ax.set_xlabel("Data Points")
                ax.set_ylabel(variable + " (°C)")
                
                canvas.draw()
                # Automatically switch to Plot tab
                notebook.select(0)
                return
    elif variable == "Min Temperature":
        # Look for ALL minimum temperature columns and combine their data
        matching_cols = []
        for col in data.columns:
            if "minimum temperature" in normalize_column_name(col):
                matching_cols.append(col)
        
        if matching_cols:
            # Combine data from all matching columns
            all_temp_data = pd.Series(dtype=float)
            for col in matching_cols:
                col_data = pd.to_numeric(data[col], errors='coerce').dropna()
                all_temp_data = pd.concat([all_temp_data, col_data], ignore_index=True)
            
            if len(all_temp_data) > 0:
                clean_data = all_temp_data
                
                # Clear previous plot
                ax.clear()
                
                if plot_type == "Line Plot":
                    ax.plot(range(len(clean_data)), clean_data, marker='o')
                elif plot_type == "Bar Chart":
                    ax.bar(range(len(clean_data)), clean_data)
                elif plot_type == "Histogram":
                    ax.hist(clean_data, bins=15)
                elif plot_type == "Scatter Plot":
                    ax.scatter(range(len(clean_data)), clean_data)
                
                ax.set_title(variable + " - " + plot_type)
                ax.set_xlabel("Data Points")
                ax.set_ylabel(variable + " (°C)")
                
                canvas.draw()
                # Automatically switch to Plot tab
                notebook.select(0)
                return
    elif variable == "Rainfall":
        # Look for rainfall column
        for col in data.columns:
            if "rainfall" in col.lower():
                col_name = col
                break
    
    # Fallback to original matching if specific mapping didn't work
    if col_name is None:
        for col in data.columns:
            if variable in col or ("temp" in col.lower() and "temperature" in variable.lower()):
                col_name = col
                break
    
    if col_name is None:
        return
    
    # Clean data
    clean_data = pd.to_numeric(data[col_name], errors='coerce').dropna()
    
    # Clear previous plot
    ax.clear()
    
    # Create plot
    if plot_type == "Line Plot":
        ax.plot(range(len(clean_data)), clean_data, marker='o')
    elif plot_type == "Bar Chart":
        ax.bar(range(len(clean_data)), clean_data)
    elif plot_type == "Histogram":
        ax.hist(clean_data, bins=15)
    elif plot_type == "Scatter Plot":
        ax.scatter(range(len(clean_data)), clean_data)
    
    ax.set_title(variable + " - " + plot_type)
    ax.set_xlabel("Data Points")
    
    # Set y-axis label with appropriate units
    if variable == "Max Temperature" or variable == "Min Temperature":
        ax.set_ylabel(variable + " (°C)")
    elif variable == "Rainfall":
        ax.set_ylabel(variable + " (mm)")
    elif variable == "Max Wind Speed":
        ax.set_ylabel(variable + " (km/h)")
    else:
        ax.set_ylabel(variable)
    
    canvas.draw()
    
    # Automatically switch to Plot tab
    notebook.select(0)

def show_stats():
    """Show basic statistics"""
    data = get_selected_data()
    
    if data.empty:
        stats_text.delete(1.0, tk.END)
        stats_text.insert(1.0, "No data available")
        # Automatically switch to Statistics tab
        notebook.select(1)
        return
    
    variable = variable_var.get()
      # Find column
    col_name = None
    
    # Create mapping for more robust column matching
    if variable == "Max Wind Speed":
        # Look for maximum wind gust speed column
        for col in data.columns:
            if "speed of maximum wind gust" in col.lower():
                col_name = col
                break
    elif variable == "Max Temperature":
        # Look for ALL maximum temperature columns and combine their data
        matching_cols = []
        for col in data.columns:
            if "maximum temperature" in normalize_column_name(col):
                matching_cols.append(col)
        
        if matching_cols:
            # Combine data from all matching columns
            all_temp_data = pd.Series(dtype=float)
            for col in matching_cols:
                col_data = pd.to_numeric(data[col], errors='coerce').dropna()
                all_temp_data = pd.concat([all_temp_data, col_data], ignore_index=True)
            
            if len(all_temp_data) > 0:
                clean_data = all_temp_data
                
                stats_info = "STATISTICS\n"
                stats_info = stats_info + "="*20 + "\n\n"
                stats_info = stats_info + "Count: " + str(len(clean_data)) + "\n"
                stats_info = stats_info + "Mean: " + str(round(clean_data.mean(), 2)) + "\n"
                stats_info = stats_info + "Median: " + str(round(clean_data.median(), 2)) + "\n"
                stats_info = stats_info + "Standard Deviation: " + str(round(clean_data.std(), 2)) + "\n"
                stats_info = stats_info + "Minimum: " + str(round(clean_data.min(), 2)) + "\n"
                stats_info = stats_info + "Maximum: " + str(round(clean_data.max(), 2)) + "\n"
                
                stats_text.delete(1.0, tk.END)
                stats_text.insert(1.0, stats_info)
                # Automatically switch to Statistics tab
                notebook.select(1)
                return
    elif variable == "Min Temperature":
        # Look for ALL minimum temperature columns and combine their data
        matching_cols = []
        for col in data.columns:
            if "minimum temperature" in normalize_column_name(col):
                matching_cols.append(col)
        
        if matching_cols:
            # Combine data from all matching columns
            all_temp_data = pd.Series(dtype=float)
            for col in matching_cols:
                col_data = pd.to_numeric(data[col], errors='coerce').dropna()
                all_temp_data = pd.concat([all_temp_data, col_data], ignore_index=True)
            
            if len(all_temp_data) > 0:
                clean_data = all_temp_data
                
                stats_info = "STATISTICS\n"
                stats_info = stats_info + "="*20 + "\n\n"
                stats_info = stats_info + "Count: " + str(len(clean_data)) + "\n"
                stats_info = stats_info + "Mean: " + str(round(clean_data.mean(), 2)) + "\n"
                stats_info = stats_info + "Median: " + str(round(clean_data.median(), 2)) + "\n"
                stats_info = stats_info + "Standard Deviation: " + str(round(clean_data.std(), 2)) + "\n"
                stats_info = stats_info + "Minimum: " + str(round(clean_data.min(), 2)) + "\n"
                stats_info = stats_info + "Maximum: " + str(round(clean_data.max(), 2)) + "\n"
                
                stats_text.delete(1.0, tk.END)
                stats_text.insert(1.0, stats_info)
                # Automatically switch to Statistics tab
                notebook.select(1)
                return
    elif variable == "Rainfall":
        # Look for rainfall column
        for col in data.columns:
            if "rainfall" in col.lower():
                col_name = col
                break
    
    # Fallback to original matching if specific mapping didn't work
    if col_name is None:
        for col in data.columns:
            if variable in col or ("temp" in col.lower() and "temperature" in variable.lower()):
                col_name = col
                break
    
    if col_name is None:
        stats_text.delete(1.0, tk.END)
        stats_text.insert(1.0, "Variable not found")
        # Automatically switch to Statistics tab
        notebook.select(1)
        return
    
    # Clean data and calculate stats
    clean_data = pd.to_numeric(data[col_name], errors='coerce').dropna()
    
    stats_info = "STATISTICS\n"
    stats_info = stats_info + "="*20 + "\n\n"
    stats_info = stats_info + "Count: " + str(len(clean_data)) + "\n"
    stats_info = stats_info + "Mean: " + str(round(clean_data.mean(), 2)) + "\n"
    stats_info = stats_info + "Std Dev: " + str(round(clean_data.std(), 2)) + "\n"
    stats_info = stats_info + "Min: " + str(round(clean_data.min(), 2)) + "\n"
    stats_info = stats_info + "Max: " + str(round(clean_data.max(), 2)) + "\n"
    stats_info = stats_info + "Median: " + str(round(clean_data.median(), 2)) + "\n"
    
    stats_text.delete(1.0, tk.END)
    stats_text.insert(1.0, stats_info)
    
    # Automatically switch to Statistics tab
    notebook.select(1)

def clear_plot():
    """Clear the plot"""
    ax.clear()
    ax.text(0.5, 0.5, "Plot Cleared", ha='center', va='center', transform=ax.transAxes)
    canvas.draw()

def update_display_controls():
    """Show/hide controls based on display mode"""
    display_choice = display_var.get()
    
    if display_choice == "Single Month":
        # Show month selection in correct position (before variable frame)
        month_frame.pack(fill=tk.X, padx=10, pady=5, before=var_frame)
    else:  # All Data
        # Hide month selection
        month_frame.pack_forget()

# Load data first
load_csv_files()

# Create main window
root = tk.Tk()
root.title("Weather Data Viewer")
root.geometry("1000x700")

# Main frame
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Control panel
control_frame = tk.LabelFrame(main_frame, text="Controls")
control_frame.pack(fill=tk.X, pady=(0, 10))

# Display options
display_frame = tk.Frame(control_frame)
display_frame.pack(fill=tk.X, padx=10, pady=5)

tk.Label(display_frame, text="Display:").grid(row=0, column=0, padx=5)

display_var = tk.StringVar(value="Single Month")
tk.Radiobutton(display_frame, text="Single Month", variable=display_var, value="Single Month", command=update_display_controls).grid(row=0, column=1, padx=5)
tk.Radiobutton(display_frame, text="All Data", variable=display_var, value="All Data", command=update_display_controls).grid(row=0, column=2, padx=5)

# Month selection
month_frame = tk.Frame(control_frame)
month_frame.pack(fill=tk.X, padx=10, pady=5)

tk.Label(month_frame, text="Month:").grid(row=0, column=0, padx=5)
month_var = tk.StringVar(value="24-05")
month_combo = ttk.Combobox(month_frame, textvariable=month_var, values=months_list, state="readonly")
month_combo.grid(row=0, column=1, padx=5)

# Variable selection
var_frame = tk.Frame(control_frame)
var_frame.pack(fill=tk.X, padx=10, pady=5)

tk.Label(var_frame, text="Variable:").grid(row=0, column=0, padx=5)
variable_var = tk.StringVar(value="Max Temperature")
var_combo = ttk.Combobox(var_frame, textvariable=variable_var, values=["Max Temperature", "Min Temperature", "Rainfall", "Max Wind Speed"], state="readonly")
var_combo.grid(row=0, column=1, padx=5)

tk.Label(var_frame, text="Plot Type:").grid(row=0, column=2, padx=5)
plot_var = tk.StringVar(value="Line Plot")
plot_combo = ttk.Combobox(var_frame, textvariable=plot_var, values=["Line Plot", "Bar Chart", "Histogram", "Scatter Plot"], state="readonly")
plot_combo.grid(row=0, column=3, padx=5)

# Buttons - positioned just above the chart/notebook area
button_frame = tk.Frame(main_frame)
button_frame.pack(fill=tk.X, padx=10, pady=5)

tk.Button(button_frame, text="Create Plot", command=create_plot).pack(side=tk.LEFT, padx=5)
tk.Button(button_frame, text="Show Stats", command=show_stats).pack(side=tk.LEFT, padx=5)
tk.Button(button_frame, text="Clear Plot", command=clear_plot).pack(side=tk.LEFT, padx=5)

# Create notebook for plot and stats
notebook = ttk.Notebook(main_frame)
notebook.pack(fill=tk.BOTH, expand=True)

# Plot tab
plot_frame = tk.Frame(notebook)
notebook.add(plot_frame, text="Plot")

# Setup plot
fig, ax = plt.subplots(figsize=(8, 5))
canvas = FigureCanvasTkAgg(fig, plot_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Stats tab
stats_frame = tk.Frame(notebook)
notebook.add(stats_frame, text="Statistics")

# Setup stats text
stats_text = tk.Text(stats_frame, font=("Courier", 10))
stats_scrollbar = tk.Scrollbar(stats_frame, orient=tk.VERTICAL, command=stats_text.yview)
stats_text.configure(yscrollcommand=stats_scrollbar.set)

stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Show initial message
ax.text(0.5, 0.5, "Click 'Create Plot' to start", ha='center', va='center', transform=ax.transAxes)
canvas.draw()

# Initialize UI controls
update_display_controls()

# Run the application
root.mainloop()