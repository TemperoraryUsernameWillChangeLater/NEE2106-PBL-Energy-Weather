# Simple Weather Data Visualization GUI
# Using only syntax from example syntax.py file

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import os

# Simple variables to store data
data_dict = {}
months_list = ["24-05", "24-06", "24-07", "24-08", "24-09", "24-10", "24-11", "24-12", "25-01", "25-02", "25-03", "25-04"] # Hardcoded var names for months
days_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"]
years_list = ["2024", "2025"]

# Get the absolute path to the CSV files
csv_folder = r"c:\Users\gabri\Documents\Python\(NEE2106) Computer Programming For Electrical Engineers\PBL Project - Energy and Weather\GUI" # Has to be hardcoded due 

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
    
    elif display_choice == "All Data":
        all_data = pd.DataFrame()
        for month in months_list:
            if month in data_dict:
                if all_data.empty:
                    all_data = data_dict[month].copy()
                else:
                    all_data = pd.concat([all_data, data_dict[month]], ignore_index=True)
        return all_data
    
    else:  # Date Range
        # For simplicity, just return first month data
        month = month_var.get()
        if month in data_dict:
            return data_dict[month]
        else:
            return pd.DataFrame()

def create_plot():
    """Create plot based on selections"""
    data = get_selected_data()
    
    if data.empty:
        return
    
    variable = variable_var.get()
    plot_type = plot_var.get()
    
    # Find column that matches variable
    col_name = None
    for col in data.columns:
        if variable in col or "temp" in col.lower() and "temp" in variable.lower():
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
    ax.set_ylabel(variable)
    
    canvas.draw()

def show_stats():
    """Show basic statistics"""
    data = get_selected_data()
    
    if data.empty:
        stats_text.delete(1.0, tk.END)
        stats_text.insert(1.0, "No data available")
        return
    
    variable = variable_var.get()
    
    # Find column
    col_name = None
    for col in data.columns:
        if variable in col or "temp" in col.lower() and "temp" in variable.lower():
            col_name = col
            break
    
    if col_name is None:
        stats_text.delete(1.0, tk.END)
        stats_text.insert(1.0, "Variable not found")
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

def clear_plot():
    """Clear the plot"""
    ax.clear()
    ax.text(0.5, 0.5, "Plot Cleared", ha='center', va='center', transform=ax.transAxes)
    canvas.draw()

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
tk.Radiobutton(display_frame, text="Single Month", variable=display_var, value="Single Month").grid(row=0, column=1, padx=5)
tk.Radiobutton(display_frame, text="All Data", variable=display_var, value="All Data").grid(row=0, column=2, padx=5)
tk.Radiobutton(display_frame, text="Date Range", variable=display_var, value="Date Range").grid(row=0, column=3, padx=5)

# Month selection
month_frame = tk.Frame(control_frame)
month_frame.pack(fill=tk.X, padx=10, pady=5)

tk.Label(month_frame, text="Month:").grid(row=0, column=0, padx=5)
month_var = tk.StringVar(value="24-05")
month_combo = ttk.Combobox(month_frame, textvariable=month_var, values=months_list, state="readonly")
month_combo.grid(row=0, column=1, padx=5)

# Date range selection (day/month/year dropdowns)
date_frame = tk.Frame(control_frame)
date_frame.pack(fill=tk.X, padx=10, pady=5)

tk.Label(date_frame, text="Start Date:").grid(row=0, column=0, padx=5)

# Start date dropdowns
start_day_var = tk.StringVar(value="01")
start_day_combo = ttk.Combobox(date_frame, textvariable=start_day_var, values=days_list, state="readonly", width=5)
start_day_combo.grid(row=0, column=1, padx=2)

start_month_var = tk.StringVar(value="05")
start_month_combo = ttk.Combobox(date_frame, textvariable=start_month_var, values=["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"], state="readonly", width=5)
start_month_combo.grid(row=0, column=2, padx=2)

start_year_var = tk.StringVar(value="2024")
start_year_combo = ttk.Combobox(date_frame, textvariable=start_year_var, values=years_list, state="readonly", width=8)
start_year_combo.grid(row=0, column=3, padx=2)

tk.Label(date_frame, text="End Date:").grid(row=0, column=4, padx=(10, 5))

# End date dropdowns
end_day_var = tk.StringVar(value="31")
end_day_combo = ttk.Combobox(date_frame, textvariable=end_day_var, values=days_list, state="readonly", width=5)
end_day_combo.grid(row=0, column=5, padx=2)

end_month_var = tk.StringVar(value="12")
end_month_combo = ttk.Combobox(date_frame, textvariable=end_month_var, values=["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"], state="readonly", width=5)
end_month_combo.grid(row=0, column=6, padx=2)

end_year_var = tk.StringVar(value="2024")
end_year_combo = ttk.Combobox(date_frame, textvariable=end_year_var, values=years_list, state="readonly", width=8)
end_year_combo.grid(row=0, column=7, padx=2)

# Variable selection
var_frame = tk.Frame(control_frame)
var_frame.pack(fill=tk.X, padx=10, pady=5)

tk.Label(var_frame, text="Variable:").grid(row=0, column=0, padx=5)
variable_var = tk.StringVar(value="Temperature")
var_combo = ttk.Combobox(var_frame, textvariable=variable_var, values=["Temperature", "Rainfall", "Humidity"], state="readonly")
var_combo.grid(row=0, column=1, padx=5)

tk.Label(var_frame, text="Plot Type:").grid(row=0, column=2, padx=5)
plot_var = tk.StringVar(value="Line Plot")
plot_combo = ttk.Combobox(var_frame, textvariable=plot_var, values=["Line Plot", "Bar Chart", "Histogram", "Scatter Plot"], state="readonly")
plot_combo.grid(row=0, column=3, padx=5)

# Buttons
button_frame = tk.Frame(control_frame)
button_frame.pack(fill=tk.X, padx=10, pady=10)

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

# Run the application
root.mainloop()
