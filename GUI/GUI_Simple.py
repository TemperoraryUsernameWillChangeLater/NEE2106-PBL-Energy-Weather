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
        try:
            # Check if all date fields are filled
            if not all([start_day_var.get(), start_month_var.get(), start_year_var.get(),
                       end_day_var.get(), end_month_var.get(), end_year_var.get()]):
                print("Please select complete start and end dates")
                return pd.DataFrame()
              # Get start and end dates
            start_day = int(start_day_var.get())
            start_month = int(start_month_var.get())
            start_year = int(start_year_var.get())
            end_day = int(end_day_var.get())
            end_month = int(end_month_var.get())
            end_year = int(end_year_var.get())
              # Validate that start date is not greater than end date
            try:
                from datetime import datetime
                start_date = datetime(start_year, start_month, start_day)
                end_date = datetime(end_year, end_month, end_day)
                
                if start_date > end_date:
                    messagebox.showerror("Invalid Date Range", 
                                       f"Start date ({start_date.strftime('%Y-%m-%d')}) cannot be greater than end date ({end_date.strftime('%Y-%m-%d')}).\n\nPlease select a valid date range.")
                    return pd.DataFrame()
                    
            except ValueError as date_error:
                messagebox.showerror("Invalid Date", 
                                   f"Invalid date selected: {str(date_error)}\n\nPlease check your date selections and ensure they are valid dates.")
                return pd.DataFrame()
            
            # Convert to month format used in data_dict
            start_month_str = str(start_year)[-2:] + "-" + str(start_month).zfill(2)
            end_month_str = str(end_year)[-2:] + "-" + str(end_month).zfill(2)
            
            filtered_data = pd.DataFrame()
            
            # If same month, filter by day range
            if start_month_str == end_month_str and start_month_str in data_dict:
                month_data = data_dict[start_month_str].copy()
                # Filter by day range (assuming row index corresponds to day)
                start_idx = max(0, start_day - 1)
                end_idx = min(len(month_data), end_day)
                filtered_data = month_data.iloc[start_idx:end_idx]
            
            # If different months, gt data from multiple monthse
            else:
                for month in months_list:
                    if month in data_dict:
                        month_data = data_dict[month].copy()
                        
                        # Check if month is in range
                        month_num = int(month.split("-")[1])
                        year_num = int("20" + month.split("-")[0])
                        
                        if (year_num > start_year or (year_num == start_year and month_num >= start_month)) and \
                           (year_num < end_year or (year_num == end_year and month_num <= end_month)):
                            
                            # Apply day filtering for start and end months
                            if month == start_month_str:
                                start_idx = max(0, start_day - 1)
                                month_data = month_data.iloc[start_idx:]
                            elif month == end_month_str:
                                end_idx = min(len(month_data), end_day)
                                month_data = month_data.iloc[:end_idx]
                            
                            if filtered_data.empty:
                                filtered_data = month_data.copy()
                            else:
                                filtered_data = pd.concat([filtered_data, month_data], ignore_index=True)
            
            return filtered_data
            
        except Exception as e:
            print("Error in date range filtering:", str(e))
            # Fallback to single month
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
    
    # Create mapping for more robust column matching
    if variable == "Max Wind Speed":
        # Look for maximum wind gust speed column
        for col in data.columns:
            if "speed of maximum wind gust" in col.lower():
                col_name = col
                break
    elif variable == "Max Temperature":
        # Look for maximum temperature column
        for col in data.columns:
            if "maximum temperature" in col.lower():
                col_name = col
                break
    elif variable == "Min Temperature":
        # Look for minimum temperature column
        for col in data.columns:
            if "minimum temperature" in col.lower():
                col_name = col
                break
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
        # Automatically switch to Statistics tab        notebook.select(1)
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
        # Look for maximum temperature column
        for col in data.columns:
            if "maximum temperature" in col.lower():
                col_name = col
                break
    elif variable == "Min Temperature":
        # Look for minimum temperature column
        for col in data.columns:
            if "minimum temperature" in col.lower():
                col_name = col
                break
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
        # Show month selection, hide date range
        month_frame.pack(fill=tk.X, padx=10, pady=5)
        date_frame.pack_forget()
    elif display_choice == "All Data":
        # Hide both month selection and date range
        month_frame.pack_forget()
        date_frame.pack_forget()
    else:  # Date Range
        # Hide month selection, show date range
        month_frame.pack_forget()
        date_frame.pack(fill=tk.X, padx=10, pady=5)

def get_month_max_days(month_str):
    """Get max days for a month based on data rows"""
    if month_str in data_dict:
        return len(data_dict[month_str])
    
    # If month doesn't exist in data, return 0 to prevent selection
    print(f"Warning: Month {month_str} not found in data")
    return 0

# Global variable to keep track of timer
info_label_timer = None

def clear_info_label():
    """Clear the dynamic info label"""
    info_label_var.set("")

def set_info_label_with_timer(message):
    """Set info label message and schedule it to clear after 3 seconds"""
    global info_label_timer
    
    # Cancel any existing timer
    if info_label_timer:
        root.after_cancel(info_label_timer)
    
    # Set the message
    info_label_var.set(message)
    
    # Schedule clearing after 3 seconds (3000 milliseconds)
    info_label_timer = root.after(3000, clear_info_label)

def update_start_year_months(*args):
    """Update start month options when start year changes"""
    try:
        start_year = start_year_var.get()
        available_start_months = []
        
        if start_year:  # Only update if year is selected
            for month_num in range(1, 13):
                month_str = start_year[-2:] + "-" + str(month_num).zfill(2)
                if month_str in data_dict:
                    available_start_months.append(str(month_num).zfill(2))
            
            start_month_combo['values'] = available_start_months
            if start_month_var.get() not in available_start_months and available_start_months:
                start_month_var.set("")  # Clear selection instead of auto-selecting
            
            # Update dynamic info label for start date
            if available_start_months:
                months_text = ", ".join(available_start_months)
                set_info_label_with_timer(f"Start Year {start_year}: months {months_text} available")
            else:
                set_info_label_with_timer(f"Start Year {start_year}: No data available")
        else:
            start_month_combo['values'] = []
            start_month_var.set("")
            
        print(f"Updated start year months - {start_year}: {available_start_months}")
        
    except Exception as e:
        print("Error updating start year months:", str(e))

def update_end_year_months(*args):
    """Update end month options when end year changes"""
    try:
        end_year = end_year_var.get()
        available_end_months = []
        
        if end_year:  # Only update if year is selected
            for month_num in range(1, 13):
                month_str = end_year[-2:] + "-" + str(month_num).zfill(2)
                if month_str in data_dict:
                    available_end_months.append(str(month_num).zfill(2))
            
            end_month_combo['values'] = available_end_months
            if end_month_var.get() not in available_end_months and available_end_months:
                end_month_var.set("")  # Clear selection instead of auto-selecting
            
            # Update dynamic info label for end date
            if available_end_months:
                months_text = ", ".join(available_end_months)
                set_info_label_with_timer(f"End Year {end_year}: months {months_text} available")
            else:
                set_info_label_with_timer(f"End Year {end_year}: No data available")
        else:
            end_month_combo['values'] = []
            end_month_var.set("")
            
        print(f"Updated end year months - {end_year}: {available_end_months}")
        
    except Exception as e:
        print("Error updating end year months:", str(e))

def update_available_months(*args):
    """Update available months based on selected year (legacy function for compatibility)"""
    # This function is kept for any remaining bindings, but the work is now split
    update_start_year_months()
    update_end_year_months()

def update_day_limits(*args):
    """Update day dropdown limits based on selected months"""
    try:
        # Update start day limits
        start_month = start_month_var.get()
        start_year = start_year_var.get()
        
        if start_month and start_year:
            start_month_str = start_year[-2:] + "-" + start_month.zfill(2)
            start_max_days = get_month_max_days(start_month_str)
            
            if start_max_days == 0:
                start_day_combo['values'] = ["--"]
                start_day_var.set("--")
            else:
                start_day_values = [str(i).zfill(2) for i in range(1, start_max_days + 1)]
                start_day_combo['values'] = start_day_values
                # Adjust current selection if it exceeds limits
                current_start_day = int(start_day_var.get()) if start_day_var.get().isdigit() else 1
                if current_start_day > start_max_days:
                    start_day_var.set(str(start_max_days).zfill(2))
                elif not start_day_var.get():
                    start_day_var.set("01")  # Set default if empty
        else:
            start_day_combo['values'] = []
            start_day_var.set("")
        
        # Update end day limits  
        end_month = end_month_var.get()
        end_year = end_year_var.get()
        
        if end_month and end_year:
            end_month_str = end_year[-2:] + "-" + end_month.zfill(2)
            end_max_days = get_month_max_days(end_month_str)
            
            if end_max_days == 0:
                end_day_combo['values'] = ["--"]
                end_day_var.set("--")
            else:
                end_day_values = [str(i).zfill(2) for i in range(1, end_max_days + 1)]
                end_day_combo['values'] = end_day_values
                # Adjust current selection if it exceeds limits
                current_end_day = int(end_day_var.get()) if end_day_var.get().isdigit() else 1
                if current_end_day > end_max_days:
                    end_day_var.set(str(end_max_days).zfill(2))
                elif not end_day_var.get():
                    end_day_var.set("01")  # Set default if empty
        else:
            end_day_combo['values'] = []
            end_day_var.set("")
            
        # Debug output
        if start_month and start_year and end_month and end_year:
            start_month_str = start_year[-2:] + "-" + start_month.zfill(2)
            end_month_str = end_year[-2:] + "-" + end_month.zfill(2)
            start_max_days = get_month_max_days(start_month_str)
            end_max_days = get_month_max_days(end_month_str)
            print(f"Updated day limits - Start: {start_month_str} -> max {start_max_days} days, End: {end_month_str} -> max {end_max_days} days")
            
    except Exception as e:
        print("Error updating day limits:", str(e))
        print("Debug info:")
        print("start_month:", start_month_var.get())
        print("start_year:", start_year_var.get())
        print("end_month:", end_month_var.get())
        print("end_year:", end_year_var.get())

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
tk.Radiobutton(display_frame, text="Date Range", variable=display_var, value="Date Range", command=update_display_controls).grid(row=0, column=3, padx=5)

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

tk.Label(date_frame, text="Start Date (YYYY-MM-DD):").grid(row=0, column=0, padx=5)

# Start date dropdowns - Year, Month, Day order
start_year_var = tk.StringVar(value="")
start_year_combo = ttk.Combobox(date_frame, textvariable=start_year_var, values=years_list, state="readonly", width=8)
start_year_combo.grid(row=0, column=1, padx=2)

start_month_var = tk.StringVar(value="")
start_month_combo = ttk.Combobox(date_frame, textvariable=start_month_var, values=[], state="readonly", width=5)
start_month_combo.grid(row=0, column=2, padx=2)

start_day_var = tk.StringVar(value="")
start_day_combo = ttk.Combobox(date_frame, textvariable=start_day_var, values=[], state="readonly", width=5)
start_day_combo.grid(row=0, column=3, padx=2)

tk.Label(date_frame, text="End Date (YYYY-MM-DD):").grid(row=0, column=4, padx=(10, 5))

# End date dropdowns - Year, Month, Day order
end_year_var = tk.StringVar(value="")
end_year_combo = ttk.Combobox(date_frame, textvariable=end_year_var, values=years_list, state="readonly", width=8)
end_year_combo.grid(row=0, column=5, padx=2)

end_month_var = tk.StringVar(value="")
end_month_combo = ttk.Combobox(date_frame, textvariable=end_month_var, values=[], state="readonly", width=5)
end_month_combo.grid(row=0, column=6, padx=2)

end_day_var = tk.StringVar(value="")
end_day_combo = ttk.Combobox(date_frame, textvariable=end_day_var, values=[], state="readonly", width=5)
end_day_combo.grid(row=0, column=7, padx=2)

# Dynamic info label
info_label_var = tk.StringVar(value="")
info_label = tk.Label(date_frame, textvariable=info_label_var, fg="blue", font=("Arial", 9))
info_label.grid(row=0, column=8, padx=(10, 5), sticky="w")

# Add event bindings for dynamic updates
start_year_var.trace('w', update_start_year_months)
start_year_var.trace('w', update_day_limits)
start_month_var.trace('w', update_day_limits)
end_year_var.trace('w', update_end_year_months)
end_year_var.trace('w', update_day_limits)
end_month_var.trace('w', update_day_limits)

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

# Initialize UI controls and day limits
update_display_controls()
update_start_year_months()
update_end_year_months()
update_day_limits()

# Run the application
root.mainloop()
