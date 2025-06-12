# Simple Weather Data Visualization GUI

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg # use for embedding matplotlib plots in tkinter
import pandas as pd
import os # Used for relative filepaths

data_dict = {}
months_list = ["24-05", "24-06", "24-07", "24-08", "24-09", "24-10", "24-11", "24-12", "25-01", "25-02", "25-03", "25-04"]
csv_folder = os.path.join(os.path.dirname(__file__), "..", "Datasets") # Relative file paths used

def normalize_column_name(col_name): # Clean up column names by removing encoding artifacts and converting to lowercase
    """Handle encoding artifacts in column names"""
    return str(col_name).lower().replace('ï¿½', 'o').replace('°', 'o') # Removes abnormal naming conventions in csv headers

def load_csv_files(): # Load all monthly CSV files from the datasets folder into data_dict
    """Load CSV files from datasets folder"""
    for month in months_list:
        try:
            filename = month + ".csv" # String concatenation
            full_path = os.path.join(csv_folder, filename)
            df = pd.read_csv(full_path, skiprows=7, encoding='latin-1') # Skip 7 rows
            data_dict[month] = df # Stores in dataframe using month str as key
            print("Loaded:", filename)
        except Exception as e:
            print("Could not load:", filename, "Error:", str(e))

def get_selected_data(): # Get data based on user's display choice (single month or all data combined)
    """Get data based on selection"""
    display_choice = display_var.get() # Gets display choice in GUI
    
    if display_choice == "Single Month": # If user selects single month
        month = month_var.get()
        if month in data_dict: # Checks if month is in data_dict
            return data_dict[month] # copy the dataframe for that month
        else:
            return pd.DataFrame() # Empty DataFrame if month not found
    
    else:  # All Data
        all_data = pd.DataFrame()
        for month in months_list: 
            if month in data_dict:
                if all_data.empty: # If all_data  empty, copy the first month data
                    all_data = data_dict[month].copy()
                else: # Concatenate the data for all months in DataFrame
                    all_data = pd.concat([all_data, data_dict[month]], ignore_index=True)
        return all_data

def combine_temperature_columns(data, temp_type): # Combine multiple temperature columns of same type (max/min) into single series
    """Helper function to combine temperature columns"""
    matching_cols = []
    search_term = f"{temp_type.lower()} temperature"
    
    for col in data.columns:
        if search_term in normalize_column_name(col):
            matching_cols.append(col)
    
    if matching_cols:
        all_temp_data = pd.Series(dtype=float)
        for col in matching_cols:
            col_data = pd.to_numeric(data[col], errors='coerce').dropna()
            all_temp_data = pd.concat([all_temp_data, col_data], ignore_index=True)
        return all_temp_data
    return None

def get_variable_data(data, variable): # Extract and clean data for a specific variable (temperature, rainfall, wind speed)
    """Helper function to extract and clean data for a specific variable"""
    col_name = None
    clean_data = None
    
    # Handle temperature variables with combined columns
    if variable in ["Max Temperature", "Min Temperature"]:
        temp_type = "maximum" if variable == "Max Temperature" else "minimum"
        clean_data = combine_temperature_columns(data, temp_type)
        return clean_data
    
    # Handle other variables
    elif variable == "Max Wind Speed":
        for col in data.columns:
            if "speed of maximum wind gust" in col.lower(): 
                col_name = col
                break
    elif variable == "Rainfall":
        for col in data.columns:
            if "rainfall" in col.lower():
                col_name = col
                break
    
    # Fallback matching
    if col_name is None:
        for col in data.columns:
            if variable in col or ("temp" in col.lower() and "temperature" in variable.lower()):
                col_name = col
                break
    
    if col_name is None:
        return None
    
    # Clean data and return
    clean_data = pd.to_numeric(data[col_name], errors='coerce').dropna()
    return clean_data

def create_plot_visualization(clean_data, plot_type, variable): # Create matplotlib plot with proper titles, labels, and units
    """Helper function to create the actual plot visualization"""
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
    
    # Set y-axis label with units
    if variable == "Max Temperature" or variable == "Min Temperature":
        ax.set_ylabel(variable + " (°C)")
    elif variable == "Rainfall":
        ax.set_ylabel(variable + " (mm)")
    elif variable == "Max Wind Speed":
        ax.set_ylabel(variable + " (km/h)")
    else:
        ax.set_ylabel(variable)
    
    canvas.draw()
    notebook.select(0)

def create_plot(): # Main function to create and display plots based on user selections
    """Create plot based on selections"""
    data = get_selected_data()
    
    if data.empty:
        messagebox.showwarning("No Data", "No data available for the selected month or variable.")
        return
    
    variable = variable_var.get()
    plot_type = plot_var.get()
    
    # Get cleaned data for the selected variable
    clean_data = get_variable_data(data, variable)
    
    if clean_data is None or len(clean_data) == 0:
        messagebox.showwarning("No Data", f"No data available for {variable}.")
        return
    
    # Create the plot visualization
    create_plot_visualization(clean_data, plot_type, variable)

def generate_stats_text(clean_data, variable): # Generate formatted statistics text with calculations like mean, median, std dev
    """Helper function to generate statistics text"""
    stats_info = f"STATISTICS - {variable}\n"
    stats_info = stats_info + "="*30 + "\n\n"
    stats_info = stats_info + "Count: " + str(len(clean_data)) + "\n"
    stats_info = stats_info + "Mean: " + str(round(clean_data.mean(), 2)) + "\n"
    stats_info = stats_info + "Standard Deviation: " + str(round(clean_data.std(), 2)) + "\n\n"
    
    # 5-Number Summary
    stats_info = stats_info + "5-NUMBER SUMMARY:\n"
    stats_info = stats_info + "-" * 20 + "\n"
    stats_info = stats_info + "Minimum: " + str(round(clean_data.min(), 2)) + "\n"
    stats_info = stats_info + "Q1 (25th percentile): " + str(round(clean_data.quantile(0.25), 2)) + "\n"
    stats_info = stats_info + "Median (Q2): " + str(round(clean_data.median(), 2)) + "\n"
    stats_info = stats_info + "Q3 (75th percentile): " + str(round(clean_data.quantile(0.75), 2)) + "\n"
    stats_info = stats_info + "Maximum: " + str(round(clean_data.max(), 2)) + "\n"
    return stats_info

def display_stats(stats_info): # Display formatted statistics in the GUI text widget and switch to stats tab
    """Helper function to display statistics in the text widget"""
    stats_text.delete(1.0, tk.END)
    stats_text.insert(1.0, stats_info)
    notebook.select(1)

def show_stats(): # Calculate and display statistical summary for selected variable and data range
    """Show basic statistics"""
    data = get_selected_data()
    
    if data.empty:
        stats_text.delete(1.0, tk.END)
        stats_text.insert(1.0, "No data available")
        notebook.select(1)
        return
    
    variable = variable_var.get()
    
    # Get cleaned data for the selected variable
    clean_data = get_variable_data(data, variable)
    
    if clean_data is None or len(clean_data) == 0:
        stats_text.delete(1.0, tk.END)
        stats_text.insert(1.0, f"No data available for {variable}")
        notebook.select(1)
        return
    
    # Generate and display statistics
    stats_info = generate_stats_text(clean_data, variable)
    display_stats(stats_info)

def clear_plot(): # Clear the plot canvas and display a "Plot Cleared" message
    """Clear the plot"""
    ax.clear()
    ax.text(0.5, 0.5, "Plot Cleared", ha='center', va='center', transform=ax.transAxes)
    canvas.draw()

def update_display_controls(): # Show or hide month selection controls based on display mode choice
    """Show/hide controls based on display mode"""
    display_choice = display_var.get()
    
    if display_choice == "Single Month":
        month_frame.pack(fill=tk.X, padx=10, pady=5, before=var_frame)
    else:
        month_frame.pack_forget()

def create_gui(): # Create and configure the complete GUI interface with all widgets and event handlers
    """Create and setup the main GUI"""
    global root, main_frame, notebook, ax, canvas, stats_text
    global display_var, month_var, variable_var, plot_var, month_frame, var_frame
    
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

    # Buttons
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

def main(): # Main function to run the application - loads data and starts GUI
    """Main function to run the application"""
    # Load data first
    load_csv_files()
    
    # Create and run GUI
    create_gui()
    root.mainloop()

# Run the application
if __name__ == "__main__":
    main()