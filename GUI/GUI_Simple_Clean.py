# Simple Weather Data Visualization GUI

import tkinter as tk  # Import tkinter for GUI creation
from tkinter import ttk  # Import themed tkinter widgets
from tkinter import messagebox  # Import message box for alerts
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Import matplotlib canvas for tkinter
import pandas as pd  # Import pandas for data manipulation
import os  # Import os for file path operations

# Simple variables to store data
data_dict = {}  # Dictionary to store loaded CSV data for each month
months_list = ["24-05", "24-06", "24-07", "24-08", "24-09", "24-10", "24-11", "24-12", "25-01", "25-02", "25-03", "25-04"]  # List of available months

# Get the relative path to the CSV files
csv_folder = os.path.join(os.path.dirname(__file__), "..", "Datasets")  # Create path to Datasets folder relative to this script

def normalize_column_name(col_name):
    """Normalize column names to handle encoding artifacts"""
    return str(col_name).lower().replace('ï¿½', 'o').replace('°', 'o')  # Convert to lowercase and replace encoding artifacts with 'o'

def load_csv_files():
    """Load CSV files"""
    for month in months_list:  # Loop through each month in the list
        try:  # Try to load each file
            filename = month + ".csv"  # Create filename by adding .csv extension
            full_path = os.path.join(csv_folder, filename)  # Create full file path
            df = pd.read_csv(full_path, skiprows=7, encoding='latin-1')  # Read CSV file, skip first 7 rows, use latin-1 encoding
            data_dict[month] = df  # Store dataframe in dictionary with month as key
            print("Loaded:", filename)  # Print success message
        except Exception as e:  # If loading fails
            print("Could not load:", filename, "Error:", str(e))  # Print error message

def get_selected_data():
    """Get data based on selection"""
    display_choice = display_var.get()  # Get the current display choice (Single Month or All Data)
    
    if display_choice == "Single Month":  # If user selected single month
        month = month_var.get()  # Get the selected month
        if month in data_dict:  # If the month data exists
            return data_dict[month]  # Return the dataframe for that month
        else:  # If month data doesn't exist
            return pd.DataFrame()  # Return empty dataframe
    
    else:  # All Data - if user selected all data
        all_data = pd.DataFrame()  # Create empty dataframe to store combined data
        for month in months_list:  # Loop through each month
            if month in data_dict:  # If data exists for this month
                if all_data.empty:  # If combined dataframe is empty
                    all_data = data_dict[month].copy()  # Copy first month's data
                else:  # If combined dataframe already has data
                    all_data = pd.concat([all_data, data_dict[month]], ignore_index=True)  # Concatenate with existing data
        return all_data  # Return combined dataframe

def create_plot():
    """Create plot based on selections"""
    data = get_selected_data()  # Get the data based on user selection
    
    if data.empty:  # If no data is available
        return  # Exit function early
    
    variable = variable_var.get()  # Get selected variable to plot
    plot_type = plot_var.get()  # Get selected plot type
    col_name = None  # Initialize column name variable
    
    # Create mapping for more robust column matching
    if variable == "Max Wind Speed":  # If user selected max wind speed
        # Look for maximum wind gust speed column
        for col in data.columns:  # Loop through all column names
            if "speed of maximum wind gust" in col.lower():  # Check if column contains wind speed info
                col_name = col  # Set the matching column name
                break  # Stop searching once found
    elif variable == "Max Temperature":  # If user selected max temperature
        # Look for ALL maximum temperature columns and combine their data
        matching_cols = []  # List to store matching column names
        for col in data.columns:  # Loop through all columns
            if "maximum temperature" in normalize_column_name(col):  # Check if column contains max temperature
                matching_cols.append(col)  # Add column to matching list
        
        if matching_cols:  # If matching columns were found
            # Combine data from all matching columns
            all_temp_data = pd.Series(dtype=float)  # Create empty series for temperature data
            for col in matching_cols:  # Loop through matching columns
                col_data = pd.to_numeric(data[col], errors='coerce').dropna()  # Convert to numeric, remove invalid data
                all_temp_data = pd.concat([all_temp_data, col_data], ignore_index=True)  # Combine with existing data
            
            if len(all_temp_data) > 0:  # If we have valid data
                clean_data = all_temp_data  # Use the combined temperature data
                
                # Clear previous plot
                ax.clear()  # Clear the matplotlib axes
                
                if plot_type == "Line Plot":  # If line plot selected
                    ax.plot(range(len(clean_data)), clean_data, marker='o')  # Create line plot with markers
                elif plot_type == "Bar Chart":  # If bar chart selected
                    ax.bar(range(len(clean_data)), clean_data)  # Create bar chart
                elif plot_type == "Histogram":  # If histogram selected
                    ax.hist(clean_data, bins=15)  # Create histogram with 15 bins
                elif plot_type == "Scatter Plot":  # If scatter plot selected
                    ax.scatter(range(len(clean_data)), clean_data)  # Create scatter plot
                
                ax.set_title(variable + " - " + plot_type)  # Set plot title
                ax.set_xlabel("Data Points")  # Set x-axis label
                ax.set_ylabel(variable + " (°C)")  # Set y-axis label with temperature units
                
                canvas.draw()  # Refresh the plot canvas
                # Automatically switch to Plot tab
                notebook.select(0)  # Switch to the first tab (Plot tab)
                return  # Exit function
    elif variable == "Min Temperature":  # If user selected min temperature
        # Look for ALL minimum temperature columns and combine their data
        matching_cols = []  # List to store matching column names
        for col in data.columns:  # Loop through all columns
            if "minimum temperature" in normalize_column_name(col):  # Check if column contains min temperature
                matching_cols.append(col)  # Add column to matching list
        
        if matching_cols:  # If matching columns were found
            # Combine data from all matching columns
            all_temp_data = pd.Series(dtype=float)  # Create empty series for temperature data
            for col in matching_cols:  # Loop through matching columns
                col_data = pd.to_numeric(data[col], errors='coerce').dropna()  # Convert to numeric, remove invalid data
                all_temp_data = pd.concat([all_temp_data, col_data], ignore_index=True)  # Combine with existing data
            
            if len(all_temp_data) > 0:  # If we have valid data
                clean_data = all_temp_data  # Use the combined temperature data
                
                # Clear previous plot
                ax.clear()  # Clear the matplotlib axes
                
                if plot_type == "Line Plot":  # If line plot selected
                    ax.plot(range(len(clean_data)), clean_data, marker='o')  # Create line plot with markers
                elif plot_type == "Bar Chart":  # If bar chart selected
                    ax.bar(range(len(clean_data)), clean_data)  # Create bar chart
                elif plot_type == "Histogram":  # If histogram selected
                    ax.hist(clean_data, bins=15)  # Create histogram with 15 bins
                elif plot_type == "Scatter Plot":  # If scatter plot selected
                    ax.scatter(range(len(clean_data)), clean_data)  # Create scatter plot
                
                ax.set_title(variable + " - " + plot_type)  # Set plot title
                ax.set_xlabel("Data Points")  # Set x-axis label
                ax.set_ylabel(variable + " (°C)")  # Set y-axis label with temperature units
                
                canvas.draw()  # Refresh the plot canvas
                # Automatically switch to Plot tab
                notebook.select(0)  # Switch to the first tab (Plot tab)
                return  # Exit function
    elif variable == "Rainfall":  # If user selected rainfall
        # Look for rainfall column
        for col in data.columns:  # Loop through all column names
            if "rainfall" in col.lower():  # Check if column contains rainfall info
                col_name = col  # Set the matching column name
                break  # Stop searching once found
    
    # Fallback to original matching if specific mapping didn't work
    if col_name is None:  # If no specific column was found
        for col in data.columns:  # Loop through all columns
            if variable in col or ("temp" in col.lower() and "temperature" in variable.lower()):  # Check for general matches
                col_name = col  # Set the matching column name
                break  # Stop searching once found
    
    if col_name is None:  # If still no column found
        return  # Exit function
    
    # Clean data
    clean_data = pd.to_numeric(data[col_name], errors='coerce').dropna()  # Convert to numeric and remove invalid values
    
    # Clear previous plot
    ax.clear()  # Clear the matplotlib axes
    
    # Create plot
    if plot_type == "Line Plot":  # If line plot selected
        ax.plot(range(len(clean_data)), clean_data, marker='o')  # Create line plot with markers
    elif plot_type == "Bar Chart":  # If bar chart selected
        ax.bar(range(len(clean_data)), clean_data)  # Create bar chart
    elif plot_type == "Histogram":  # If histogram selected
        ax.hist(clean_data, bins=15)  # Create histogram with 15 bins
    elif plot_type == "Scatter Plot":  # If scatter plot selected
        ax.scatter(range(len(clean_data)), clean_data)  # Create scatter plot
    
    ax.set_title(variable + " - " + plot_type)  # Set plot title combining variable and plot type
    ax.set_xlabel("Data Points")  # Set x-axis label
    
    # Set y-axis label with appropriate units
    if variable == "Max Temperature" or variable == "Min Temperature":  # If temperature variable
        ax.set_ylabel(variable + " (°C)")  # Set y-axis label with Celsius units
    elif variable == "Rainfall":  # If rainfall variable
        ax.set_ylabel(variable + " (mm)")  # Set y-axis label with millimeter units
    elif variable == "Max Wind Speed":  # If wind speed variable
        ax.set_ylabel(variable + " (km/h)")  # Set y-axis label with km/h units
    else:  # For any other variable
        ax.set_ylabel(variable)  # Set y-axis label without units
    
    canvas.draw()  # Refresh the plot canvas
    
    # Automatically switch to Plot tab
    notebook.select(0)  # Switch to the first tab (Plot tab)

def show_stats():
    """Show basic statistics"""
    data = get_selected_data()  # Get the data based on user selection
    
    if data.empty:  # If no data is available
        stats_text.delete(1.0, tk.END)  # Clear the statistics text widget
        stats_text.insert(1.0, "No data available")  # Insert "no data" message
        # Automatically switch to Statistics tab
        notebook.select(1)  # Switch to the second tab (Statistics tab)
        return  # Exit function early
    
    variable = variable_var.get()  # Get the selected variable
    col_name = None  # Initialize column name variable
    
    # Create mapping for more robust column matching
    if variable == "Max Wind Speed":  # If user selected max wind speed
        # Look for maximum wind gust speed column
        for col in data.columns:  # Loop through all column names
            if "speed of maximum wind gust" in col.lower():  # Check if column contains wind speed info
                col_name = col  # Set the matching column name
                break  # Stop searching once found
    elif variable == "Max Temperature":  # If user selected max temperature
        # Look for ALL maximum temperature columns and combine their data
        matching_cols = []  # List to store matching column names
        for col in data.columns:  # Loop through all columns
            if "maximum temperature" in normalize_column_name(col):  # Check if column contains max temperature
                matching_cols.append(col)  # Add column to matching list
        
        if matching_cols:  # If matching columns were found
            # Combine data from all matching columns
            all_temp_data = pd.Series(dtype=float)  # Create empty series for temperature data
            for col in matching_cols:  # Loop through matching columns
                col_data = pd.to_numeric(data[col], errors='coerce').dropna()  # Convert to numeric, remove invalid data
                all_temp_data = pd.concat([all_temp_data, col_data], ignore_index=True)  # Combine with existing data
              if len(all_temp_data) > 0:  # If we have valid data
                clean_data = all_temp_data  # Use the combined temperature data
                
                stats_info = f"STATISTICS - {variable}\n"  # Start building statistics string with dynamic title
                stats_info = stats_info + "="*30 + "\n\n"  # Add separator line
                stats_info = stats_info + "Count: " + str(len(clean_data)) + "\n"  # Add count of data points
                stats_info = stats_info + "Mean: " + str(round(clean_data.mean(), 2)) + "\n"  # Add mean value rounded to 2 decimals
                stats_info = stats_info + "Median: " + str(round(clean_data.median(), 2)) + "\n"  # Add median value rounded to 2 decimals
                stats_info = stats_info + "Standard Deviation: " + str(round(clean_data.std(), 2)) + "\n"  # Add standard deviation rounded to 2 decimals
                stats_info = stats_info + "Minimum: " + str(round(clean_data.min(), 2)) + "\n"  # Add minimum value rounded to 2 decimals
                stats_info = stats_info + "Maximum: " + str(round(clean_data.max(), 2)) + "\n"  # Add maximum value rounded to 2 decimals
                
                stats_text.delete(1.0, tk.END)  # Clear the statistics text widget
                stats_text.insert(1.0, stats_info)  # Insert the statistics information
                # Automatically switch to Statistics tab
                notebook.select(1)  # Switch to the second tab (Statistics tab)
                return  # Exit function
    elif variable == "Min Temperature":  # If user selected min temperature
        # Look for ALL minimum temperature columns and combine their data
        matching_cols = []  # List to store matching column names
        for col in data.columns:  # Loop through all columns
            if "minimum temperature" in normalize_column_name(col):  # Check if column contains min temperature
                matching_cols.append(col)  # Add column to matching list
        
        if matching_cols:  # If matching columns were found
            # Combine data from all matching columns
            all_temp_data = pd.Series(dtype=float)  # Create empty series for temperature data
            for col in matching_cols:  # Loop through matching columns
                col_data = pd.to_numeric(data[col], errors='coerce').dropna()  # Convert to numeric, remove invalid data
                all_temp_data = pd.concat([all_temp_data, col_data], ignore_index=True)  # Combine with existing data
              if len(all_temp_data) > 0:  # If we have valid data
                clean_data = all_temp_data  # Use the combined temperature data
                
                stats_info = f"STATISTICS - {variable}\n"  # Start building statistics string with dynamic title
                stats_info = stats_info + "="*30 + "\n\n"  # Add separator line
                stats_info = stats_info + "Count: " + str(len(clean_data)) + "\n"  # Add count of data points
                stats_info = stats_info + "Mean: " + str(round(clean_data.mean(), 2)) + "\n"  # Add mean value rounded to 2 decimals
                stats_info = stats_info + "Median: " + str(round(clean_data.median(), 2)) + "\n"  # Add median value rounded to 2 decimals
                stats_info = stats_info + "Standard Deviation: " + str(round(clean_data.std(), 2)) + "\n"  # Add standard deviation rounded to 2 decimals
                stats_info = stats_info + "Minimum: " + str(round(clean_data.min(), 2)) + "\n"  # Add minimum value rounded to 2 decimals
                stats_info = stats_info + "Maximum: " + str(round(clean_data.max(), 2)) + "\n"  # Add maximum value rounded to 2 decimals
                
                stats_text.delete(1.0, tk.END)  # Clear the statistics text widget
                stats_text.insert(1.0, stats_info)  # Insert the statistics information
                # Automatically switch to Statistics tab
                notebook.select(1)  # Switch to the second tab (Statistics tab)
                return  # Exit function
    elif variable == "Rainfall":  # If user selected rainfall
        # Look for rainfall column
        for col in data.columns:  # Loop through all column names
            if "rainfall" in col.lower():  # Check if column contains rainfall info
                col_name = col  # Set the matching column name
                break  # Stop searching once found
    
    # Fallback to original matching if specific mapping didn't work
    if col_name is None:  # If no specific column was found
        for col in data.columns:  # Loop through all columns
            if variable in col or ("temp" in col.lower() and "temperature" in variable.lower()):  # Check for general matches
                col_name = col  # Set the matching column name
                break  # Stop searching once found
    
    if col_name is None:  # If still no column found
        stats_text.delete(1.0, tk.END)  # Clear the statistics text widget
        stats_text.insert(1.0, "Variable not found")  # Insert error message
        # Automatically switch to Statistics tab
        notebook.select(1)  # Switch to the second tab (Statistics tab)
        return  # Exit function
      # Clean data and calculate stats
    clean_data = pd.to_numeric(data[col_name], errors='coerce').dropna()  # Convert to numeric and remove invalid values
    
    stats_info = f"STATISTICS - {variable}\n"  # Start building statistics string with dynamic title
    stats_info = stats_info + "="*30 + "\n\n"  # Add separator line
    stats_info = stats_info + "Count: " + str(len(clean_data)) + "\n"  # Add count of data points
    stats_info = stats_info + "Mean: " + str(round(clean_data.mean(), 2)) + "\n"  # Add mean value rounded to 2 decimals
    stats_info = stats_info + "Std Dev: " + str(round(clean_data.std(), 2)) + "\n"  # Add standard deviation rounded to 2 decimals
    stats_info = stats_info + "Min: " + str(round(clean_data.min(), 2)) + "\n"  # Add minimum value rounded to 2 decimals
    stats_info = stats_info + "Max: " + str(round(clean_data.max(), 2)) + "\n"  # Add maximum value rounded to 2 decimals
    stats_info = stats_info + "Median: " + str(round(clean_data.median(), 2)) + "\n"  # Add median value rounded to 2 decimals
    
    stats_text.delete(1.0, tk.END)  # Clear the statistics text widget
    stats_text.insert(1.0, stats_info)  # Insert the statistics information
    
    # Automatically switch to Statistics tab
    notebook.select(1)  # Switch to the second tab (Statistics tab)

def clear_plot():
    """Clear the plot"""
    ax.clear()  # Clear the matplotlib axes
    ax.text(0.5, 0.5, "Plot Cleared", ha='center', va='center', transform=ax.transAxes)  # Add centered text message
    canvas.draw()  # Refresh the plot canvas

def update_display_controls():
    """Show/hide controls based on display mode"""
    display_choice = display_var.get()  # Get the current display choice
    
    if display_choice == "Single Month":  # If single month is selected
        # Show month selection in correct position (before variable frame)
        month_frame.pack(fill=tk.X, padx=10, pady=5, before=var_frame)  # Make month selection visible
    else:  # All Data - if all data is selected
        # Hide month selection
        month_frame.pack_forget()  # Hide the month selection frame

# Load data first
load_csv_files()  # Load all CSV files at startup

# Create main window
root = tk.Tk()  # Create the main tkinter window
root.title("Weather Data Viewer")  # Set window title
root.geometry("1000x700")  # Set window size

# Main frame
main_frame = tk.Frame(root)  # Create main container frame
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)  # Pack frame to fill window with padding

# Control panel
control_frame = tk.LabelFrame(main_frame, text="Controls")  # Create labeled frame for controls
control_frame.pack(fill=tk.X, pady=(0, 10))  # Pack frame horizontally with bottom padding

# Display options
display_frame = tk.Frame(control_frame)  # Create frame for display options
display_frame.pack(fill=tk.X, padx=10, pady=5)  # Pack frame horizontally with padding

tk.Label(display_frame, text="Display:").grid(row=0, column=0, padx=5)  # Create label for display options

display_var = tk.StringVar(value="Single Month")  # Create variable to store display choice, default to Single Month
tk.Radiobutton(display_frame, text="Single Month", variable=display_var, value="Single Month", command=update_display_controls).grid(row=0, column=1, padx=5)  # Create radio button for single month option
tk.Radiobutton(display_frame, text="All Data", variable=display_var, value="All Data", command=update_display_controls).grid(row=0, column=2, padx=5)  # Create radio button for all data option

# Month selection
month_frame = tk.Frame(control_frame)  # Create frame for month selection
month_frame.pack(fill=tk.X, padx=10, pady=5)  # Pack frame horizontally with padding

tk.Label(month_frame, text="Month:").grid(row=0, column=0, padx=5)  # Create label for month selection
month_var = tk.StringVar(value="24-05")  # Create variable to store selected month, default to May 2024
month_combo = ttk.Combobox(month_frame, textvariable=month_var, values=months_list, state="readonly")  # Create dropdown for month selection
month_combo.grid(row=0, column=1, padx=5)  # Position dropdown in grid

# Variable selection
var_frame = tk.Frame(control_frame)  # Create frame for variable selection
var_frame.pack(fill=tk.X, padx=10, pady=5)  # Pack frame horizontally with padding

tk.Label(var_frame, text="Variable:").grid(row=0, column=0, padx=5)  # Create label for variable selection
variable_var = tk.StringVar(value="Max Temperature")  # Create variable to store selected variable, default to Max Temperature
var_combo = ttk.Combobox(var_frame, textvariable=variable_var, values=["Max Temperature", "Min Temperature", "Rainfall", "Max Wind Speed"], state="readonly")  # Create dropdown for variable selection
var_combo.grid(row=0, column=1, padx=5)  # Position dropdown in grid

tk.Label(var_frame, text="Plot Type:").grid(row=0, column=2, padx=5)  # Create label for plot type selection
plot_var = tk.StringVar(value="Line Plot")  # Create variable to store selected plot type, default to Line Plot
plot_combo = ttk.Combobox(var_frame, textvariable=plot_var, values=["Line Plot", "Bar Chart", "Histogram", "Scatter Plot"], state="readonly")  # Create dropdown for plot type selection
plot_combo.grid(row=0, column=3, padx=5)  # Position dropdown in grid

# Buttons - positioned just above the chart/notebook area
button_frame = tk.Frame(main_frame)  # Create frame for buttons
button_frame.pack(fill=tk.X, padx=10, pady=5)  # Pack frame horizontally with padding

tk.Button(button_frame, text="Create Plot", command=create_plot).pack(side=tk.LEFT, padx=5)  # Create button to generate plots
tk.Button(button_frame, text="Show Stats", command=show_stats).pack(side=tk.LEFT, padx=5)  # Create button to show statistics
tk.Button(button_frame, text="Clear Plot", command=clear_plot).pack(side=tk.LEFT, padx=5)  # Create button to clear plot

# Create notebook for plot and stats
notebook = ttk.Notebook(main_frame)  # Create tabbed notebook widget
notebook.pack(fill=tk.BOTH, expand=True)  # Pack notebook to fill remaining space

# Plot tab
plot_frame = tk.Frame(notebook)  # Create frame for plot tab
notebook.add(plot_frame, text="Plot")  # Add frame as tab with "Plot" label

# Setup plot
fig, ax = plt.subplots(figsize=(8, 5))  # Create matplotlib figure and axes with specified size
canvas = FigureCanvasTkAgg(fig, plot_frame)  # Create tkinter canvas for matplotlib figure
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # Pack canvas to fill plot frame

# Stats tab
stats_frame = tk.Frame(notebook)  # Create frame for statistics tab
notebook.add(stats_frame, text="Statistics")  # Add frame as tab with "Statistics" label

# Setup stats text
stats_text = tk.Text(stats_frame, font=("Courier", 10))  # Create text widget for statistics with monospace font
stats_scrollbar = tk.Scrollbar(stats_frame, orient=tk.VERTICAL, command=stats_text.yview)  # Create vertical scrollbar for text widget
stats_text.configure(yscrollcommand=stats_scrollbar.set)  # Connect scrollbar to text widget

stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)  # Pack text widget to fill frame
stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)  # Pack scrollbar on right side

# Show initial message
ax.text(0.5, 0.5, "Click 'Create Plot' to start", ha='center', va='center', transform=ax.transAxes)  # Add centered instructional text to plot
canvas.draw()  # Draw the initial plot

# Initialize UI controls
update_display_controls()  # Set up initial display control visibility

# Run the application
root.mainloop()  # Start the tkinter event loop