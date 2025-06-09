# Weather Data Visualization GUI - User Instructions

## Overview
This is a simple weather data visualization tool that follows basic Python syntax patterns. The GUI allows you to analyze weather data from CSV files and create various plots and statistics.

## Features
1. **Data Loading**: Load weather data from CSV files (24-05.csv through 25-04.csv)
2. **Date Filtering**: Select specific months, days, and years using dropdown menus
3. **Multiple Plot Types**: Create line plots, bar charts, and histograms
4. **Basic Statistics**: Calculate mean, standard deviation, min, max, and data point count

## How to Use

### Step 1: Start the Application
Run the GUI by executing:
```
python GUI_VerySimple.py
```

### Step 2: Load Data
1. Click the "Load Data" button
2. The status will show "Data loaded successfully!" if all CSV files are found
3. If some files are missing, the tool will load what it can find

### Step 3: Select Date Range (Optional)
- **Month**: Choose a specific month or "All Data"
- **Day**: Choose a specific day (1-31) or "All"
- **Year**: Choose 2024, 2025, or "All"

### Step 4: Choose Plot Type
Select from:
- **Line Plot**: Shows temperature trends over time
- **Bar Chart**: Shows temperature as bars for each data point
- **Histogram**: Shows distribution of temperature values

### Step 5: Create Visualization
1. Click "Create Plot" to generate the chart
2. A matplotlib window will open showing your visualization
3. You can save, zoom, or pan in the matplotlib window

### Step 6: Calculate Statistics
1. Click "Calculate Statistics" to see basic statistical information
2. Results will show:
   - Mean temperature
   - Standard deviation
   - Minimum temperature
   - Maximum temperature
   - Number of data points

## Data Format Expected
The CSV files should contain at least these columns:
- `Date`: Date in a format pandas can parse
- `Temperature (°C)`: Temperature values in Celsius

## Troubleshooting
- **"No data files found"**: Make sure the CSV files (24-05.csv, etc.) are in the same directory as the GUI
- **"No data to plot"**: Check your date filters - you might have selected a combination that has no data
- **Import errors**: Make sure pandas, matplotlib, and numpy are installed

## Technical Notes
- This GUI strictly follows basic Python syntax patterns
- Uses only simple functions and variables (no complex classes or global management)
- Follows the syntax patterns shown in `example syntax.py`
- Handles errors gracefully with try/except blocks
