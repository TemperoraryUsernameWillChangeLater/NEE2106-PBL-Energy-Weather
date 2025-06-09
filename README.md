# PBL Project - Energy and Weather 🌤️⚡

## Problem-Based Learning Project for NEE2106 Computer Programming For Electrical Engineers

This repository contains a comprehensive weather data analysis and energy forecasting project developed as part of the NEE2106 course. The project focuses on analyzing weather patterns in Adelaide, South Australia, and their relationship to energy consumption and production.

## 📋 Project Overview

The **Energy and Weather PBL Project** is designed to apply programming concepts learned in the course to real-world weather data analysis. Students develop skills in data processing, statistical analysis, and GUI development while working with actual meteorological data from the Australian Bureau of Meteorology.

### 🎯 Learning Objectives

- Apply Python programming skills to real-world data analysis
- Understand the relationship between weather patterns and energy systems
- Develop GUI applications for data visualization
- Practice data cleaning, processing, and statistical analysis
- Learn about weather forecasting and energy demand prediction

## 📂 Project Structure

```
PBL Project - Energy and Weather/
├── GUI.py                                    # Main application interface
├── README.md                                 # This file
├── PBL Project Description - Energy and Weather .pdf  # Detailed project requirements
└── DataSets/                                # Weather data files
    ├── 24-05.csv  # May 2024 Adelaide weather data
    ├── 24-06.csv  # June 2024 Adelaide weather data
    ├── 24-07.csv  # July 2024 Adelaide weather data
    ├── 24-08.csv  # August 2024 Adelaide weather data
    ├── 24-09.csv  # September 2024 Adelaide weather data
    ├── 24-10.csv  # October 2024 Adelaide weather data
    ├── 24-11.csv  # November 2024 Adelaide weather data
    ├── 24-12.csv  # December 2024 Adelaide weather data
    ├── 25-01.csv  # January 2025 Adelaide weather data
    ├── 25-02.csv  # February 2025 Adelaide weather data
    ├── 25-03.csv  # March 2025 Adelaide weather data
    └── 25-04.csv  # April 2025 Adelaide weather data
```

## 🌡️ Dataset Information

### Data Source
- **Location**: Adelaide (West Terrace / Ngayirdapira), South Australia
- **Station**: 023000 (Bureau of Meteorology)
- **Period**: May 2024 - April 2025 (12 months of data)
- **Frequency**: Daily observations

### Available Weather Parameters
- **Temperature**: Minimum and maximum daily temperatures (°C)
- **Rainfall**: Daily precipitation (mm)
- **Wind**: Direction, speed, and gust information
- **Humidity**: Relative humidity at 9am and 3pm (%)
- **Pressure**: Mean sea level pressure (hPa)
- **Sunshine**: Daily sunshine hours
- **Cloud**: Cloud amount (oktas)

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas tkinter pillow
```

### Running the Application
```python
python GUI.py
```

### Data Access
The application automatically loads all weather datasets and creates month-based variables:
- `May24`, `Jun24`, `Jul24`, `Aug24` (2024 data)
- `Sep24`, `Oct24`, `Nov24`, `Dec24` (2024 data)  
- `Jan25`, `Feb25`, `Mar25`, `Apr25` (2025 data)

## 📊 Data Analysis Features

### Current Implementation
- ✅ Automated dataset loading with proper encoding
- ✅ Month-based data organization
- ✅ Data validation and integrity checks
- ✅ Support for all major weather parameters

### Planned Features
- 🔄 Temperature trend analysis
- 🔄 Rainfall pattern visualization
- 🔄 Wind direction and speed analysis
- 🔄 Seasonal weather comparison
- 🔄 Energy demand correlation analysis
- 🔄 Weather forecasting algorithms
- 🔄 Interactive data visualization

## 🔧 Technical Details

### Data Processing
- **Encoding**: Latin-1 to handle special characters (°C symbols)
- **Header Handling**: Automatic skipping of Bureau of Meteorology metadata
- **Data Validation**: Verification of correct day counts per month
- **Missing Data**: Proper handling of NaN values

### Supported Analysis
```python
# Example usage after running GUI.py
print(f"May 2024 average temperature: {May24['Maximum temperature (°C)'].mean():.1f}°C")
print(f"Total rainfall in June 2024: {Jun24['Rainfall (mm)'].sum():.1f}mm")

# Seasonal analysis
summer_months = [Dec24, Jan25, Feb25]  # Australian summer
winter_months = [Jun24, Jul24, Aug24]  # Australian winter
```

## 📈 Energy-Weather Correlation

This project explores the relationship between:
- **Temperature** ↔ **Cooling/Heating demand**
- **Wind patterns** ↔ **Wind energy generation**
- **Sunshine hours** ↔ **Solar energy production**
- **Rainfall** ↔ **Hydro energy potential**

## 🎓 Educational Value

### Programming Concepts Applied
- File I/O and CSV processing
- Data structures (dictionaries, lists)
- Control flow (loops, conditionals)
- Error handling and validation
- GUI development with tkinter
- Statistical analysis with pandas

### Real-World Applications
- Weather station data processing
- Energy grid management
- Climate analysis
- Renewable energy planning

## 📖 Project Documentation

### PBL Project Description PDF 📋

The **`PBL Project Description - Energy and Weather .pdf`** file contains the complete project specification and serves as your primary reference document. This comprehensive guide includes:

#### 📝 What's Inside the PDF:
- **Project Overview**: Detailed explanation of the Energy and Weather analysis objectives
- **Technical Requirements**: Specific programming tasks and deliverables
- **Dataset Specifications**: Information about weather data format and sources
- **Analysis Tasks**: Step-by-step breakdown of required data analysis components
- **GUI Requirements**: Interface design specifications and user interaction features
- **Assessment Criteria**: Grading rubric and evaluation standards
- **Submission Guidelines**: Deadlines, file formats, and delivery instructions
- **Learning Outcomes**: How this project aligns with course objectives

#### 🎯 How to Use the PDF:
1. **Start Here First**: Read the PDF completely before beginning any coding
2. **Reference Guide**: Keep it open while developing to check requirements
3. **Task Checklist**: Use it to track your progress through project milestones
4. **Assessment Prep**: Review grading criteria before final submission

#### 📋 Key Sections to Focus On:
- **Problem Statement**: Understanding what you need to solve
- **Functional Requirements**: What your program must do
- **Technical Constraints**: Programming standards and limitations
- **Deliverables**: What files and documentation to submit
- **Timeline**: Important dates and project phases

> 💡 **Tip**: Print or bookmark the PDF for easy reference during development. The project specification contains critical details that will guide your implementation decisions.

## 🤝 Contributing

This is a student project for NEE2106. For questions or clarifications, please contact your course instructor or teaching assistants.

## 📄 Data Attribution

Weather data sourced from the Australian Bureau of Meteorology:
- Copyright 2003 Commonwealth Bureau of Meteorology
- Station: Adelaide (West Terrace / Ngayirdapira) {023000}
- Official site for Adelaide, operational since May 2017

---
*Developed for NEE2106 Computer Programming For Electrical Engineers*