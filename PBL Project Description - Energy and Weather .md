# Predictive Modelling of Household Energy Consumption Based on Weather Pattern and Machine Learning in Python

## Project Background

The increasing demand for electricity and the need for sustainable energy practices make efficient energy resource management crucial. Traditional energy forecasting methods are often insufficient for modern power grids, especially with growing renewable energy sources and fluctuating demand. Leveraging advanced technologies like machine learning and data analytics offers a promising solution for enhanced energy management. This project aims to address these challenges by developing a sophisticated tool to predict energy demand and optimize energy usage based on various influencing factors. This includes:

1.  **Visualize Energy Consumption Data and Weather Data**: Create a graphical user interface (GUI) to visualize household energy consumption data alongside corresponding weather data such as temperature and wind.
2.  **Develop a Machine Learning Tool**: Implement a Recurrent Neural Network (RNN) model to predict energy demand and sustainably optimize energy usage from the power grid. This tool will forecast electricity demand based on various possible factors.

## Project Objectives

### PART 1: Data Visualization (Session 7 - Session 8)

* Develop a Graphic User Interface (GUI) to visualise all available data (i.e., energy consumption, solar power generation, various weather data).
* **Flexible Data Display**: The GUI should allow end users to choose between displaying data for a specific month or viewing all available data at once. Different data sets can be represented using various graph types of your choice.
* **Custom Data Range Selection**: The GUI should accept user input for a starting date and a finishing date, and display the required data within the specified time frame. Users should be able to select their preferred graph type for this display.
* **Basic Statistical Analysis**: Conduct basic statistical analysis and display the results on the GUI. The statistical analysis may include, but is not limited to, mean, standard deviation, and five-number summary.

### PART 2: Machine Learning Implementation (Session 9 - Session 10)

* **Dataset Preparation**: Split the dataset into training and testing sets. Develop and train an RNN model using the training data set. Validate the model using testing data and evaluate its performance.
* **Model Performance Evaluation**: Model performance can be evaluated by: 1) Implementing appropriate error handling mechanisms (i.e., MAE, RMSE) to measure and report the accuracy of the model; 2) Visually comparing the predicted results with actual results and highlighting any patterns of discrepancies.
* **Single Factor Analysis**: Repeat the above analysis for one other influencing factor. Train a new model and record its performance. Analyse the relationship between a single weather factor (e.g., temperature) and household energy consumption.
* **Two Factor Analysis**: Repeat the above analysis for a combined two-factor training. Record the model performance. Examine the combined effect of two factors (e.g., temperature and wind speed) on energy consumption.
* **Model Performance Summary**: Record all the RNN models and their performance in a table (i.e., data frame). Comment on your findings based on the information in this table. Apply techniques to optimize energy usage based on prediction outcomes.

## Expected Deliverables

* A fully functional GUI for data visualization.
* A trained RNN model capable of predicting energy demand.
* Analytical reports on the relationship between influencing factors and energy consumption.
* Documentation of the code, methodology, and results.

## Assessment

### PART 1: Data Visualization (Session 7 - Session 8)

At the end of session 8, you will be asked to demonstrate the features of your GUI to the lab facilitator. This demonstration is a crucial part of this project, showcasing the practical application of your work. Please ensure the GUI is fully functional and ready to be evaluated based on the following criteria:

* **Completion**: Ensure that all required features and functionalities are implemented. Your GUI should be fully operational, allowing users to perform necessary tasks without encountering errors.
* **Ease of use**: The layout should be intuitive, with clear navigation paths and easily accessible features. Users should be able to interact with the GUI without needing extensive instructions.
* **Graphical Presentation of Information**: Information should be presented clearly and effectively through graphs, charts, and other visual aids. The design should be visually attractive.

### PART 2: Machine Learning Implementation (Session 9 - Session 10)

After the completion of session 10, you will be asked to provide a team-based written report to compile your findings, analyses, and reflections. The report is limited to 1500 words (+/-10%), with a standard structure of Introduction and Background, Methodology, Results and Discussions, and Conclusion.