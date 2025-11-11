Project 53: Renewable Energy Output Prediction

This project uses PySpark's MLlib to predict renewable energy output based on synthetic weather data. The results are visualized using a Python Dash dashboard.

Project Structure

generate_energy_data.py: Python script to create a synthetic renewable_energy.csv dataset.

energy_prediction.py: PySpark script that loads the data, handles missing values, trains a Linear Regression model, and saves the predictions to energy_predictions.csv.

prediction_dashboard.py: A Python Dash application that reads energy_predictions.csv and visualizes the actual vs. predicted power output.

README_energy.md: This file.

How to Run the Project

Step 1: Install Dependencies

You need Python and a working Spark installation.

Install the required Python libraries (if you haven't already from the previous project):

pip install pyspark dash pandas plotly numpy


Note: This ML project does not require the graphframes or networkx libraries.

Step 2: Generate the Dataset

First, run the data generation script. This will create renewable_energy.csv.

python generate_energy_data.py


Step 3: Run the PySpark ML Analysis

Next, run the main analysis script using spark-submit.

spark-submit energy_prediction.py


You will see Spark output, including the model's performance metrics (RMSE and MAE). This script will create energy_predictions.csv.

Step 4: Launch the Dashboard

Finally, run the dashboard script.

python prediction_dashboard.py


Open your web browser and navigate to the URL shown in your terminal (usually http://127.0.0.1:8050/). You will see the interactive dashboard showing the actual vs. predicted trends.
