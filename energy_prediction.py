# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, avg
# from pyspark.ml.feature import VectorAssembler
# from pyspark.ml.regression import LinearRegression
# from pyspark.ml.evaluation import RegressionEvaluator
# from pyspark.ml import Pipeline

# # --- Setup Spark Session ---
# spark = SparkSession.builder \
#     .appName("RenewableEnergyPrediction") \
#     .getOrCreate()

# print("Spark session created successfully.")

# # --- 1. Load Renewable Energy Dataset ---
# print("Loading renewable_energy.csv...")
# data = spark.read.csv("renewable_energy.csv", header=True, inferSchema=True)
# data.printSchema()

# # --- 2. Handle Missing Values ---
# print("Handling missing values...")

# # Strategy: Impute with the mean for each column
# # First, calculate means for all relevant columns
# impute_cols = ["temperature", "humidity", "wind_speed", "power_output"]
# impute_values = {}

# for c in impute_cols:
#     mean_val = data.select(avg(col(c))).first()[0]
#     impute_values[c] = mean_val
#     print(f"Mean value for '{c}': {mean_val}")

# # Apply imputation
# data = data.na.fill(impute_values)

# print("Missing values imputed.")

# # --- 3. Extract Features ---
# print("Assembling feature vector...")
# # Features are: temperature, humidity, wind_speed
# # Target is: power_output

# # VectorAssembler combines multiple columns into a single vector column
# assembler = VectorAssembler(
#     inputCols=["temperature", "humidity", "wind_speed"],
#     outputCol="features",
#     handleInvalid="skip" # Skip rows with nulls if any remained
# )

# # --- 4. Build a Linear Regression Model ---
# print("Building Linear Regression model...")

# # Initialize the model
# lr = LinearRegression(featuresCol="features", labelCol="power_output")

# # A Pipeline makes it easy to chain the assembler and model
# pipeline = Pipeline(stages=[assembler, lr])

# # Split the data into training and test sets
# (trainingData, testData) = data.randomSplit([0.8, 0.2], seed=42)

# print(f"Training data count: {trainingData.count()}")
# print(f"Test data count: {testData.count()}")

# # Train the model
# model = pipeline.fit(trainingData)

# # Make predictions on the test data
# predictions = model.transform(testData)

# print("Model training complete. Predictions made on test data.")
# predictions.select("day", "power_output", "prediction").show(10)

# # --- 5. Evaluate Performance (RMSE and MAE) ---
# print("Evaluating model performance...")

# # Evaluate using RMSE
# evaluator_rmse = RegressionEvaluator(
#     labelCol="power_output", 
#     predictionCol="prediction", 
#     metricName="rmse"
# )
# rmse = evaluator_rmse.evaluate(predictions)
# print(f"Root Mean Squared Error (RMSE) on test data = {rmse}")

# # Evaluate using MAE
# evaluator_mae = RegressionEvaluator(
#     labelCol="power_output", 
#     predictionCol="prediction", 
#     metricName="mae"
# )
# mae = evaluator_mae.evaluate(predictions)
# print(f"Mean Absolute Error (MAE) on test data = {mae}")

# # --- 6. Save Results for Visualization ---
# print("Saving predictions to CSV for dashboard...")
# # We want to visualize actual vs. predicted
# output_df = predictions.select("day", "power_output", "prediction") \
#                        .orderBy("day")

# # Save to a single CSV file using pandas
# output_df.toPandas().to_csv("energy_predictions.csv", index=False)

# print("Predictions saved successfully.")

# # Optional: Print model coefficients
# lr_model = model.stages[-1]
# print(f"Coefficients: {lr_model.coefficients}")
# print(f"Intercept: {lr_model.intercept}")


# spark.stop()



from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import json  # <-- Make sure this import is here

# --- Setup Spark Session ---
spark = SparkSession.builder \
    .appName("RenewableEnergyPrediction") \
    .getOrCreate()

print("Spark session created successfully.")

# --- 1. Load Renewable Energy Dataset ---
print("Loading renewable_energy.csv...")
data = spark.read.csv("renewable_energy.csv", header=True, inferSchema=True)
data.printSchema()

# --- 2. Handle Missing Values ---
print("Handling missing values...")

# Strategy: Impute with the mean for each column
impute_cols = ["temperature", "humidity", "wind_speed", "power_output"]
impute_values = {}

for c in impute_cols:
    mean_val = data.select(avg(col(c))).first()[0]
    impute_values[c] = mean_val
    print(f"Mean value for '{c}': {mean_val}")

# Apply imputation
data = data.na.fill(impute_values)

print("Missing values imputed.")

# --- 3. Extract Features ---
print("Assembling feature vector...")
assembler = VectorAssembler(
    inputCols=["temperature", "humidity", "wind_speed"],
    outputCol="features",
    handleInvalid="skip"
)

# --- 4. Build a Linear Regression Model ---
print("Building Linear Regression model...")
lr = LinearRegression(featuresCol="features", labelCol="power_output")
pipeline = Pipeline(stages=[assembler, lr])

(trainingData, testData) = data.randomSplit([0.8, 0.2], seed=42)

print(f"Training data count: {trainingData.count()}")
print(f"Test data count: {testData.count()}")

model = pipeline.fit(trainingData)
predictions = model.transform(testData)

print("Model training complete. Predictions made on test data.")
predictions.select("day", "power_output", "prediction").show(10)

# --- 5. Evaluate Performance (RMSE and MAE) ---
print("Evaluating model performance...")

evaluator_rmse = RegressionEvaluator(
    labelCol="power_output", 
    predictionCol="prediction", 
    metricName="rmse"
)
rmse = evaluator_rmse.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE) on test data = {rmse}")

evaluator_mae = RegressionEvaluator(
    labelCol="power_output", 
    predictionCol="prediction", 
    metricName="mae"
)
mae = evaluator_mae.evaluate(predictions)
print(f"Mean Absolute Error (MAE) on test data = {mae}")

# --- 6. Save Results for Visualization ---
print("Saving predictions to CSV for dashboard...")
output_df = predictions.select("day", "power_output", "prediction") \
                       .orderBy("day")

output_df.toPandas().to_csv("energy_predictions.csv", index=False)
print("Predictions saved successfully.")

# --- NEW: Save Metrics to JSON ---
metrics = {"rmse": rmse, "mae": mae}
with open("model_metrics.json", 'w') as f:
    json.dump(metrics, f)
print("Metrics saved to model_metrics.json")
# --- End of NEW ---

lr_model = model.stages[-1]
print(f"Coefficients: {lr_model.coefficients}")
print(f"Intercept: {lr_model.intercept}")

spark.stop() 