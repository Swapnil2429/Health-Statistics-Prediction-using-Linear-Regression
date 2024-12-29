# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC This notebook will show you how to create and query a table or DataFrame that you uploaded to DBFS. [DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html) is a Databricks File System that allows you to store data for querying inside of Databricks. This notebook assumes that you have a file already inside of DBFS that you would like to read from.
# MAGIC
# MAGIC This notebook is written in **Python** so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` syntax. Python, Scala, SQL, and R are all supported.

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/Global_Health_Statistics.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "false"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)

# COMMAND ----------

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression



# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/Global_Health_Statistics.csv"
file_type = "csv"

# CSV options
infer_schema = "true"  # Change to "true" if you want to infer schema automatically
first_row_is_header = "true"  # Set this to true if the first row contains column names
delimiter = ","  # Specify the delimiter

# Load the data into a Spark DataFrame
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

# Display the Spark DataFrame
display(df)

# Convert Spark DataFrame to Pandas DataFrame
pandas_df = df.toPandas()

# Display the first few rows of the Pandas DataFrame
print(pandas_df.head())


# COMMAND ----------

#Data Exploration and Cleaning
# Display basic information
print(pandas_df.info())  # Data types and non-null counts
print(pandas_df.describe())  # Summary statistics for numeric columns

# Check for missing values
print(pandas_df.isnull().sum())

# Check unique values for categorical columns
for column in pandas_df.select_dtypes(include='object').columns:
    print(f"{column} unique values: {pandas_df[column].unique()}")


# COMMAND ----------

pandas_df = pandas_df.dropna()  # Drop all rows with missing values


# COMMAND ----------

# Drop duplicate rows
pandas_df = pandas_df.drop_duplicates()


# COMMAND ----------

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(pandas_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Distribution of key features
sns.histplot(pandas_df['Mortality Rate (%)'], bins=30, kde=True)
plt.title('Mortality Rate Distribution')
plt.show()

# COMMAND ----------

# Drop the 'Country' column if not needed for prediction
# pandas_df = pandas_df.drop(columns=['Country'])

# One-hot encode the 'Country' column (for machine learning models)
pandas_df = pd.get_dummies(pandas_df, columns=['Country'], drop_first=True)

# Now check the data types again
print(pandas_df.dtypes)

# Now you can proceed with the rest of your pipeline, such as splitting the data, training the model, etc.


# COMMAND ----------

# Get the list of columns with numeric data types (int, float, double)
numeric_columns = [col[0] for col in df.dtypes if col[1] in ['int', 'double', 'float']]

# Select only numeric columns to create a new DataFrame
numeric_spark_df = df.select(*numeric_columns)

# Show the new DataFrame with numeric columns
numeric_spark_df.show()


# COMMAND ----------

numeric_spark_df
newpandas_df = numeric_spark_df.toPandas()


# COMMAND ----------

scaler = StandardScaler()
numeric_columns = ['Prevalence Rate (%)', 'Incidence Rate (%)', 'Healthcare Access (%)']
pandas_df[numeric_columns] = scaler.fit_transform(newpandas_df[numeric_columns])


# COMMAND ----------




# Assuming your DataFrame is named 'newpandas_df'

# 1. Select feature columns and the target column
# Replace with the actual numeric columns you want to use for training
feature_columns = ['Prevalence Rate (%)', 'Incidence Rate (%)', 'Healthcare Access (%)', 'Doctors per 1000',
                   'Hospital Beds per 1000', 'Recovery Rate (%)', 'Improvement in 5 Years (%)', 
                   'Per Capita Income (USD)', 'Education Index', 'Urbanization Rate (%)']  # Adjust as needed

target_column = 'Prevalence Rate (%)'  # Replace with your target column

# 2. Prepare the feature matrix (X) and target vector (y)
X = newpandas_df[feature_columns]  # Feature columns
y = newpandas_df[target_column]    # Target column

# 3. Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make predictions on the test data
y_pred = model.predict(X_test)

# 6. Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation results
print(f"Mean Squared Error (MSE): {mse}")
print(f"R2 Score: {r2}")

# 7. Optional: Show the first few predictions vs actual values
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results.head())


# COMMAND ----------

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
print("Cross-Validation MSE:", -scores.mean())


# COMMAND ----------

# Train-test split and evaluate performance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Assuming X and y are your features and target variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on both training and testing data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate the performance on both training and testing data
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Training MSE: {train_mse}")
print(f"Test MSE: {test_mse}")
print(f"Training R²: {train_r2}")
print(f"Test R²: {test_r2}")


# COMMAND ----------

import matplotlib.pyplot as plt

# Plot actual vs predicted values for training data
plt.scatter(y_train, model.predict(X_train))
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red')
plt.title("Training: Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

# Plot actual vs predicted values for test data
plt.scatter(y_test, model.predict(X_test))
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.title("Test: Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

