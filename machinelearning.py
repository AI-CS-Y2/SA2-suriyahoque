import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression  # For regression task
from sklearn.neighbors import KNeighborsRegressor  # For regression task
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# Load your dataset (make sure to provide the correct path)
df = pd.read_csv('wearable_tech_sleep_quality_1.csv')

# Check the first few rows of your dataset to confirm column names and structure
print(df.head())

# Ensure you are selecting the correct features and target columns
X = df[['Heart_Rate_Variability', 'Body_Temperature', 'Movement_During_Sleep', 'Caffeine_Intake_mg', 
        'Stress_Level', 'Bedtime_Consistency', 'Light_Exposure_hours']]  # Adjust columns as needed

# Assuming 'Sleep_Duration_Hours' is your target variable
y = df['Sleep_Duration_Hours']  # Adjust this if your target column is different

# Split the data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Linear Regression model (for continuous target)
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_reg_pred = lin_reg.predict(X_test)

# Initialize K-Nearest Neighbors Regressor (for continuous target)
knn_reg = KNeighborsRegressor(n_neighbors=3)
knn_reg.fit(X_train, y_train)
knn_reg_pred = knn_reg.predict(X_test)

# Evaluation function for regression (using MSE and R-squared)
def evaluate_model_regression(y_true, y_pred, model_name):
    print(f"\nPerformance Metrics for {model_name}:")
    print(f"Mean Squared Error: {mean_squared_error(y_true, y_pred):.2f}")
    print(f"R-squared: {r2_score(y_true, y_pred):.2f}")

# Evaluate Linear Regression
evaluate_model_regression(y_test, lin_reg_pred, "Linear Regression")

# Evaluate KNN Regressor
evaluate_model_regression(y_test, knn_reg_pred, "K-Nearest Neighbors Regressor")

# Prediction for new data
new_data = pd.DataFrame([[79.93, 37.19, 1.32, 107.62, 2.77, 0.66, 7.93]], 
                        columns=['Heart_Rate_Variability', 'Body_Temperature', 'Movement_During_Sleep', 
                                 'Caffeine_Intake_mg', 'Stress_Level', 'Bedtime_Consistency', 
                                 'Light_Exposure_hours'])  # Adjust this for actual new data

lin_reg_prediction = lin_reg.predict(new_data)
knn_reg_prediction = knn_reg.predict(new_data)

print(f"\nLinear Regression Prediction for {new_data.values}: {lin_reg_prediction[0]:.2f}")
print(f"KNN Regressor Prediction for {new_data.values}: {knn_reg_prediction[0]:.2f}")

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize Decision Tree Regressor (another model for regression)
dt_reg = DecisionTreeRegressor(random_state=42)
dt_reg.fit(X_train, y_train)
dt_reg_pred = dt_reg.predict(X_test)

# Evaluate the Decision Tree Regressor
evaluate_model_regression(y_test, dt_reg_pred, "Decision Tree Regressor")

# Prediction for new data using the scaled features
new_data_scaled = scaler.transform(new_data)  # Don't forget to scale new data
dt_reg_prediction = dt_reg.predict(new_data_scaled)

print(f"Decision Tree Regressor Prediction for {new_data.values}: {dt_reg_prediction[0]:.2f}")