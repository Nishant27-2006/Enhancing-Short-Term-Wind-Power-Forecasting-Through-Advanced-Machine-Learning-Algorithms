import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('wtk_site_metadata.csv')

# Feature Engineering
data['latitude_wind'] = data['latitude'] * data['wind_speed']
data['longitude_wind'] = data['longitude'] * data['wind_speed']

# Clean the data
data_clean = data.dropna(subset=['longitude', 'latitude', 'fraction_of_usable_area', 'capacity', 'wind_speed', 'capacity_factor'])

# Splitting the data into features and target
X = data_clean[['longitude', 'latitude', 'fraction_of_usable_area', 'capacity', 'wind_speed', 'latitude_wind', 'longitude_wind']]
y = data_clean['capacity_factor']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Models initialization
lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=50, random_state=42)
gbr = GradientBoostingRegressor(n_estimators=50, random_state=42)
xgb = XGBRegressor(n_estimators=50, random_state=42)
lgbm = LGBMRegressor(n_estimators=50, random_state=42)

# Train models
models = [lr, rf, gbr, xgb, lgbm]
model_names = ['Linear Regression', 'RandomForest', 'GradientBoosting', 'XGBoost', 'LightGBM']
predictions = {}
rmse_scores = {}
r2_scores = {}
mae_scores = {}

for model, name in zip(models, model_names):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    rmse_scores[name] = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_scores[name] = r2_score(y_test, y_pred)
    mae_scores[name] = mean_absolute_error(y_test, y_pred)

# Create DataFrame for results
results_df = pd.DataFrame({
    'Model': model_names,
    'RMSE': [rmse_scores[name] for name in model_names],
    'R-squared': [r2_scores[name] for name in model_names],
    'MAE': [mae_scores[name] for name in model_names]
})

# Print RMSE, R-squared, and MAE comparison
print(results_df)

# 1. Vertical RMSE bar plot
plt.figure(figsize=(10, 6))
plt.bar(results_df['Model'], results_df['RMSE'], color='skyblue')
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.title('RMSE Comparison of Models')
plt.show()

# 2. Vertical R-squared bar plot
plt.figure(figsize=(10, 6))
plt.bar(results_df['Model'], results_df['R-squared'], color='lightcoral')
plt.xlabel('Model')
plt.ylabel('R-squared')
plt.title('R-squared Comparison of Models')
plt.show()

# 3. Vertical MAE bar plot
plt.figure(figsize=(10, 6))
plt.bar(results_df['Model'], results_df['MAE'], color='lightgreen')
plt.xlabel('Model')
plt.ylabel('MAE')
plt.title('MAE Comparison of Models')
plt.show()

# 4. Performance improvement over Linear Regression (RMSE)
baseline_rmse = rmse_scores['Linear Regression']
improvement = {name: (baseline_rmse - rmse_scores[name]) / baseline_rmse * 100 for name in model_names if name != 'Linear Regression'}

plt.figure(figsize=(10, 6))
plt.bar(list(improvement.keys()), list(improvement.values()), color='purple')
plt.xlabel('Model')
plt.ylabel('Performance Improvement (%)')
plt.title('Performance Improvement over Linear Regression (RMSE)')
plt.show()

# 5. ROC-like curve (Error-rate vs Threshold) for each model
plt.figure(figsize=(10, 6))
thresholds = np.linspace(0, 1, 100)

for name, y_pred in predictions.items():
    errors = []
    for t in thresholds:
        errors.append(np.mean(np.abs(y_pred - y_test) > t))
    plt.plot(thresholds, errors, label=f'{name}')

plt.xlabel('Threshold')
plt.ylabel('Error Rate')
plt.title('ROC-like Curve (Error-rate vs Threshold)')
plt.legend()
plt.show()
