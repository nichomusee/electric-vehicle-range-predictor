#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 1. Load dataset (make sure filename is correct)
df = pd.read_csv("electric_vehicles_spec_2025.csv")

# 2. Handle missing values (simplified cleaning)
df['model'].fillna("Unknown Model", inplace=True)
df['torque_nm'].fillna(df['torque_nm'].median(), inplace=True)
df['fast_charging_power_kw_dc'].fillna(df['fast_charging_power_kw_dc'].median(), inplace=True)
df['cargo_volume_l'] = pd.to_numeric(df['cargo_volume_l'], errors='coerce')
df['cargo_volume_l'].fillna(df['cargo_volume_l'].median(), inplace=True)

# 3. Scale selected features
scaler = StandardScaler()
X = scaler.fit_transform(df[['battery_capacity_kWh', 'top_speed_kmh',
                             'fast_charging_power_kw_dc', 'torque_nm', 'length_mm']])
y = df['range_km']

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train models
lr_model = LinearRegression().fit(X_train, y_train)
rf_model = RandomForestRegressor(random_state=42).fit(X_train, y_train)
svr_model = SVR(kernel='rbf').fit(X_train, y_train)
knn_model = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)

# 6. Evaluation function (manual RMSE calculation)
def evaluate(name, model):
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))  # manual square root
    r2 = r2_score(y_test, pred)
    print(f"\n{name} Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")

# 7. Evaluate all models
evaluate("Linear Regression", lr_model)
evaluate("Random Forest", rf_model)
evaluate("SVR", svr_model)
evaluate("KNN", knn_model)

# 8. Save scaler + best model (Random Forest)
joblib.dump(scaler, "scaler.pkl")
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(lr_model, "lr_model.pkl")
joblib.dump(svr_model, "svr_model.pkl")
joblib.dump(knn_model, "knn_model.pkl")

