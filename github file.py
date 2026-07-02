import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. DATA GENERATION
# ==========================================
np.random.seed(101)
n_rides = 250
ride_data = {
    'Ride_ID': [f'RIDE_{2000 + i}' for i in range(n_rides)],
    'Distance_KM': np.random.uniform(2.0, 45.0, size=n_rides),
    'Traffic_Delay_Mins': np.random.randint(0, 60, size=n_rides),
    'Weather_Condition': np.random.choice(['Clear', 'Rainy', 'Stormy', np.nan], size=n_rides, p=[0.6, 0.2, 0.1, 0.1]),
    'Base_Fare': np.random.choice([200, 300, 500, np.nan], size=n_rides, p=[0.5, 0.3, 0.1, 0.1])
}
df_rides = pd.DataFrame(ride_data)


# ==========================================
# 2. DATA CLEANING & FEATURE ENGINEERING
# ==========================================
# Impute missing values
most_frequent_weather = df_rides['Weather_Condition'].mode()[0]
df_rides['Weather_Condition'] = df_rides['Weather_Condition'].fillna(most_frequent_weather)

mean_fare = round(df_rides['Base_Fare'].mean(), 2)
df_rides['Base_Fare'] = df_rides['Base_Fare'].fillna(mean_fare)

# Advanced features
df_rides['Estimated_Time_Mins'] = (df_rides['Distance_KM'] * 2) + df_rides['Traffic_Delay_Mins']

conditions = [
    (df_rides['Weather_Condition'] == 'Stormy') & (df_rides['Traffic_Delay_Mins'] > 30),
    (df_rides['Weather_Condition'] == 'Rainy')
]
choices = [1.5, 1.2]
df_rides['Surge_Multiplier'] = np.select(conditions, choices, default=1.0)

# Categorical Encoding
weather_dummies = pd.get_dummies(df_rides['Weather_Condition'], prefix='Weather').astype(int)
df_ml = pd.concat([df_rides, weather_dummies], axis=1)

# Generate Target Variable
df_ml['Trip_Price'] = (df_ml['Base_Fare'] * df_ml['Surge_Multiplier']) + (df_ml['Distance_KM'] * 25)


# ==========================================
# 3. MATRIX ISOLATION & SPLITTING
# ==========================================
# Drop non-numeric and target columns for X
X = df_ml.drop(columns=['Ride_ID', 'Weather_Condition', 'Trip_Price'])
y = df_ml['Trip_Price']

# Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# ==========================================
# 4. SCALING THE FEATURES
# ==========================================
scaler = StandardScaler()

# Fit on training data ONLY, then transform both
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# 5. MODEL TRAINING & EVALUATION
# ==========================================
# Initialize and Fit
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Evaluate
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_test_pred)

# --- EXECUTIVE EVALUATION REPORT ---
print("="*50)
print("             MODEL PERFORMANCE METRICS             ")
print("="*50)
print(f" Test RMSE (Root Mean Squared Error): {rmse_test:.2f}")
print(f" Test R² Score (Accuracy):            {r2_test:.4f}")
print("="*50)
