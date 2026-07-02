# End-to-End Predictive Analytics Pipeline: Ride-Hailing Matrix Engine

## Project Overview
This repository contains a production-grade, modular Data Engineering and Machine Learning preprocessing pipeline. Built simulating corporate-level workflows, the engine processes messy, raw synthetic ride-hailing data, performs advanced structural feature engineering, applies validation guardrails to prevent data leakage, and fits a high-precision predictive model.

---

## Pipeline Architecture & Key Features

### 1. Data Cleaning & Standardisation
* **Missing Value Imputation:** Handled missing tracking arrays dynamically—imputing missing values with column-specific modes (`Weather_Condition`) and dynamic structural means (`Base_Fare`).

### 2. Advanced Feature Engineering
* **Algorithmic Distance Scale:** Built an internal tracking matrix `Estimated_Time_Mins` using distance vectors combined with external traffic delay patterns.
* **Conditional Surge Multiplier:** Developed automated pricing multipliers using highly optimized vector operations (`np.select`) to dynamically calculate adjustments for bad weather and traffic thresholds.

### 3. Machine Learning Core Matrix Engine
* **Feature Isolation:** Structural removal of absolute string identifiers (`Ride_ID`) and non-encoded parameters to build pristine `X` and `y` matrix arrays.
* **Data Leakage Guardrails:** Segmented data using an 80/20 train-test split layout. Enforced absolute mathematical isolation by strictly training the `StandardScaler` on the training partition (`X_train`) and only transforming the test vector (`X_test`).
* **Predictive Performance:** Fitted an optimized Scikit-Learn `LinearRegression` engine. Evaluated tracking anomalies using Root Mean Squared Error (RMSE) and verified variance coverage using the $R^2$ Metric.

---

## Tech Stack
* **Language:** Python 3
* **Libraries:** NumPy, Pandas, Scikit-Learn

---

## How to Run & Replicate

1. Clone this repository:
   ```bash
   git clone [https://github.com/RohanAhmed01/ride-fare-ml-pipeline.git](https://github.com/RohanAhmed01/ride-fare-ml-pipeline.git)
