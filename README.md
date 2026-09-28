# AI-ML Club Recruitment Task 2026

## 1. Candidate Details
* **Name:** AARAV SINGH
* **Year:** 2nd Year/3rd SEM
* **Branch/Degree:** B.Tech(CSE-Core)

## 2. Tasks Completed
* Task 1: Air Quality Forecasting

## 3. Problem Statement
The goal of this project was to analyze historical air quality measurements from the UCI dataset and build a machine learning regression model to accurately predict future Carbon Monoxide (CO) levels based on temporal and lagged features.

## 4. Approach
1. **Data Preprocessing:** Handled invalid `-200` sensor readings by converting them to NaNs and forward-filling them to maintain time-series continuity.
2. **EDA:** Plotted distributions and correlation heatmaps to understand relationships between the various pollutants.
3. **Feature Engineering:** Extracted hour, day, and month from the datetime index. Created a 1-hour lag feature (`CO_lag_1`) and a 4-hour rolling average feature to provide temporal context to the model.
4. **Modeling:** Performed a strict chronological 80/20 train-test split to avoid data leakage, then trained a Random Forest Regressor.

## 5. Technologies Used
* Python
* Pandas & NumPy (Data Manipulation)
* Matplotlib & Seaborn (Data Visualization)
* Scikit-Learn (Machine Learning)

## 6. Results
* **MAE:** 0.3010
* **MSE:** 0.2298
* **RMSE:** 0.4794
* **R2 Score:** 0.8787
* **Outcome:** The model successfully tracked general pollution trends. The feature importance analysis revealed that the 1-hour lagged feature was overwhelmingly the strongest predictor of future pollution levels.

## 7. Key Learnings
1. Learned how to properly clean messy time-series data, specifically dealing with placeholder garbage values without breaking the chronological sequence.
2. Understood the critical importance of chronological splitting to prevent data leakage in time-series forecasting.
3. Learned how to engineer lagged and rolling-window features to give machine learning models historical context.

## 8. Challenges
* **Challenge:** Handling missing values in a time-series format without randomly dropping rows.
* **Solution:** Used forward-filling (`ffill()`) to carry the last valid observation forward, which is a logical assumption for gradually changing environmental data.
