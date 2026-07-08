# DevelopersHub Data Science & Analytics Internship — Phase 2

## Task 3: Energy Consumption Time Series Forecasting

### Objective
Forecast short-term household energy consumption using historical time-based 
patterns and compare the performance of ARIMA, Prophet, and XGBoost models.

### Dataset
- **Source:** Kaggle — Household Power Consumption Dataset (UCI)
- **Original Size:** 2,075,259 rows (minute-by-minute readings)
- **After Resampling:** 1,433 daily records (2006-2010)
- **Target:** Global Active Power (kW)

---

### Approach
- Parsed datetime and resampled minute data to daily averages
- Engineered time-based features: Month, Day, DayofWeek, IsWeekend, Year
- Added Lag Features (1, 7, 30 days) and Rolling Averages (7, 30 days)
- Split data by time order — last 90 days as test set
- Trained and compared ARIMA, Prophet, and XGBoost models
- Evaluated using MAE, RMSE, and R2 Score

---

### Model Results

| Model | MAE | RMSE | R2 | Rank |
|-------|-----|------|----|------|
| XGBoost | 0.1854 | 0.2524 | 0.07 | 🥇 Best |
| Prophet | 0.2138 | 0.2919 | -0.23 | 🥈 2nd |
| ARIMA | 0.5320 | 0.5909 | -4.05 | 🥉 Worst |

---

### Why XGBoost Won?
- Used lag features (yesterday, last week, last month power values)
- Lag features directly tell the model recent consumption patterns
- XGBoost learned non-linear relationships between features and power usage
- Rolling averages (7-day, 30-day) gave the model smoothed historical context
- Result: Lowest MAE (0.1854) and only positive R2 score (0.07)

### Why Prophet Underperformed?
- Prophet is designed for longer range forecasting (months/years ahead)
- Our 90-day test window is relatively short for Prophet's strength
- Prophet handled weekly and yearly seasonality well but missed sharp spikes
- Without lag features, it couldn't capture recent consumption patterns
- Still decent — MAE of 0.2138 is reasonable for energy forecasting

### Why ARIMA Failed?
- ARIMA is designed for short, simple, stationary sequences
- Our training data had 1,313 points with complex weekly + yearly seasonality
- ARIMA couldn't handle multiple seasonality patterns simultaneously
- Result: Predicted a nearly flat line (~0.58 kW) — essentially just the mean
- R2 of -4.05 confirms it performed worse than simply predicting the average

---

### Key Insights
- Lag features are the strongest predictors — yesterday's power predicts today's
- Energy consumption shows clear seasonality: higher in winter, lower in summer
- XGBoost with feature engineering outperforms dedicated time series models
- ARIMA is not suitable for long, complex, multi-seasonal time series data
- Random spikes in consumption are caused by human behavior — hard to predict

---

### Dataset
- **Source:** [Household Power Consumption Dataset](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set)
- **Note:** Dataset is too large for GitHub (2M+ rows). Download directly from Kaggle link above and place `household_power_consumption.txt` in the project folder.
- **Original Size:** 2,075,259 rows (minute-by-minute readings)
- **After Resampling:** 1,433 daily records (2006-2010)

---

### Libraries Used
`pandas` `numpy` `matplotlib` `scikit-learn` `xgboost` `prophet` `statsmodels`

---
*DevelopersHub Corporation — Data Science & Analytics Internship Phase 2*
