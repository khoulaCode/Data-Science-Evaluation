import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Load the datasets
train_data_path = "C:/Users/71589/Data-Science-Evaluation/TS3/Train Coffee Sales.csv"
test_data_path = "C:/Users/71589/Data-Science-Evaluation/TS3/Test Coffee Sales.csv"

@st.cache_data
def load_data():
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    return train_data, test_data

# Load and preprocess data
train_data, test_data = load_data()

# Data Exploration and Preprocessing
st.title("☕ Coffee Sales Analysis and Forecasting 💸")
st.markdown(
    """
    Welcome to the Coffee Sales Analysis Dashboard! This app provides an insightful analysis of coffee sales data and forecasts future sales.
    Let's dive into the data and explore trends, patterns, and make predictions! 🚀
    """
)

# Convert datetime
train_data['datetime'] = pd.to_datetime(train_data['datetime'])
train_data.set_index('datetime', inplace=True)

# Handle missing values
train_data['card'].fillna('Unknown', inplace=True)
train_data['money'].fillna(train_data['money'].median(), inplace=True)

# Feature Engineering
train_data['hour'] = train_data.index.hour
train_data['day_of_week'] = train_data.index.day_name()
train_data['month'] = train_data.index.month

# Overview of the Dataset
st.header("📊 Dataset Overview")
st.write(f"**Rows**: {train_data.shape[0]:,}")
st.write(f"**Columns**: {train_data.shape[1]:,}")
st.dataframe(train_data.head(3))

# Cash Type and Coffee Name Distribution
st.subheader("💰 Cash Type Distribution")
st.bar_chart(train_data['cash_type'].value_counts(), use_container_width=True)

st.subheader("☕ Coffee Name Distribution")
st.bar_chart(train_data['coffee_name'].value_counts(), use_container_width=True)

# Coffee Sales Over Time
st.subheader("📈 Coffee Sales Over Time")
train_data['month'] = train_data.index.to_period('M')
monthly_money_avg = train_data.groupby('month')['money'].transform('mean')

overall_avg = train_data['money'].mean()
daily_avg = train_data['money'].resample('D').mean()
weekly_rolling_avg = daily_avg.rolling(window=7, center=True).mean()

plt.figure(figsize=(10, 5))
plt.axhline(y=overall_avg, color='grey', alpha=0.5, linestyle='--', label='Overall Average')
plt.plot(daily_avg, alpha=0.5, color='#FF7F50', label='Daily Average')
plt.plot(weekly_rolling_avg, color='#1E90FF', alpha=0.75, label='Weekly Rolling Average')
plt.xlabel('Date')
plt.ylabel('Money')
plt.title('Daily Coffee Sales Over Time', fontsize=16, color='#2E8B57')
plt.legend(loc='upper right')
plt.grid(False)
st.pyplot(plt)

# Coffee Sales per Coffee Name
st.subheader("🎻 Coffee Sales per Coffee Name")
plt.figure(figsize=(8, 4))
sns.violinplot(x='coffee_name', y='money', data=train_data, palette='Set3')
plt.title('Coffee Sales per Coffee Name', fontsize=16, color='#FF6347')
plt.xlabel('Coffee Name')
plt.ylabel('Money')
plt.xticks(rotation=45)
st.pyplot(plt)

# Average Coffee Sales per Month
st.subheader("📅 Average Coffee Sales per Month")
plt.figure(figsize=(6, 3))
sns.boxplot(x='money', y='month', data=train_data, palette='coolwarm')
plt.title('Average Coffee Sales per Month', fontsize=16, color='#4682B4')
plt.xlabel('Money')
plt.ylabel('Month')
plt.grid(False)
st.pyplot(plt)

# Sales per Hour
st.subheader("⏰ Sales per Hour")
sales_per_hour = train_data.groupby('hour')['money'].sum().reset_index()
plt.figure(figsize=(6, 3))
plt.bar(sales_per_hour['hour'], sales_per_hour['money'], color='#32CD32')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Sales')
plt.title('Number of Sales per Hour', fontsize=16, color='#FF4500')
plt.xticks(range(0, 24))
plt.grid(axis='y', linestyle='--', linewidth=0.5)
st.pyplot(plt)

# Stationarity Test
st.subheader("📉 Stationarity Test")
result = adfuller(train_data['money'])
st.write(f"**ADF Statistic**: {result[0]}")
st.write(f"**p-value**: {result[1]}")
SIGNIFICANCE_LEVEL = 0.05
if result[1] < SIGNIFICANCE_LEVEL:
    st.write("✅ The time series is stationary!")
else:
    st.write("⚠️ The time series is not stationary!")

# Time Series Decomposition
st.subheader("🔍 Time Series Decomposition")
daily_data = train_data['money'].resample('D').sum()
result = seasonal_decompose(daily_data.dropna(), model='additive', period=7)
result.plot()
plt.suptitle("Time Series Decomposition", fontsize=16, color='#800080')
st.pyplot(plt)

# Log Transformation and Differencing
st.subheader("🔄 Log Transformation and Differencing")
daily_data_log = np.log1p(daily_data.clip(lower=1))  # Apply log(1 + x) transformation, avoid log of 0
daily_data_diff = daily_data_log.diff().dropna()  # Differencing to remove trend

# Identify ARIMA (SARIMAX) Parameters
st.subheader("🔍 Identify ARIMA Parameters using ACF and PACF")

# ACF and PACF Plots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot ACF
plot_acf(daily_data_diff, lags=20, ax=axes[0])
axes[0].set_title("Autocorrelation Function (ACF)")

# Plot PACF
plot_pacf(daily_data_diff, lags=20, ax=axes[1])
axes[1].set_title("Partial Autocorrelation Function (PACF)")

st.pyplot(fig)

# Fit SARIMA model based on identified parameters
st.subheader("🔧 SARIMAX Modeling & Forecasting")
P, Q, D, s = 1, 1, 1, 7
p_AR_term, d_differencing_term, q_MA_term = 2, 1, 2  # Adjust these based on ACF/PACF plots

model = SARIMAX(daily_data_diff, order=(p_AR_term, d_differencing_term, q_MA_term),
                seasonal_order=(P, D, Q, s))
model_fit = model.fit(disp=False)
st.write(model_fit.summary())

# Forecasting with the model
forecast_diff = model_fit.forecast(steps=30)
forecast_log = daily_data_log[-1] + forecast_diff.cumsum()
forecast = np.expm1(forecast_log)  # Reverse log(1 + x) transformation

# Calculate Metrics
y_true = daily_data[-len(forecast):]

mae = mean_absolute_error(y_true, forecast)
rmse = np.sqrt(mean_squared_error(y_true, forecast))
r2 = r2_score(y_true, forecast)
rse = np.sum(np.square(y_true - forecast)) / np.sum(np.square(y_true - np.mean(y_true)))

# Display Metrics
st.subheader(f"📅 Forecast and Evaluation Metrics")
st.write(f"**Mean Absolute Error (MAE)**: {mae:.2f}")
st.write(f"**Root Mean Squared Error (RMSE)**: {rmse:.2f}")
st.write(f"**Relative Squared Error (RSE)**: {rse:.2f}")
st.write(f"**R-squared (R²)**: {r2:.4f}")

# Plot Forecast
plt.figure(figsize=(8, 4))
plt.plot(daily_data, label='Historical', color='#4169E1')
plt.plot(forecast, label='Forecast', color='#FF1493')
plt.xlabel('Date')
plt.ylabel('Coffee Sales Sum')
plt.title(f'Coffee Sales Daily Sum Forecast with Evaluation Metrics', fontsize=16, color='#008080')
plt.legend()
st.pyplot(plt)

# Plotting Predictions Only
st.subheader("🔮 Predictions Plot")

# Generate predictions using the model
forecast_diff = model_fit.forecast(steps=30)
forecast_log = daily_data_log[-1] + forecast_diff.cumsum()
forecast = np.expm1(forecast_log)  # Reverse log(1 + x) transformation

# Handle any infinities or extreme values in forecast
forecast.replace([np.inf, -np.inf], 0, inplace=True)
forecast.fillna(0, inplace=True)

# Plot only the predictions
plt.figure(figsize=(8, 4))
plt.plot(pd.date_range(start=daily_data.index[-1], periods=len(forecast), freq='D'), forecast, color='#FF1493', label='Predictions')
plt.xlabel('Date')
plt.ylabel('Predicted Coffee Sales Sum')
plt.title('Coffee Sales Prediction for the Next 30 Days', fontsize=16, color='#008080')
plt.legend()
st.pyplot(plt)

st.markdown("**Thank you for using the Coffee Sales Analysis Dashboard! Enjoy your ☕**", unsafe_allow_html=True)
