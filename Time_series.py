import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#============= Streamlit Title and Description ======================================

st.title("Time Series Analysis and Forecasting of Coffee Sales")
st.write("""
### Project Overview:
This analysis aims to study coffee sales data collected from vending machines. 
The purpose is to uncover underlying patterns in the data and forecast future sales to assist in inventory and financial planning. 
**SARIMA model** was chosen for its ability to handle seasonality in the time series data.

- **Data**: The dataset includes daily coffee sales with features such as payment type, coffee type, and transaction time.
- **Objective**: The primary goal is to predict future coffee sales to optimize inventory and stock management.
""")

#============= 1- Data Preprocessing ======================================

# Load the dataset
file_path = r"C:\Users\71591\Desktop\dataset\Train Coffee Sales.csv"
df = pd.read_csv(file_path)

# Convert 'date' and 'datetime' columns to datetime format
df['date'] = pd.to_datetime(df['date'])
df['datetime'] = pd.to_datetime(df['datetime'])

# Handle missing values in the 'card' column using forward fill method
df['card'] = df['card'].ffill()

# Aggregate total sales by coffee type and payment method before encoding
sales_by_coffee_type = df.groupby('coffee_name')['money'].sum().reset_index()
sales_by_payment_method = df.groupby('cash_type')['money'].sum().reset_index()

# Merge these features back into the original dataframe before encoding
df = df.merge(sales_by_coffee_type, on='coffee_name', how='left', suffixes=('', '_total_by_coffee'))
df = df.merge(sales_by_payment_method, on='cash_type', how='left', suffixes=('', '_total_by_payment'))

# One-hot encode the 'cash_type' and 'coffee_name' columns
df_encoded = pd.get_dummies(df, columns=['cash_type', 'coffee_name'])

#============= Data Visualization ======================================

# Time Series Plot - Sales over time
st.subheader('Time Series Plot: Coffee Sales Over Time')
plt.figure(figsize=(10, 6))
plt.plot(df_encoded['datetime'], df_encoded['money'])
plt.title('Coffee Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales (Money)')
st.pyplot(plt)

# Sales by Coffee Type
st.subheader('Bar Plot: Total Sales by Coffee Type')
plt.figure(figsize=(8, 5))
df.groupby('coffee_name')['money'].sum().plot(kind='bar')
plt.title('Total Sales by Coffee Type')
plt.xlabel('Coffee Type')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
st.pyplot(plt)

# Sales by Payment Method
st.subheader('Bar Plot: Total Sales by Payment Method')
plt.figure(figsize=(8, 5))
df.groupby('cash_type')['money'].sum().plot(kind='bar')
plt.title('Total Sales by Payment Method')
plt.xlabel('Payment Method')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
st.pyplot(plt)

# Distribution of Sales (money)
st.subheader('Distribution Plot: Coffee Sales')
plt.figure(figsize=(8, 5))
df_encoded['money'].plot(kind='hist', bins=20, edgecolor='black')
plt.title('Distribution of Coffee Sales')
plt.xlabel('Sales Amount (Money)')
plt.ylabel('Frequency')
st.pyplot(plt)

#============= 2- Feature Engineering =====================================

# Extract time-based features from 'datetime'
df_encoded['hour'] = df_encoded['datetime'].dt.hour
df_encoded['day_of_week'] = df_encoded['datetime'].dt.dayofweek
df_encoded['month'] = df_encoded['datetime'].dt.month
df_encoded['week_of_year'] = df_encoded['datetime'].dt.isocalendar().week

# Create lag features and rolling statistics
df_encoded['lag_1'] = df_encoded['money'].shift(1)
df_encoded['rolling_mean_7'] = df_encoded['money'].rolling(window=7).mean()

# Drop rows with NaN values created by shifting or rolling
df_encoded.dropna(inplace=True)

#============= Decomposition Plot ======================================

st.subheader("Time Series Decomposition")
st.write("""
To understand the components of our coffee sales data, we decompose the time series into **trend**, **seasonality**, and **residual** components.
This allows us to observe the underlying patterns that are influencing sales performance over time.
""")

# Perform seasonal decomposition of the time series
decomposition = seasonal_decompose(df_encoded['money'], model='additive', period=7)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))

# Plot the decomposition components
decomposition.observed.plot(ax=ax1, title='Observed', legend=False)
decomposition.trend.plot(ax=ax2, title='Trend', legend=False)
decomposition.seasonal.plot(ax=ax3, title='Seasonality', legend=False)
decomposition.resid.plot(ax=ax4, title='Residuals', legend=False)

plt.tight_layout()
st.pyplot(fig)

#============= ACF and PACF Plots ======================================

st.subheader("ACF and PACF Plots")
st.write("""
The **ACF (Autocorrelation Function)** and **PACF (Partial Autocorrelation Function)** plots are used to identify the presence of any autoregressive or moving average components in the time series.
These plots help in selecting the appropriate lags for our SARIMA model.
""")

# Plot ACF and PACF
fig_acf, ax_acf = plt.subplots(1, 1, figsize=(10, 4))
plot_acf(df_encoded['money'], lags=40, ax=ax_acf)
st.pyplot(fig_acf)

fig_pacf, ax_pacf = plt.subplots(1, 1, figsize=(10, 4))
plot_pacf(df_encoded['money'], lags=40, ax=ax_pacf)
st.pyplot(fig_pacf)

#============== 3- Model Building and Evaluation ============================

st.subheader("SARIMA Model Building and Evaluation")

# Load the training data (already preprocessed and encoded)
df_train = df_encoded.copy()  # Use the encoded train data from preprocessing

# Load the test data
df_test = pd.read_csv(r"C:\Users\71591\Desktop\dataset\Test Coffee Sales.csv")

# Convert 'datetime' column to datetime format and set it as index for the test data
df_test['datetime'] = pd.to_datetime(df_test['datetime'])
df_test.set_index('datetime', inplace=True)

# Extract the 'money' column for training and testing
y_train = df_train['money']
y_test = df_test['money']

# Step 1: Fit the SARIMA model on the training data
sarima = SARIMAX(y_train, order=(1,2,1), seasonal_order=(1,1,1,7))
sarima_fit = sarima.fit()

# Print model summary
st.write(sarima_fit.summary())

# Step 2: Make predictions on the test set
y_pred = sarima_fit.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1, dynamic=False)

# Step 3: Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.write(f"Mean Absolute Error: {mae:.2f}")
st.write(f"Root Mean Squared Error: {rmse:.2f}")
st.write(f"R Squared Error: {r2:.2f}")

# Step 4: Plot the results for Training, Test, and Predicted data

# Reset index for both train and test data to ensure proper alignment
df_train = df_train.reset_index()
df_test = df_test.reset_index()

# Create a combined dataframe for proper x-axis alignment
fig, ax = plt.subplots(figsize=(10, 5))

# Plot the actual training data
ax.plot(df_train['datetime'], y_train, label='Training Data', color='blue')

# Plot the actual test data
ax.plot(df_test['datetime'], y_test, label='Test Data', color='green')

# Plot the predictions for the test data (predictions must align with test data)
ax.plot(df_test['datetime'], y_pred, label='Predictions', color='red')

# Formatting the plot
plt.gcf().autofmt_xdate()
ax.set_title('SARIMA Model Predictions vs Actual Sales')
ax.set_xlabel('Date')
ax.set_ylabel('Sales (Money)')
ax.legend()
st.pyplot(fig)

#=======================4-Forecasting========================#
st.subheader("Forecasting Future Coffee Sales")

n_steps = 30  # number of future steps to forecast (e.g., 30 days)
forecast = sarima_fit.get_forecast(steps=n_steps)

# Extract forecasted values
forecasted_values = forecast.predicted_mean
confidence_intervals = forecast.conf_int()

# Generate future dates for forecasting
last_date = df_test['datetime'].max()
forecast_dates = pd.date_range(last_date, periods=n_steps + 1, freq='D')[1:]

# Plot the forecasted values with confidence intervals
fig_forecast, ax_forecast = plt.subplots(figsize=(10, 5))

# Plot the forecasted values
ax_forecast.plot(forecast_dates, forecasted_values, label='Forecasted Sales', color='orange')

# Plot confidence intervals
ax_forecast.fill_between(forecast_dates, confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1], color='orange', alpha=0.3)

# Formatting the plot
plt.gcf().autofmt_xdate()
ax_forecast.set_title('Forecasted Coffee Sales for Next 30 Days')
ax_forecast.set_xlabel('Date')
ax_forecast.set_ylabel('Sales (Money)')
ax_forecast.legend()
st.pyplot(fig_forecast)

st.write(forecasted_values)
