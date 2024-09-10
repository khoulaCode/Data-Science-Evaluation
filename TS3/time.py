import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# Set Streamlit page configuration
st.set_page_config(page_title='Coffee Sales Time Series Analysis', layout='wide')

# Title of the application
st.title('Coffee Sales Time Series Analysis')

# Load data from CSV files in the same directory as the .py file
train_data = pd.read_csv('Train Coffee Sales.csv')
test_data = pd.read_csv('Test Coffee Sales.csv')

# Display Data Overview
st.header('Data Overview')

col1, col2 = st.columns(2)

with col1:
    st.subheader('Training Data')
    st.write(train_data.head())
    st.write('Training Data Shape:', train_data.shape)

with col2:
    st.subheader('Test Data')
    st.write(test_data.head())
    st.write('Test Data Shape:', test_data.shape)

# Data Preprocessing
st.header('Data Preprocessing')
st.write('Cleaning and preprocessing the data...')

# Drop unnecessary columns
columns_to_drop = ['card', 'datetime', 'cash_type', 'coffee_name']
train_data.drop(columns=columns_to_drop, inplace=True)
test_data.drop(columns=columns_to_drop, inplace=True)

st.write(f'Dropped columns: {columns_to_drop}')

# Convert date columns to datetime format
train_data['date'] = pd.to_datetime(train_data['date'], format='%Y-%m-%d', errors='coerce')
test_data['date'] = pd.to_datetime(test_data['date'], format='%Y-%m-%d', errors='coerce')

# Handle missing values
st.write('Handling missing values...')
train_data.dropna(inplace=True)
test_data.dropna(inplace=True)

st.write('Missing values handled. Here are the data shapes after cleaning:')
col1, col2 = st.columns(2)
with col1:
    st.write('Training Data Shape:', train_data.shape)
with col2:
    st.write('Test Data Shape:', test_data.shape)

# Feature Engineering
st.header('Feature Engineering')
st.write('Deriving new features from the data...')

for df in [train_data, test_data]:
    # Extracting time-based features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek

    # Sorting by date to ensure proper lagging
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)

# Creating lag features on training data
train_data['lag1'] = train_data['money'].shift(1)
train_data['lag7'] = train_data['money'].shift(7)
train_data['rolling_mean_7'] = train_data['money'].rolling(window=7).mean()

# Drop rows with NaN values resulted from lag features
train_data.dropna(inplace=True)

st.write('Feature engineering complete. Here are the first few rows of the training data:')
st.write(train_data.head())

# Exploratory Data Analysis
st.header('Exploratory Data Analysis')

# Sales Over Time
st.subheader('Sales Over Time')
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train_data['date'], train_data['money'], label='Sales')
ax.set_title('Sales Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Sales (Money)')
ax.legend()
st.pyplot(fig)

# Sales Distribution by Coffee Type
st.subheader('Sales Distribution by Coffee Type')
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=train_data, x='month', ax=ax)  # Adjusted to plot month since coffee_name was dropped
ax.set_title('Sales Distribution by Month')
ax.set_xlabel('Month')
ax.set_ylabel('Count')
plt.xticks(rotation=45)
st.pyplot(fig)

# ACF and PACF Plots
st.subheader('Autocorrelation and Partial Autocorrelation Plots')
fig, ax = plt.subplots(2, 1, figsize=(12, 8))

# ACF Plot
plot_acf(train_data['money'], ax=ax[0], lags=40)
ax[0].set_title('Autocorrelation Function (ACF)')

# PACF Plot
plot_pacf(train_data['money'], ax=ax[1], lags=40)
ax[1].set_title('Partial Autocorrelation Function (PACF)')

plt.tight_layout()
st.pyplot(fig)

# Decomposition of the time series
st.subheader('Time Series Decomposition')
decomposition = sm.tsa.seasonal_decompose(train_data['money'], model='additive', period=7)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))

# Plot the observed data
decomposition.observed.plot(ax=ax1)
ax1.set_ylabel('Observed')

# Plot the trend component
decomposition.trend.plot(ax=ax2)
ax2.set_ylabel('Trend')

# Plot the seasonal component
decomposition.seasonal.plot(ax=ax3)
ax3.set_ylabel('Seasonal')

# Plot the residual component
decomposition.resid.plot(ax=ax4)
ax4.set_ylabel('Residual')

plt.tight_layout()
st.pyplot(fig)



# Model Building and Forecasting
st.header('Model Building and Forecasting')

# Model selection
model_option = st.selectbox('Select a model to build:', ('Prophet', 'ARIMA', 'SARIMA'))

if model_option in ['ARIMA', 'SARIMA']:
    st.write(f'Building {model_option} model...')

    # Determine if seasonal components are needed
    if model_option == 'SARIMA':
        seasonal = True
        m = st.number_input('Enter the seasonal period (m):', min_value=1, max_value=365, value=12)
    else:
        seasonal = False
        m = 1  # Non-seasonal

    # Automatically determine the best parameters using auto_arima
    try:
        with st.spinner('Fitting the model...'):
            model = pm.auto_arima(
                train_data['money'],
                start_p=1, start_q=1,
                max_p=5, max_q=5,
                d=None,              # Let auto_arima determine 'd'
                seasonal=seasonal,
                m=m if seasonal else 1,
                start_P=0, start_Q=0,
                max_P=2, max_Q=2,
                D=1 if seasonal else 0,  # Seasonal differencing
                trace=True,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )

        st.write('Best Model Parameters:')
        st.write(f'Order: {model.order}')
        if seasonal:
            st.write(f'Seasonal Order: {model.seasonal_order}')
        st.write(f'AIC: {model.aic()}')

        # Forecasting
        st.write('Generating forecasts...')
        n_periods = len(test_data)
        forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)

        # Assign forecasts to test_data
        forecast_series = pd.Series(forecast, index=test_data.index)
        test_data[f'forecast_{model_option.lower()}'] = forecast_series

        # Handle any potential NaN values in forecasts
        test_data[f'forecast_{model_option.lower()}'].fillna(method='ffill', inplace=True)

        st.write(f'{model_option} Forecast')
        st.write(test_data[['date', 'money', f'forecast_{model_option.lower()}']].head())

        # Evaluation
        st.header('Evaluation Metrics')
        mse = mean_squared_error(test_data['money'], test_data[f'forecast_{model_option.lower()}'])
        mae = mean_absolute_error(test_data['money'], test_data[f'forecast_{model_option.lower()}'])
        r2 = r2_score(test_data['money'], test_data[f'forecast_{model_option.lower()}'])

        st.write(f'Mean Squared Error (MSE): {mse:.2f}')
        st.write(f'Mean Absolute Error (MAE): {mae:.2f}')
        st.write(f'R² Score: {r2:.2f}')

        # Forecast Visualization (Show all previous data along with predicted)
        st.subheader('Forecast Visualization')
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot training data
        ax.plot(train_data['date'], train_data['money'], label='Historical Data (Train)', color='blue')
        
        # Plot test data (actual)
        ax.plot(test_data['date'], test_data['money'], label='Actual Data (Test)', color='green')
        
        # Plot forecasted data
        ax.plot(test_data['date'], test_data[f'forecast_{model_option.lower()}'], label='Forecasted Data', color='red')

        ax.fill_between(
            test_data['date'],
            conf_int[:, 0],
            conf_int[:, 1],
            color='pink',
            alpha=0.3,
            label='Confidence Interval'
        )
        
        ax.set_title(f'Historical vs Forecasted Sales ({model_option})')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sales (Money)')
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f'An error occurred while building the {model_option} model: {e}')

elif model_option == 'Prophet':
    st.write('Building Prophet model...')

    try:
        with st.spinner('Fitting the Prophet model...'):
            # Prepare data for Prophet
            prophet_train = train_data[['date', 'money']].rename(columns={'date': 'ds', 'money': 'y'})

            prophet_model = Prophet()
            prophet_model.fit(prophet_train)

            # Create future dataframe
            future = prophet_model.make_future_dataframe(periods=len(test_data), freq='D')
            forecast = prophet_model.predict(future)

            # Extract forecast for the test period
            forecast_test = forecast.tail(len(test_data)).set_index(test_data.index)

            # Assign forecasts to test_data
            test_data['forecast_prophet'] = forecast_test['yhat'].values

        st.write('Prophet Forecast')
        st.write(test_data[['date', 'money', 'forecast_prophet']].head())

        # Evaluation
        st.header('Evaluation Metrics')
        mse = mean_squared_error(test_data['money'], test_data['forecast_prophet'])
        mae = mean_absolute_error(test_data['money'], test_data['forecast_prophet'])
        r2 = r2_score(test_data['money'], test_data['forecast_prophet'])

        st.write(f'Mean Squared Error (MSE): {mse:.2f}')
        st.write(f'Mean Absolute Error (MAE): {mae:.2f}')
        st.write(f'R² Score: {r2:.2f}')

        # Forecast Visualization (Show all previous data along with predicted)
        st.subheader('Forecast Visualization')
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot training data
        ax.plot(train_data['date'], train_data['money'], label='Historical Data (Train)', color='blue')
        
        # Plot test data (actual)
        ax.plot(test_data['date'], test_data['money'], label='Actual Data (Test)', color='green')
        
        # Plot forecasted data
        ax.plot(test_data['date'], test_data['forecast_prophet'], label='Forecasted Data', color='red')

        ax.set_title('Historical vs Forecasted Sales (Prophet)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sales (Money)')
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f'An error occurred while building the Prophet model: {e}')
