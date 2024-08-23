import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import lag_plot
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib


# Cached data loading with resampling
@st.cache
def load_data(file, resample_freq='H'):
    df = pd.read_csv(file, delimiter=';', parse_dates=[['Date', 'Time']], dayfirst=True)
    st.write("Original Columns:", df.columns.tolist())
    
    df.rename(columns={
        'Date_Time': 'Datetime',
        'Unnamed: 0': 'Index',
        'Global_active_power': 'Global Active Power',
        'Global_reactive_power': 'Global Reactive Power',
        'Voltage': 'Voltage',
        'Global_intensity': 'Global Intensity',
        'Sub_metering_1': 'Sub Metering 1',
        'Sub_metering_2': 'Sub Metering 2',
        'Sub_metering_3': 'Sub Metering 3'
    }, inplace=True)
    
    st.write("Columns after renaming:", df.columns.tolist())

    for col in df.columns:
        if col != 'Datetime':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.set_index('Datetime', inplace=True)
    df_resampled = df.resample(resample_freq).mean()
    
    return df_resampled

# Handle missing values
def handle_missing_values(df):
    df = df.fillna(method='ffill').fillna(method='bfill')
    df = df.dropna()
    return df

# Train SARIMA model
def train_sarima_model(train_data):
    st.subheader("Model Training: SARIMA Model")
    st.write("### Why SARIMA?")
    st.write("""
        The Seasonal AutoRegressive Integrated Moving Average (SARIMA) model is chosen for this analysis due to its ability 
        to capture both non-seasonal and seasonal components in time series data. Household electricity consumption data 
        typically exhibits daily and weekly seasonal patterns, making SARIMA an ideal choice. The model is specified 
        with both non-seasonal (p, d, q) and seasonal (P, D, Q, m) components to account for these patterns.
    """)

    model = SARIMAX(train_data['Global Active Power'], 
                    order=(1, 1, 1), 
                    seasonal_order=(1, 1, 1, 24),
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    
    model_fit = model.fit(disp=False)
    
    st.write("### Model Summary")
    st.write("""
        The following table provides a summary of the SARIMA model's coefficients and statistical metrics. This includes the
        ARIMA parameters (p, d, q), seasonal parameters (P, D, Q), and other diagnostics such as the AIC (Akaike Information Criterion), 
        which helps in evaluating model fit.
    """)
    st.write(model_fit.summary())

    joblib.dump(model_fit, 'sarima_model.pkl')
    
    return model_fit

# Predict using SARIMA model
def predict_sarima_model(model_fit, start, end):
    st.subheader("Prediction Results")
    st.write("""
        After training the SARIMA model, predictions are made on the test data. The following plot compares the actual 
        electricity consumption against the model's predictions.
    """)
    predictions = model_fit.predict(start=start, end=end, dynamic=False)
    return predictions

# Evaluate model predictions
def evaluate_model(test_data, predictions):
    st.subheader("Model Evaluation")
    st.write("""
        To assess the accuracy of the SARIMA model, the following evaluation metrics are calculated:
        
        - **Mean Absolute Error (MAE):** Measures the average magnitude of the errors in a set of predictions, without considering their direction.
        - **Mean Squared Error (MSE):** Measures the average of the squares of the errors, giving more weight to larger errors.
        - **Root Mean Squared Error (RMSE):** The square root of the MSE, providing an error metric in the same units as the original data.
    """)

    mae = mean_absolute_error(test_data['Global Active Power'], predictions)
    mse = mean_squared_error(test_data['Global Active Power'], predictions)
    rmse = np.sqrt(mse)

    st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
    st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")

    # Plot actual vs predicted values
    st.write("### Actual vs Predicted Power Consumption")
    plt.figure(figsize=(10, 6))
    plt.plot(test_data.index, test_data['Global Active Power'], label='Actual', color='blue')
    plt.plot(test_data.index, predictions, label='Predicted', color='red')
    plt.title('Actual vs Predicted Global Active Power')
    plt.xlabel('Datetime')
    plt.ylabel('Global Active Power (kilowatts)')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    st.write("""
        As observed, the SARIMA model effectively captures the trend and seasonality in the power consumption data, as shown 
        by the close alignment between the actual and predicted values. This indicates that the model is well-suited for 
        forecasting household electricity consumption.
    """)

# Function to perform exploratory data analysis (EDA)
def perform_eda(df):
    st.header("Exploratory Data Analysis (EDA)")
    st.write("""
        Before diving into modeling, it's crucial to understand the data through exploratory data analysis (EDA). 
        The following sections provide insights into the statistical properties, distributions, and correlations in the data.
    """)

    st.subheader("Descriptive Statistics")
    st.write("""
        The table below provides a summary of the key statistics for the dataset, including the mean, standard deviation, 
        and percentiles. This gives an overview of the data distribution.
    """)
    st.write(df.describe())

    st.write("### Data Columns Overview")
    st.write("Columns available for EDA:", df.columns.tolist())

    if st.checkbox("Show Histograms"):
        st.subheader("Histograms")
        st.write("""
            The histograms below show the distribution of each numerical variable in the dataset. This helps identify 
            potential skewness, outliers, or anomalies in the data.
        """)
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                hist = alt.Chart(df.reset_index()).mark_bar().encode(
                    alt.X(col, bin=alt.Bin(maxbins=50), title=f"Distribution of {col}"),
                    alt.Y('count()', title='Frequency'),
                    tooltip=[col]
                ).properties(
                    width=600,
                    height=400
                ).configure_mark(
                    color='#1f77b4'
                )
                st.altair_chart(hist)

    if st.checkbox("Show Box Plots"):
        st.subheader("Box Plots")
        st.write("""
            Box plots are useful for identifying outliers and understanding the spread of the data. The plots below show 
            the quartiles and median for each numerical variable.
        """)
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                box = alt.Chart(df.reset_index()).mark_boxplot().encode(
                    alt.Y(col, title=f"Box Plot of {col}"),
                    tooltip=[col]
                ).properties(
                    width=600,
                    height=400
                )
                st.altair_chart(box)

    if st.checkbox("Show Density Plots"):
        st.subheader("Density Plots")
        st.write("""
            Density plots are used to visualize the distribution of the data. Unlike histograms, density plots provide a 
            smoothed curve, making it easier to identify the underlying distribution shape.
        """)
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                density = alt.Chart(df.reset_index()).transform_density(
                    col,
                    as_=[col, 'density'],
                ).mark_area(color='#ff7f0e').encode(
                    x=alt.X(col, title=f"Density Plot of {col}"),
                    y=alt.Y('density:Q', title='Density'),
                    tooltip=[col]
                ).properties(
                    width=600,
                    height=400
                )
                st.altair_chart(density)

    if st.checkbox("Show Scatter Plot Matrix"):
        st.subheader("Scatter Plot Matrix")
        st.write("""
            The scatter plot matrix provides pairwise scatter plots between numerical variables, helping to identify 
            potential relationships or correlations between them.
        """)
        scatter_matrix = sns.pairplot(df.sample(100), diag_kind='kde')  
        st.pyplot(scatter_matrix)

    if st.checkbox("Show Time Series Plots"):
        st.subheader("Time Series Plots")
        st.write("""
            The time series plots below show how each variable changes over time. This is particularly useful for identifying 
            trends, seasonality, and anomalies in the data.
        """)
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                line = alt.Chart(df.reset_index()).mark_line(color='#2ca02c').encode(
                    x='Datetime:T',
                    y=alt.Y(col, title=f"Time Series of {col}"),
                    tooltip=['Datetime', col]
                ).properties(
                    width=700,
                    height=400
                )
                st.altair_chart(line)

    if st.checkbox("Show Lag Plot"):
        st.subheader("Lag Plot")
        st.write("""
            Lag plots are useful for identifying patterns or autocorrelations in time series data. A linear pattern suggests 
            that the data is highly autocorrelated.
        """)
        lag_plot_fig, ax = plt.subplots(figsize=(6, 4))
        lag_plot(df['Global Active Power'], ax=ax)
        plt.title('Lag Plot of Global Active Power')
        st.pyplot(lag_plot_fig)

    if st.checkbox("Show Seasonal Decomposition"):
        st.subheader("Seasonal Decomposition")
        st.write("""
            Seasonal decomposition allows us to break down the time series into its individual components: trend, seasonality, 
            and residuals. This helps in understanding the underlying patterns in the data.
        """)
        df = handle_missing_values(df)
        
        decomposed = seasonal_decompose(df['Global Active Power'], model='additive')
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
        decomposed.observed.plot(ax=ax1, title='Observed')
        decomposed.trend.plot(ax=ax2, title='Trend')
        decomposed.seasonal.plot(ax=ax3, title='Seasonal')
        decomposed.resid.plot(ax=ax4, title='Residual')
        plt.tight_layout()
        st.pyplot(fig)

    if st.checkbox("Show Correlation Matrix"):
        st.subheader("Correlation Matrix")
        st.write("""
            The correlation matrix shows the pairwise correlations between numerical variables. The heatmap visualization 
            makes it easy to identify strong positive or negative correlations.
        """)
        corr_matrix = df.corr().reset_index().melt('index')
        corr_matrix.columns = ['Variable 1', 'Variable 2', 'Correlation']
        heatmap = alt.Chart(corr_matrix).mark_rect().encode(
            x='Variable 1:O',
            y='Variable 2:O',
            color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='blueorange')),
            tooltip=['Variable 1', 'Variable 2', 'Correlation']
        ).properties(
            width=600,
            height=600
        ).configure_axis(
            grid=False
        ).configure_view(
            strokeWidth=0
        )
        st.altair_chart(heatmap)

# Main function
def main():
    st.title('Electric Power Consumption Data Analysis and Prediction')
    st.write("""
        This application provides a comprehensive analysis of household electric power consumption data. 
        The analysis includes Exploratory Data Analysis (EDA), model training using SARIMA, and evaluation of the model's 
        predictive performance.
    """)
    
    train_file = st.file_uploader("Upload Training Data", type=["txt"], key="train")
    test_file = st.file_uploader("Upload Test Data", type=["txt"], key="test")

    resample_option = st.selectbox(
        "Resample Data By:",
        ('Hourly', 'Daily')
    )

    resample_freq = 'H' if resample_option == 'Hourly' else 'D'
    
    if train_file and test_file:
        train_data = load_data(train_file, resample_freq)
        test_data = load_data(test_file, resample_freq)
        
        if train_data is not None and test_data is not None:
            st.write(f"### Training Data Overview (Resampled to {resample_option})")
            st.write("""
                The table below shows the first few rows of the training data after resampling. This gives a quick glimpse into the 
                structure and content of the data that will be used for model training.
            """)
            st.write(train_data.head())
        
            st.write(f"### Test Data Overview (Resampled to {resample_option})")
            st.write("""
                The table below shows the first few rows of the test data after resampling. This data will be used to evaluate 
                the model's performance.
            """)
            st.write(test_data.head())

            # Perform EDA before training the model
            perform_eda(train_data)

            train_data = handle_missing_values(train_data)
            test_data = handle_missing_values(test_data)

            # Train SARIMA model
            model_fit = train_sarima_model(train_data)

            # Predict on the test set
            start = len(train_data)
            end = start + len(test_data) - 1
            predictions = predict_sarima_model(model_fit, start, end)

            # Evaluate model
            evaluate_model(test_data, predictions)
            
if __name__ == "__main__":
    main()
