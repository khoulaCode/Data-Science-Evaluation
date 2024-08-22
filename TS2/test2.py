import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from prophet import Prophet

class TimeSeries:
    def __init__(self, train_df, stores_df, features_df):
        self.train_df = train_df
        self.stores_df = stores_df
        self.features_df = features_df
        self.merged_df = None

    def explore_data(self):
        """
        Perform exploratory data analysis (EDA) on the provided datasets.
        """
        st.subheader("Training Data Summary")
        st.write(self.train_df.describe())
        
        st.subheader("Stores Data Summary")
        st.write(self.stores_df.describe())
        
        st.subheader("Features Data Summary")
        st.write(self.features_df.describe())

        # Check for missing values
        st.subheader("Missing Values in Training Data")
        st.write(self.train_df.isnull().sum())
        
        st.subheader("Missing Values in Stores Data")
        st.write(self.stores_df.isnull().sum())
        
        st.subheader("Missing Values in Features Data")
        st.write(self.features_df.isnull().sum())

        # Correlation matrix
        st.subheader("Correlation Matrix of Training Data")
        numeric_df = self.train_df.select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
        corr_matrix = numeric_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        st.pyplot(plt.gcf())

        # Distribution of sales
        st.subheader("Sales Distribution")
        sns.histplot(self.train_df['Weekly_Sales'], kde=True, color='blue', bins=30)
        st.pyplot(plt.gcf())

        # Sales distribution over time
        st.subheader("Sales Distribution Over Time")
        sns.lineplot(data=self.train_df, x='Date', y='Weekly_Sales')
        st.pyplot(plt.gcf())

        # Boxplot to identify outliers
        st.subheader("Outliers in Weekly Sales")
        sns.boxplot(x=self.train_df['Weekly_Sales'])
        st.pyplot(plt.gcf())

        # Pairplot for relationships between numerical features
        st.subheader("Pairplot of Numerical Features")
        sns.pairplot(numeric_df)
        st.pyplot(plt.gcf())



    def merge_data(self):
        """
        Merges the train DataFrame with the stores DataFrame and then with the features DataFrame
        based on the 'Store' and 'Date' columns.
        """
        self.train_df.columns = self.train_df.columns.str.strip().str.title()
        self.stores_df.columns = self.stores_df.columns.str.strip().str.title()
        self.features_df.columns = self.features_df.columns.str.strip().str.title()

        self.train_df['Date'] = pd.to_datetime(self.train_df['Date'], errors='coerce')
        self.features_df['Date'] = pd.to_datetime(self.features_df['Date'], errors='coerce')

        merged_store = pd.merge(self.train_df, self.stores_df, on='Store', how='left')
        self.merged_df = pd.merge(merged_store, self.features_df, on=['Store', 'Date'], how='left')

        st.subheader("Merged DataFrame Head")
        st.dataframe(self.merged_df.head())

    def preprocess_for_prophet(self):
        """
        Prepares the data for Prophet modeling by renaming columns to 'ds' and 'y'.
        """
        if self.merged_df is not None:
            df = self.merged_df.groupby('Date')['Weekly_Sales'].sum().reset_index()
            df.rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'}, inplace=True)
            return df
        else:
            st.error("DataFrames are not merged yet. Please merge the DataFrames first.")
            return None

    def fit_predict_prophet(self, train_df, test_df):
        """
        Fits a Prophet model on the training data and makes predictions on the test data.
        """
        prophet_model = Prophet()
        prophet_model.fit(train_df)

        future = test_df[['ds']].copy()
        forecast = prophet_model.predict(future)

        # Calculate evaluation metrics
        y_true = test_df['y'].values
        y_pred = forecast['yhat'].values

        mse = mean_squared_error(y_true, y_pred)
        rmse = sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        st.subheader("Prophet Model Evaluation")
        st.write(f"RMSE: {rmse}")
        st.write(f"MSE: {mse}")
        st.write(f"R²: {r2}")

        # Plot the predictions
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(test_df['ds'], y_true, label='Actual Sales', color='blue')
        ax.plot(test_df['ds'], y_pred, label='Forecasted Sales', color='orange')
        ax.set_title('Prophet Model - Actual vs Forecasted Sales')
        ax.set_xlabel('Date')
        ax.set_ylabel('Weekly Sales')
        ax.legend()
        st.pyplot(fig)

    def fit_predict_sarima(self, train_df, test_df, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52)):
        """
        Fits a SARIMA model on the training data and makes predictions on the test data.
        """
        train_series = train_df.set_index('ds')['y']
        test_series = test_df.set_index('ds')['y']

        sarima_model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order)
        sarima_result = sarima_model.fit(disp=False)

        # Forecast
        forecast = sarima_result.get_forecast(steps=len(test_series))
        y_pred = forecast.predicted_mean
        y_true = test_series

        # Calculate evaluation metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        st.subheader("SARIMA Model Evaluation")
        st.write(f"RMSE: {rmse}")
        st.write(f"MSE: {mse}")
        st.write(f"R²: {r2}")

        # Plot the predictions
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(test_series.index, y_true, label='Actual Sales', color='blue')
        ax.plot(test_series.index, y_pred, label='Forecasted Sales', color='orange')
        ax.set_title('SARIMA Model - Actual vs Forecasted Sales')
        ax.set_xlabel('Date')
        ax.set_ylabel('Weekly Sales')
        ax.legend()
        st.pyplot(fig)

def main():
    st.title("Time Series Analysis and Forecasting")

    # Upload data files
    train_file = st.file_uploader("Upload Training Data CSV", type="csv")
    stores_file = st.file_uploader("Upload Stores Data CSV", type="csv")
    features_file = st.file_uploader("Upload Features Data CSV", type="csv")

    if train_file and stores_file and features_file:
        # Read the uploaded files into pandas DataFrames
        train_df = pd.read_csv(train_file)
        stores_df = pd.read_csv(stores_file)
        features_df = pd.read_csv(features_file)

        # Initialize analysis object
        ts_analysis = TimeSeries(train_df, stores_df, features_df)

        # Perform Data Exploration
        ts_analysis.explore_data()

        # Merge and preprocess the training data
        ts_analysis.merge_data()
        train_prophet_df = ts_analysis.preprocess_for_prophet()

        # Prepare test data
        ts_analysis2 = TimeSeries(train_df, stores_df, features_df)
        ts_analysis2.merge_data()
        test_prophet_df = ts_analysis2.preprocess_for_prophet()

        # Fit and predict with Prophet
        if train_prophet_df is not None and test_prophet_df is not None:
            ts_analysis.fit_predict_prophet(train_prophet_df, test_prophet_df)

        # Fit and predict with SARIMA
        if train_prophet_df is not None and test_prophet_df is not None:
            ts_analysis.fit_predict_sarima(train_prophet_df, test_prophet_df)

if __name__ == "__main__":
    main()
