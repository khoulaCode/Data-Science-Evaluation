import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from streamlit_option_menu import option_menu
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tools.eval_measures import aic, bic
from prophet import Prophet
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Function to fill missing values based on Year, Diphtheria
def fill_missing_values_yd(df, kaggle_df):
    # Remove leading and trailing spaces from column names
    df.columns = df.columns.str.strip()
    kaggle_df.columns = kaggle_df.columns.str.strip()

    # Merge operation with explicit column naming
    merged_df = pd.merge(
        df, 
        kaggle_df[['Country', 'Year', 'Population_mln']], 
        how='left', 
        on=['Country', 'Year']
    )

    # Convert Population_mln to Population in merged_df
    merged_df['Population_kaggle'] = merged_df['Population_mln'] * 1_000_000

    # Fill missing Population values in df with the values from merged_df
    df['Population'] = df['Population'].fillna(merged_df['Population_kaggle'])

    # Additional filling logic based on Year and Diphtheria
    for col in ['Population', 'GDP', 'Hepatitis B']:
        for index, row in df[df[col].isnull()].iterrows():
            matching_rows = df[
                (df['Year'] == row['Year']) & 
                (df['Diphtheria'] == row['Diphtheria']) & 
                df[col].notnull()
            ]
            if not matching_rows.empty:
                df.at[index, col] = matching_rows.iloc[0][col]

    return df

# Load the datasets
df_train = pd.read_csv('train Life Expectancy Data.csv')
df_test = pd.read_csv('test Life Expectancy Data.csv')
df_kaggle = pd.read_csv('Life-Expectancy-Data-Updated.csv')

# Calculate missing values before filling for both datasets
missing_population_before = df_train['Population'].isnull().sum()
missing_gdp_before = df_train['GDP'].isnull().sum()
missing_hep_before = df_train['Hepatitis B'].isnull().sum()

missing_population_before_test = df_test['Population'].isnull().sum()
missing_gdp_before_test = df_test['GDP'].isnull().sum()
missing_hep_before_test = df_test['Hepatitis B'].isnull().sum()

# Apply the function to both train and test datasets
df_train_filled = fill_missing_values_yd(df_train, df_kaggle)
df_test_filled = fill_missing_values_yd(df_test, df_kaggle)

# Calculate missing values after filling for both datasets
missing_population_after_final_fill_train = df_train_filled['Population'].isnull().sum()
missing_gdp_after_final_fill_train = df_train_filled['GDP'].isnull().sum()
missing_hep_after_final_fill_train = df_train_filled['Hepatitis B'].isnull().sum()

missing_population_after_final_fill_test = df_test_filled['Population'].isnull().sum()
missing_gdp_after_final_fill_test = df_test_filled['GDP'].isnull().sum()
missing_hep_after_final_fill_test = df_test_filled['Hepatitis B'].isnull().sum()

# Fill remaining missing values with column mean
df_train.fillna(df_train.mean(numeric_only=True), inplace=True)
df_test.fillna(df_train.mean(numeric_only=True), inplace=True)

# Load the datasets
df_train_ts_original = pd.read_csv('train.csv')
df_test_ts_original = pd.read_csv('test.csv')
df_features = pd.read_csv('features.csv')
df_stores = pd.read_csv('stores.csv')

def merge_with_stores(train, test, stores):
    """
    Merge the train and test datasets with the stores dataset to add store-specific information.
    """
    train_merged = pd.merge(train, stores, on='Store', how='left')
    test_merged = pd.merge(test, stores, on='Store', how='left')
    return train_merged, test_merged

def merge_with_features(train, test, features):
    """
    Merge the train and test datasets with the features dataset to include additional feature information.
    """
    train_merged = pd.merge(train, features, on=['Store', 'Date'], how='left')
    test_merged = pd.merge(test, features, on=['Store', 'Date'], how='left')
    return train_merged, test_merged

def fill_missing_values(df_train, df_test):
    """
    Fill missing values for CPI and Unemployment in the test dataset using the mean of each store from the training dataset.
    """
    # Calculate mean values for CPI and Unemployment for each store
    store_means = df_train.groupby('Store')[['CPI', 'Unemployment']].mean().reset_index()

    # Merge the store means with the test dataset
    df_test = pd.merge(df_test, store_means, on='Store', suffixes=('', '_mean'))

    # Fill missing values in CPI and Unemployment with the store mean values
    df_test['CPI'].fillna(df_test['CPI_mean'], inplace=True)
    df_test['Unemployment'].fillna(df_test['Unemployment_mean'], inplace=True)

    # Drop the mean columns used for filling
    df_test.drop(columns=['CPI_mean', 'Unemployment_mean'], inplace=True)

    return df_test

def fill_markdown_missing_values(df_train, df_test):
    """
    Fill missing values for MarkDown columns in both training and test datasets with 0.
    """
    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']

    # Fill missing values with 0 in the training dataset
    for col in markdown_cols:
        df_train[col].fillna(0, inplace=True)

    # Fill missing values with 0 in the test dataset
    for col in markdown_cols:
        df_test[col].fillna(0, inplace=True)

    return df_train, df_test

def display_date_column(dataset, dataset_name):
    if 'Date' in dataset.columns:
        combined_dates = pd.concat([dataset['Date'].head(5), dataset['Date'].tail(5)])
        st.write(combined_dates)
    else:
        st.write(f"No 'Date' column in {dataset_name} dataset")

def one_hot_encode_columns(df_train, df_test, columns, prefix):
    for column in columns:
        # Perform one-hot encoding on the train set
        df_train = pd.get_dummies(df_train, columns=[column], prefix=prefix)
        # Perform one-hot encoding on the test set with the same categories
        df_test = pd.get_dummies(df_test, columns=[column], prefix=prefix)
        
        # Align the test set with the train set to ensure both have the same columns
        df_train, df_test = df_train.align(df_test, join='outer', axis=1, fill_value=0)
    
    # Drop the original boolean columns
    for column in columns:
        if column in df_train.columns:
            df_train = df_train.drop(columns=[column])
        if column in df_test.columns:
            df_test = df_test.drop(columns=[column])

    return df_train, df_test

def seasonal_decomposition_plot(series, freq):
    """
    Perform and plot seasonal decomposition of a time series.
    """
    decomposition = seasonal_decompose(series, model='additive', period=freq)
    plt.figure(figsize=(14, 10))
    plt.subplot(4, 1, 1)
    plt.plot(decomposition.observed)
    plt.title('Observed')
    plt.subplot(4, 1, 2)
    plt.plot(decomposition.trend)
    plt.title('Trend')  
    plt.subplot(4, 1, 3)
    plt.plot(decomposition.seasonal)
    plt.title('Seasonal')
    plt.subplot(4, 1, 4)
    plt.plot(decomposition.resid)
    plt.title('Residual')
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)

def plot_acf_pacf(series, lags=50):
    """
    Plots the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) of a time series vertically.
    """
    fig, ax = plt.subplots(2, 1, figsize=(10, 12))

    # Plot ACF
    plot_acf(series, lags=lags, ax=ax[0])
    ax[0].set_title('Autocorrelation Function (ACF)')
    ax[0].set_xlabel('Lags')
    ax[0].set_ylabel('Autocorrelation')
    
    # Plot PACF
    plot_pacf(series, lags=lags, ax=ax[1])
    ax[1].set_title('Partial Autocorrelation Function (PACF)')
    ax[1].set_xlabel('Lags')
    ax[1].set_ylabel('Partial Autocorrelation')
    
    # Adjust layout and render the plot
    plt.tight_layout()
    st.subheader("Autocorrelation and Partial Autocorrelation Function Plots")
    st.pyplot(fig)

def auto_regressive_analysis(series, max_lag=30):
    """
    Perform Auto-Regressive (AR) Analysis to find the best lag based on AIC and BIC.
    """
    best_aic = float('inf')
    best_bic = float('inf')
    best_aic_lag = None
    best_bic_lag = None
    
    for lag in range(1, max_lag + 1):
        try:
            model = AutoReg(series, lags=lag).fit()
            model_aic = model.aic
            model_bic = model.bic
            if model_aic < best_aic:
                best_aic = model_aic
                best_aic_lag = lag
            if model_bic < best_bic:
                best_bic = model_bic
                best_bic_lag = lag
        except Exception as e:
            print(f"Failed for lag {lag}: {e}")
            continue
    
    return best_aic_lag, best_bic_lag

def moving_average_analysis(series, windows):
    """
    Calculate and return moving averages for given window sizes.
    """
    ma_results = {}
    for window in windows:
        ma_results[f'MA-{window}'] = series.rolling(window=window).mean()

    return ma_results

def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"File {file_path} not found.")
        return pd.DataFrame()

def fit_prophet_model(train_df, regressors=None):
    model = Prophet(daily_seasonality=True, 
                    weekly_seasonality=True, 
                    yearly_seasonality=True, 
                    seasonality_mode='multiplicative', 
                    changepoint_prior_scale=0.1)
    
    if regressors:
        for regressor in regressors:
            model.add_regressor(regressor)
    
    model.fit(train_df)
    return model

def forecast_train_data(model, train_df):
    future_train = train_df[['ds']].copy()
    forecast_train = model.predict(future_train)
    return forecast_train

def forecast_test_data(model, test_df):
    future_test = test_df[['ds']].copy()
    forecast_test = model.predict(future_test)
    return forecast_test

def fit_predict_prophet(train_df, test_df, regressors=None):
    """
    Fits a Prophet model on the training data and makes predictions on the test data.
    Calculates RMSE, MSE, and R² for the training data predictions.
    """
    # Normalize target variable
    scaler = MinMaxScaler()
    train_df['y'] = scaler.fit_transform(train_df[['y']])

    # Fit the Prophet model with optional regressors
    model = fit_prophet_model(train_df, regressors)
    
    # Forecast on the training data
    forecast_train = model.predict(train_df[['ds'] + regressors])
    
    # Calculate metrics on the training data
    y_true_train = train_df['y'].values
    y_pred_train = forecast_train['yhat'].values

    mse_train = mean_squared_error(y_true_train, y_pred_train)
    rmse_train = sqrt(mse_train)
    r2_train = r2_score(y_true_train, y_pred_train)

    st.subheader("Prophet Model Training Evaluation Metrics")
    st.write(f"RMSE: {rmse_train}")
    st.write(f"MSE: {mse_train}")
    st.write(f"R²: {r2_train}")

    # Forecast on the test data
    forecast_test = model.predict(test_df[['ds'] + regressors])

# Merge datasets
df_train_ts, df_test_ts = merge_with_stores(df_train_ts_original, df_test_ts_original, df_stores)
df_train_ts, df_test_ts = merge_with_features(df_train_ts, df_test_ts, df_features)

# Create copies before filling missing values
df_train_ts_before = df_train_ts.copy()
df_test_ts_before = df_test_ts.copy()

# Fill missing values
df_test_ts = fill_missing_values(df_train_ts, df_test_ts)
df_train_ts, df_test_ts = fill_markdown_missing_values(df_train_ts, df_test_ts)

# Create copies after filling missing values
df_train_ts_after = df_train_ts.copy()
df_test_ts_after = df_test_ts.copy()

# Set up the main title
st.title("Data Analysis and Forecasting")

# Sidebar with options menu
with st.sidebar:
    selected = option_menu(
        'Evaluation', 
        ["Linear Regression", "Time Series"],
        icons=['bar-chart', 'chart-line'], 
        menu_icon='cast', 
        default_index=0
    )

# Linear Regression Tab
if selected == "Linear Regression":
    st.subheader("Data Overview")
    st.write("""
    This application explores the WHO Life Expectancy dataset, including country-specific health indicators such as life expectancy, mortality rates, and health expenditures. The dataset features variables like GDP, immunization coverage, and disease prevalence across different years. Explore detailed insights into dataset structure, summary statistics, and missing values to understand factors affecting life expectancy.
    """)

    st.header("Exploratory Data Analysis (EDA) on Training Dataset")
    
    st.write("Here’s a quick look at the training dataset:")
    st.write(df_train.head())

    st.subheader("Data Summary")
    st.write("Summary statistics of the training dataset, including counts, means, standard deviations, and percentiles, help understand the overall distribution and central tendencies of the data.")
    st.write(df_train.describe())

    st.subheader("Missing Data Before and After Merging with a Relevant Dataset")    
    # Create a DataFrame to hold the missing values before and after filling
    data = {
        'Data': ['Before Merging', 'After Merging'],
        'Training Population': [missing_population_before, missing_population_after_final_fill_train],
        'Training GDP': [missing_gdp_before, missing_gdp_after_final_fill_train],
        'Training Hepatitis B': [missing_hep_before, missing_hep_after_final_fill_train],
        'Testing Population': [missing_population_before_test, missing_population_after_final_fill_test],
        'Testing GDP': [missing_gdp_before_test, missing_gdp_after_final_fill_test],
        'Testing Hepatitis B': [missing_hep_before_test, missing_hep_after_final_fill_test]
    }

    # Convert the dictionary into a DataFrame
    df_missing_values = pd.DataFrame(data)

    # Dropdown menu for selecting between training and testing data
    option = st.selectbox(
        'Select Data to Display',
        ('Training Data', 'Testing Data')
    )

    # Display the table based on the selection
    if option == 'Training Data':
        st.subheader("Training Data Missing Values")
        st.table(df_missing_values[['Data', 'Training Population', 'Training GDP', 'Training Hepatitis B']])
    else:
        st.subheader("Testing Data Missing Values")
        st.table(df_missing_values[['Data', 'Testing Population', 'Testing GDP', 'Testing Hepatitis B']])

    # Calculate missing values
    missing_values = df_test.isnull().sum()
    missing_values_df = pd.DataFrame(missing_values[missing_values > 0], columns=['Missing Values'])

    # Display the formatted table
    st.subheader("Missing Values Remaining in the Dataset After Filling with Mean")

    # Sum of all missing values
    total_missing_values = missing_values.sum()
    st.write(f"**Total Missing Values**: {total_missing_values}")

    # Plot correlation heatmap
    st.subheader("Correlation Heatmap")
    numeric_df = df_train.select_dtypes(include=['number'])
    st.write("**Correlation Heatmap**: Shows the correlation coefficients between all numeric variables, helping identify strong relationships.")
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot(plt)

    # Drop the 'under-five deaths' column for high correlation with 'infant deaths' shown in the heatmap
    df_train = df_train.drop(columns=['under-five deaths'], errors='ignore')
    df_test = df_test.drop(columns=['under-five deaths'], errors='ignore')

    # Plot correlation heatmap
    st.subheader("Correlation Heatmap with the 'under-five deaths' column dropped for high correlation")
    numeric_df = df_train.select_dtypes(include=['number'])
    st.write("**Correlation Heatmap**: Shows the correlation coefficients between all numeric variables, helping identify strong relationships.")
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot(plt)

    # Plot pairplot
    st.subheader("Pairplot of Selected Features")
    selected_features = ['Life expectancy', 'Adult Mortality', 'Alcohol', 'BMI', 'GDP']
    numeric_features = [feature for feature in selected_features if feature in numeric_df.columns]
    st.write("**Pairplot**: Displays scatter plots between pairs of features and histograms for individual features, revealing relationships and distributions.")
    sns.pairplot(df_train[numeric_features])
    st.pyplot(plt)

    # Plot histograms of numeric features
    st.subheader("Histograms of Numeric Features")
    numeric_features = numeric_df.columns
    num_plots = len(numeric_features)
    cols = 3
    rows = (num_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    axes = axes.flatten()
    for i, column in enumerate(numeric_features):
        sns.histplot(df_train[column].dropna(), kde=True, ax=axes[i], bins=20)
        axes[i].set_title(f'Histogram of {column}')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    st.write("**Histogram**: Visualizes the distribution of a single feature by displaying the frequency of different value ranges. It helps in understanding the underlying distribution of the data, detecting outliers, and identifying any skewness or patterns within the feature.")
    st.pyplot(fig)

    # Plot box plots of numeric features
    st.subheader("Box Plots of Numeric Features")
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    axes = axes.flatten()
    for i, column in enumerate(numeric_features):
        sns.boxplot(x=df_train[column], ax=axes[i])
        axes[i].set_title(f'Box Plot of {column}')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
    for j in range(i + 1, rows * cols):
        fig.delaxes(axes[j])
    st.write("**Box Plots**: Display the distribution of values with quartiles and potential outliers.")
    plt.tight_layout()
    st.pyplot(fig)

    # Prepare data for Linear Regression
    st.header("Linear Regression Model")
    
    # Choose only relevant columns to be used in the regression model for more accurate results
    features = ['Year', 'Status', 'Adult Mortality', 'infant deaths', 'Alcohol', 
    'percentage expenditure', 'Hepatitis B', 'Measles', 'BMI', 
    'Polio', 'Total expenditure', 'Diphtheria', 'HIV/AIDS', 'GDP', 
    'Population', 'Income composition of resources', 'Schooling']

    # Identify categorical columns
    categorical_cols = df_train.select_dtypes(include=['object', 'category']).columns

    # Initialize label encoder
    label_encoders = {}

    # Apply label encoding to categorical columns
    for col in categorical_cols:
        # Initialize a new LabelEncoder for each column
        label_encoders[col] = LabelEncoder()
        # Fit encoder on training data
        df_train[col] = label_encoders[col].fit_transform(df_train[col])
        # Transform test data using the same encoder
        df_test[col] = df_test[col].map(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1)

    # Handle cases where unseen labels are mapped to -1, which indicates unseen labels
    df_test[categorical_cols] = df_test[categorical_cols].fillna(-1)  # Replace NaN with -1

    # Load training data
    X_train = df_train[features]  # Features from the training CSV
    y_train = df_train['Life expectancy']  # Target variable from the training CSV

    # Load testing data
    X_test = df_test[features]  # Features from the testing CSV
    y_test = df_test['Life expectancy']  # Target variable from the testing CSV

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fit the Linear Regression model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Display model performance metrics
    st.write("**Linear Regression Model Performance**:")
    st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
    st.write(f"R^2 Score: {r2_score(y_test, y_pred)}")

    # Plotting Predicted vs. Real Values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label='Predicted vs. Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Perfect Fit')
    plt.xlabel('Actual Life Expectancy')
    plt.ylabel('Predicted Life Expectancy')
    plt.title('Predicted vs Actual Life Expectancy')
    plt.legend()
    plt.grid(True)

    # Render the plot in Streamlit
    st.pyplot(plt)
    plt.close()

    # Hyperparameter Tuning with GridSearchCV for Ridge Regression
    st.header("Hyperparameter Tuning with Ridge Regression")
    st.write("**Ridge Regression**: Regularization helps control overfitting, but may not capture complex patterns as effectively.")
    
    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    grid_search = GridSearchCV(Ridge(), param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_

    # Evaluate the Grid Search Best Model
    y_pred_best = best_model.predict(X_test_scaled)
    st.write("**Best Model from Grid Search Performance**:")
    st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_best)}")
    st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_best)}")
    st.write(f"R^2 Score: {r2_score(y_test, y_pred_best)}")

    # Plotting Predicted vs. Real Values for the Best Model
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_best, color='green', alpha=0.5, label='Predicted vs. Actual (Best Model)')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Perfect Fit')
    plt.xlabel('Actual Life Expectancy')
    plt.ylabel('Predicted Life Expectancy (Best Model)')
    plt.title('Predicted vs Actual Life Expectancy (Best Model)')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    # Gradient Boosting Regressor
    st.header("Gradient Boosting Regressor")
    st.write("**Gradient Boosting Regressor**: Sequentially builds on errors of previous models to capture complex relationships and improve accuracy.")
    gbr = GradientBoostingRegressor()
    gbr.fit(X_train_scaled, y_train)
    y_pred_gbr = gbr.predict(X_test_scaled)

    st.write("**Gradient Boosting Regressor Performance**:")
    st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_gbr)}")
    st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_gbr)}")
    st.write(f"R^2 Score: {r2_score(y_test, y_pred_gbr)}")

    # Plotting Predicted vs. Real Values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_best, color='navy', alpha=0.5, label='Predicted vs. Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Perfect Fit')
    plt.xlabel('Actual Life Expectancy')
    plt.ylabel('Predicted Life Expectancy')
    plt.title('Predicted vs Actual Life Expectancy')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    # Random Forest Regressor
    st.header("Random Forest Regressor")
    st.write("**Random Forest Regressor**: Aggregates predictions from multiple trees, reducing overfitting and handling complex interactions effectively.")
    rf = RandomForestRegressor()
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)

    st.write("**Random Forest Regressor Performance**:")
    st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_rf)}")
    st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_rf)}")
    st.write(f"R^2 Score: {r2_score(y_test, y_pred_rf)}")

    # Plotting Predicted vs. Real Values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='orange', alpha=0.5, label='Predicted vs. Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Perfect Fit')
    plt.xlabel('Actual Life Expectancy')
    plt.ylabel('Predicted Life Expectancy')
    plt.title('Predicted vs Actual Life Expectancy')
    plt.legend()
    plt.grid(True)

    # Render the plot in Streamlit
    st.pyplot(plt)
    plt.close()

# Time Series Tab (for Walmart Sales Forecast)
if selected == "Time Series":
    
    # Short explanation of the dataset
    st.header("Dataset Overview")
    st.write("""
    This application explores and analyzes datasets related to sales forecasting. The datasets include:
    - **Training Dataset**: Historical sales data used to train forecasting models.
    - **Test Dataset**: Data used for evaluating the performance of forecasting models.
    - **Features Dataset**: Additional features providing context to the sales data, such as store information and other attributes.

    The following tabs provide detailed insights into the data before and after merging with store and feature information.
    """)
    st.header("Exploratory Data Analysis (EDA)")

    st.subheader("Training Dataset Before Merging")
    st.write("Initial view of the training dataset before merging with stores and features:")
    st.write(df_train_ts_original.head())

    st.subheader("Test Dataset Before Merging")
    st.write("Initial view of the test dataset before merging with stores and features:")
    st.write(df_test_ts_original.head())

    st.subheader("Stores Dataset")
    st.write("Stores dataset containing store-specific information:")
    st.write(df_stores.head())
    
    st.subheader("Features Dataset")
    st.write("Features dataset with additional features for the training and test data:")
    st.write(df_features.head())

    st.header("Exploratory Data Analysis (EDA) on Merged Datasets")

    st.subheader("Merged Training Dataset Overview")
    st.write("Overview of the training dataset after merging with stores and features:")
    st.write(df_train_ts.head())

    st.subheader("Merged Training Dataset Summary")
    st.write("Summary statistics of the merged training dataset, including counts, means, standard deviations, and percentiles:")
    st.write(df_train_ts.describe())

    st.subheader("Missing Values in Merged Training Dataset")
    
    # Create columns for displaying tables side by side
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Before Filling Missing Values**")
        missing_values_before = df_train_ts_before.isnull().sum()
        missing_values_before_df = pd.DataFrame(missing_values_before, columns=['Missing Values'])
        st.write(missing_values_before_df)

    with col2:
        st.write("**After Filling Missing Values**")
        missing_values_after = df_train_ts_after.isnull().sum()
        missing_values_after_df = pd.DataFrame(missing_values_after, columns=['Missing Values'])
        st.write(missing_values_after_df)
    
    st.write("**Handling Missing Values**: Missing values in the MarkDown columns have been filled with 0. This ensures data continuity and usability for analysis.")

    st.subheader("Merged Test Dataset Overview")
    st.write("Overview of the test dataset after merging with stores and features:")
    st.write(df_test_ts.head())

    st.subheader("Merged Test Dataset Summary")
    st.write("Summary statistics of the merged test dataset, including counts, means, standard deviations, and percentiles:")
    st.write(df_test_ts.describe())

    st.subheader("Missing Values in Merged Test Dataset")

    # Create columns for displaying tables side by side
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Before Filling Missing Values**")
        missing_values_before_test = df_test_ts_before.isnull().sum()
        missing_values_before_test_df = pd.DataFrame(missing_values_before_test, columns=['Missing Values'])
        st.write(missing_values_before_test_df)

    with col2:
        st.write("**After Filling Missing Values**")
        missing_values_after_test = df_test_ts_after.isnull().sum()
        missing_values_after_test_df = pd.DataFrame(missing_values_after_test, columns=['Missing Values'])
        st.write(missing_values_after_test_df)
    
    st.write("**Handling Missing Values**: Missing values in the MarkDown columns have been filled with 0. Additionally, missing values in the CPI and Unemployment columns in the test dataset were filled using the mean values from the training dataset.")

    # Function to display 'Date' column from a dataset
    st.subheader("Viewing the First and Last 5 Rows of the DataFrames")

    # Create columns for displaying tables side by side
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Train Dates")
        display_date_column(df_train_ts, "Train Dataset")

    with col2:
        st.subheader("Test Dates")
        display_date_column(df_test_ts, "Test Dataset")

    with col3:
        st.subheader("Features Dates")
        display_date_column(df_features, "Features Dataset")

    # Graphical Analysis Section
    st.header("Graphical Analysis")

    # Sales Over Time Line Chart
    st.subheader("Visualization of Weekly Sales Over Time")
    st.write("**Sales Over Time Line Chart**: Displays the total weekly sales, highlighting trends, peaks, and patterns across the dataset's timeframe.")
    df_train_ts['Date'] = pd.to_datetime(df_train_ts['Date'])
    weekly_sales = df_train_ts.groupby('Date')['Weekly_Sales'].sum()
    st.line_chart(weekly_sales)

    # Histogram Section
    st.subheader("Histogram")
    st.write("**Histograms**: Display the distribution of various features to help understand their underlying patterns and variations.")

    # Create columns for histograms
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Histogram of Weekly Sales**: Shows the distribution of weekly sales amounts.")
        plt.figure(figsize=(10, 6))
        sns.histplot(df_train_ts['Weekly_Sales'], bins=30, kde=True, color='skyblue')
        st.pyplot(plt)
        
        st.write("**Histogram of Temperature**: Shows the distribution of temperature values.")
        plt.figure(figsize=(10, 6))
        sns.histplot(df_train_ts['Temperature'], bins=30, kde=True, color='lightgreen')
        st.pyplot(plt)
        
        st.write("**Histogram of Fuel Price**: Shows the distribution of fuel price values.")
        plt.figure(figsize=(10, 6))
        sns.histplot(df_train_ts['Fuel_Price'], bins=30, kde=True, color='salmon')
        st.pyplot(plt)

    with col2:
        st.write("**Histogram of CPI**: Shows the distribution of Consumer Price Index values.")
        plt.figure(figsize=(10, 6))
        sns.histplot(df_train_ts['CPI'], bins=30, kde=True, color='orange')
        st.pyplot(plt)

        st.write("**Histogram of Unemployment**: Shows the distribution of unemployment rates.")
        plt.figure(figsize=(10, 6))
        sns.histplot(df_train_ts['Unemployment'], bins=30, kde=True, color='purple')
        st.pyplot(plt)

        st.write("**Histogram of Store Size**: Shows the distribution of store size values.")
        plt.figure(figsize=(10, 6))
        sns.histplot(df_train_ts['Size'], bins=30, kde=True, color='teal')
        st.pyplot(plt)

    # Heatmap Section
    st.subheader("Heatmap")
    st.write("**Correlation Heatmap**: Shows the correlation coefficients between all numeric variables, helping identify strong relationships.")
    st.write("Correlation Heatmaps shows the correlation coefficients between all numeric values are acceptable, thus, there is no need to drop any columns")
    numeric_df = df_train_ts.select_dtypes(include=['number'])
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot(plt)

    # Pairplot Section
    st.subheader("Pairplot")
    st.write("**Pairplot**: Displays scatter plots between pairs of features and histograms for individual features, revealing relationships and distributions.")
    selected_features = ['Weekly_Sales', 'Store', 'Temperature', 'Fuel_Price']
    numeric_features = [feature for feature in selected_features if feature in df_train_ts.columns]
    plt.figure(figsize=(12, 8))
    sns.pairplot(df_train_ts[numeric_features])
    st.pyplot(plt)

    # Bar Plot Section
    st.subheader("Average Weekly Sales by Store")
    st.write("**Bar Plot of Average Weekly Sales by Store**: Compares average weekly sales across stores.")

    avg_sales_by_store = df_train_ts.groupby('Store')['Weekly_Sales'].mean().reset_index()
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Store', y='Weekly_Sales', data=avg_sales_by_store, palette='viridis')
    plt.xlabel('Store')
    plt.ylabel('Average Weekly Sales')
    plt.title('Average Weekly Sales by Store')
    st.pyplot(plt)

    df_train_ts.drop(columns=['IsHoliday_y'], inplace=True)
    df_test_ts.drop(columns=['IsHoliday_y'], inplace=True)

    # Apply one hot encoding for the IsHoliday columns to remove alphabetical values
    df_train_ts, df_test_ts = one_hot_encode_columns(df_train_ts, df_test_ts, ['IsHoliday_x'], 'Holiday')
    # Convert boolean values to integers (0 and 1)
    df_train_ts[['Holiday_False', 'Holiday_True']] = df_train_ts[['Holiday_False', 'Holiday_True']].astype(int)
    df_test_ts[['Holiday_False', 'Holiday_True']] = df_test_ts[['Holiday_False', 'Holiday_True']].astype(int)

    # Show the seasonal decomposition plot
    st.subheader("Seasonal Decomposition Plot")
    st.write("**Seasonal Decomposition**: This plot breaks down the 'Weekly Sales' time series into its trend, seasonal, and residual components, providing a clear view of underlying patterns and irregularities over time.")
    seasonal_decomposition_plot(df_train_ts['Weekly_Sales'], freq=7)

    # Ensure 'Date' is the index and in datetime format
    df_train_ts['Date'] = pd.to_datetime(df_train_ts['Date'])
    df_train_ts.set_index('Date', inplace=True)
    # Aggregate the data by week
    df_train_ts_weekly = df_train_ts['Weekly_Sales'].resample('W').sum()
    # Seasonal decomposition on the aggregated data
    st.write("**Seasonal Decomposition Plot on Weekly Aggregated Data**: used to help smooth out the data and reveal more pronounced trends and seasonal patterns.")
    # Perform seasonal decomposition on weekly aggregated data
    seasonal_decomposition_plot(df_train_ts_weekly, freq=52)

    #ACF and PACF Plots
    st.subheader("ACF and PACF Plots")
    st.write("**ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function)**: These plots help in identifying the extent of correlation between the current value and its previous values over different lags. Useful for detecting seasonality and determining the order of ARIMA models.")
    plot_acf_pacf(df_train_ts['Weekly_Sales'])

    # Perform AR Analysis
    best_aic_lag, best_bic_lag = auto_regressive_analysis(df_train_ts['Weekly_Sales'], max_lag=20)
    st.subheader("Auto-Regressive Analysis")
    st.write(
        "This analysis helps determine the most appropriate lag length for an Auto-Regressive (AR) model by evaluating "
        "two criteria: Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC). AIC chooses the model that balances fit and simplicity. BIC is Similar to AIC but with a stricter penalty for more parameters."
    )
    st.write(f"**Best lag based on AIC**: {best_aic_lag}")
    st.write(f"**Best lag based on BIC**: {best_bic_lag}")

    # Perform Moving Average Analysis
    ma_results = moving_average_analysis(df_train_ts['Weekly_Sales'], windows=[4, 12, 24])
    st.subheader("Moving Average Analysis")
    st.write(
        "The moving average analysis smooths the time series data by averaging over specified window sizes. This "
        "helps to identify and interpret longer-term trends and cycles in the data."
    )
    st.write("**Moving Averages Calculated:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**MA-4:**")
        st.write(ma_results['MA-4'].tail())
    with col2:
        st.write("**MA-12:**")
        st.write(ma_results['MA-12'].tail()) 
    with col3:
        st.write("**MA-24:**")
        st.write(ma_results['MA-24'].tail())

    # Prophet Forecast
    st.subheader("Prophet Forecast and Evaluation")
    st.write("""
        Prophet is well-suited for time series forecasting, especially when dealing with daily data that may have strong seasonal effects and holiday impacts. It handles missing data and outliers robustly and provides intuitive parameter tuning for modeling seasonality. Given the nature of our data, which involves daily sales with potential seasonal patterns and trend changes, Prophet's flexibility and accuracy make it an ideal choice for generating reliable forecasts.
        """)
    st.write("**Note**: The negative R² value is due to the test dataset containing only placeholder zeros, which lack variability and do not accurately reflect actual sales data, skewing the evaluation metrics. However, the very small RMSE and MSE values indicate good results and show that the model's predictions are very close to the actual values.")
    df_train_ts.reset_index(inplace=True)
    # Ensure 'Date' column is in datetime format and rename it to 'ds'
    df_train_ts['Date'] = pd.to_datetime(df_train_ts['Date'])
    df_test_ts['Date'] = pd.to_datetime(df_test_ts['Date'])
    # Prepare Prophet DataFrames
    train_prophet_df = df_train_ts.rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'})
    test_prophet_df = df_test_ts.rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'})

    # List of regressors
    regressors = ['CPI', 'Fuel_Price', 'Temperature']
    fit_predict_prophet(train_prophet_df, test_prophet_df, regressors=regressors)

    st.subheader("LSTM model Forecast and Evaluation")
    st.write("""
        The LSTM model's performance yielded very low RMSE and MSE values, indicating that the model achieved near-perfect predictions on the test dataset. However, the R² value remains 0.0, which suggests that the model's performance may not be meaningful. This discrepancy could be due to the test dataset containing placeholder zeros, which lack variability and do not accurately reflect actual sales data, potentially skewing the evaluation metrics. Note: The model takes around 10 minutes to fully run.
        """)
    # Prepare data for LSTM
    def prepare_lstm_data(df, look_back):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[['y']])
        
        X, y = [], []
        for i in range(len(scaled_data) - look_back):
            X.append(scaled_data[i:i + look_back])
            y.append(scaled_data[i + look_back])
        
        return np.array(X), np.array(y), scaler

    look_back = 10  # Number of previous time steps to use as input
    X_train, y_train, scaler = prepare_lstm_data(train_prophet_df, look_back)
    X_test, y_test, _ = prepare_lstm_data(test_prophet_df, look_back)

    # Define the LSTM model
    model_lstm = Sequential()
    model_lstm.add(LSTM(50, input_shape=(look_back, 1), return_sequences=True))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(LSTM(50))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(Dense(1))

    model_lstm.compile(optimizer='adam', loss='mean_squared_error')

    # Train the LSTM model
    model_lstm.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.3)

    # Forecast with LSTM
    X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_lstm = model_lstm.predict(X_test_reshaped)
    predicted_lstm = scaler.inverse_transform(predicted_lstm)

    # Evaluate LSTM model
    mse_lstm = mean_squared_error(y_test, predicted_lstm)
    rmse_lstm = np.sqrt(mse_lstm)
    r2_lstm = r2_score(y_test, predicted_lstm)

    st.write(f"RMSE: {rmse_lstm}")
    st.write(f"MSE: {mse_lstm}")
    st.write(f"R²: {r2_lstm}")

    # # Combine Prophet and LTSM
    # # Align lengths if necessary
    # min_length = min(len(forecast_test['yhat']), len(predicted_lstm))
    # forecast_test_aligned = forecast_test['yhat'][:min_length]
    # predicted_lstm_aligned = predicted_lstm.flatten()[:min_length]

    # # Combine predictions (example: averaging)
    # combined_predictions = (forecast_test_aligned + predicted_lstm_aligned) / 2

    # # Evaluate combined predictions
    # mse_combined = mean_squared_error(test_prophet_df['Weekly_Sales'].iloc[:min_length], combined_predictions)
    # rmse_combined = np.sqrt(mse_combined)
    # r2_combined = r2_score(test_prophet_df['Weekly_Sales'].iloc[:min_length], combined_predictions)

    # # Display combined metrics
    # st.subheader("Combined Model Forecast and Evaluation")
    # st.write(f"RMSE: {rmse_combined}")
    # st.write(f"MSE: {mse_combined}")
    # st.write(f"R²: {r2_combined}")

    # # Plotting results
    # plt.figure(figsize=(12, 6))
    # plt.plot(test_prophet_df['Date'].iloc[look_back:], test_prophet_df['Weekly_Sales'].iloc[look_back:], label='Actual Sales')
    # plt.plot(test_prophet_df['Date'].iloc[look_back:], forecast_test_aligned, label='Prophet Predictions')
    # plt.plot(test_prophet_df['Date'].iloc[look_back:], predicted_lstm_aligned, label='LSTM Predictions')
    # plt.plot(test_prophet_df['Date'].iloc[look_back:], combined_predictions, label='Combined Predictions', linestyle='--')
    # plt.legend()
    # plt.title('Combined Model Predictions vs Actual Sales')
    # plt.xlabel('Date')
    # plt.ylabel('Weekly Sales')
    # plt.show()

