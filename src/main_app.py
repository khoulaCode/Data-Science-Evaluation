import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
import altair as alt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly


def load_data(data_path):
    df = pd.read_csv(data_path, sep=';', parse_dates={'Datetime': ['Date', 'Time']}, infer_datetime_format=True, na_values=['?'])
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d/%m/%Y %H:%M:%S')
    df.set_index('Datetime', inplace=True)
    return df

def preprocess_data(df):
    # Convert to numeric, coerce errors to handle missing or malformed data
    df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
    df['Global_reactive_power'] = pd.to_numeric(df['Global_reactive_power'], errors='coerce')
    df['Voltage'] = pd.to_numeric(df['Voltage'], errors='coerce')
    df['Global_intensity'] = pd.to_numeric(df['Global_intensity'], errors='coerce')
    df['Sub_metering_1'] = pd.to_numeric(df['Sub_metering_1'], errors='coerce')
    df['Sub_metering_2'] = pd.to_numeric(df['Sub_metering_2'], errors='coerce')
    df['Sub_metering_3'] = pd.to_numeric(df['Sub_metering_3'], errors='coerce')

    # Drop rows with NaN values resulting from coercion
    df = df.dropna()

    # Add the consumption column
    df['Consumption'] = (df['Global_active_power'] * 1000 / 60) - df['Sub_metering_1'] - df['Sub_metering_2'] - df['Sub_metering_3']

    # Resample the data to hourly frequency, taking the mean of each hour
    df_hourly = df.resample('H').mean()

    # Fill or drop any NaN values after resampling
    df_hourly.fillna(method='ffill', inplace=True)
    df_hourly.dropna(inplace=True)  # Drop any remaining NaNs

    # Add time-based features
    df_hourly['Year'] = df_hourly.index.year
    df_hourly['Month'] = df_hourly.index.month
    df_hourly['Day'] = df_hourly.index.day
    df_hourly['Hour'] = df_hourly.index.hour

    return df_hourly

def fit_prophet_model(df):
    df_prophet = df.reset_index()[['Datetime', 'Consumption']]
    df_prophet.rename(columns={'Datetime': 'ds', 'Consumption': 'y'}, inplace=True)

    model = Prophet()
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=24, freq='h')
    forecast = model.predict(future)

    return model, forecast

def plot_forecast(forecast):
    fig = alt.Chart(forecast).mark_line().encode(
        x='ds:T',
        y='yhat:Q',
        tooltip=['ds:T', 'yhat:Q']
    ).properties(
        width=800,
        height=400
    )

    return fig

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2

# Define the tabs
tab1, tab2 = st.tabs(["Linear Regression Analysis", "Time Series Analysis"])

with tab1:
    st.title('Predicting Energy Consumption')
    st.header("Linear Regression Analysis")
    st.write("""
    This application predicts energy consumption across different regions using a dataset from the UCI Machine Learning Repository.
    The dataset includes various features related to energy usage. The application cleans the data, explores it with EDA, and builds a regression model to forecast energy consumption.
    """)

    # Load datasets
    train_data_url = "..\LR2\Train_ENB2012_data.xlsx"  
    test_data_url = "..\LR2\Test_ENB2012_data.xlsx"

    train_df = pd.read_excel(train_data_url)
    test_df = pd.read_excel(test_data_url)
    # Display the raw datasets
    st.subheader("Training Dataset")
    st.write(train_df.head())

    st.subheader("Testing Dataset")
    st.write(test_df.head())

    # Define new column names
    new_column_names = {
        'X1': 'Relative Compactness',
        'X2': 'Surface Area',
        'X3': 'Wall Area',
        'X4': 'Roof Area',
        'X5': 'Overall Height',
        'X6': 'Orientation',
        'X7': 'Glazing Area',
        'X8': 'Glazing Area Distribution',
        'Y1': 'Heating Load (kWh/year)',  
        'Y2': 'Cooling Load (kWh/year)'   
    }

    # Rename columns in both training and testing datasets
    train_df.rename(columns=new_column_names, inplace=True)
    test_df.rename(columns=new_column_names, inplace=True)

    # Handling missing values in the training set
    st.subheader("Data Cleaning - Training Data")

    # Display missing values in training data
    st.write("Missing values in the training dataset:")
    st.write(train_df.isnull().sum())
    st.write("Note: There are no missing values.")

    # Clean training data
    train_df_cleaned = train_df.dropna()  # There are no null values so this doesn't do anything here.

    # Handling missing values in the testing set
    st.subheader("Data Cleaning - Testing Data")

    # Display missing values in testing data
    st.write("Missing values in the testing dataset:")
    st.write(test_df.isnull().sum())
    st.write("Note: There are no missing values.")

    # Clean testing data
    test_df_cleaned = test_df.dropna()  # There are no null values so this doesn't do anything here.

    st.write("The provided datasets are clean. They don't have any null values, inconsistencies or outliers. No further cleaning is needed.")
    # Exploratory Data Analysis (EDA)
    st.subheader("Exploratory Data Analysis (EDA)")

    # Scatter plots for key variables
    st.write("#### Scatter Plot of Heating Load vs. Cooling Load")
    fig, ax = plt.subplots()
    sns.scatterplot(data=train_df_cleaned, x='Cooling Load (kWh/year)', y='Heating Load (kWh/year)', ax=ax)
    ax.set_title("Heating Load vs Cooling Load")
    st.pyplot(fig)

    # Display Pair Plot
    st.write("#### Pair Plot")
    pairplot_fig = sns.pairplot(train_df_cleaned, vars=['Relative Compactness', 'Surface Area', 
                                                         'Wall Area', 'Roof Area', 'Overall Height', 
                                                         'Glazing Area', 'Heating Load (kWh/year)'])
    plt.title("Pair Plot of Features vs Heating Load")
    st.pyplot(pairplot_fig)

    # Correlation Heatmap
    st.write("#### Correlation Heatmap")
    corr_matrix = train_df_cleaned.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

    # Univariate Analysis - Box Plots
    def box_plots(
        *,
        data: pd.DataFrame,
        features: list,
        n_rows: int,
        n_cols: int,
        figsize: tuple = (15, 8),
    ) -> None:
        """This returns a box plot of all the specified features."""
        
        fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)

        for idx, feat in enumerate(features):
            if n_rows > 1:
                ax = axs[(idx // n_cols), (idx % n_cols)]
            else:
                ax = axs[idx]

            sns.boxplot(data=data, y=feat, ax=ax)  # Use y instead of x
            ax.set_title(f"Box Plot of {feat!r}")

        plt.tight_layout()
        st.pyplot(fig)


    # Define features for box plots
    features_to_plot = [
        'Relative Compactness',
        'Surface Area',
        'Wall Area',
        'Roof Area',
        'Overall Height',
        'Orientation',
        'Glazing Area',
        'Glazing Area Distribution',
        'Heating Load (kWh/year)',
        'Cooling Load (kWh/year)'
    ]

    # Call the box plot function
    st.write("#### Univariate Analysis - Box Plots")
    box_plots(data=train_df_cleaned, features=features_to_plot, n_rows=2, n_cols=5, figsize=(25, 15))

    # Note
    st.write("Note: There are no outliers detected in the dataset.")

    # Histogram Plot Function
    def hist_plots(
        *,
        data: pd.DataFrame,
        features: list,
        n_rows: int,
        n_cols: int,
        figsize: tuple = (15, 8),
    ) -> None:
        """This returns a histogram plot of all the specified features."""
        
        fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)

        for idx, feat in enumerate(features):
            if n_rows > 1:
                ax = axs[(idx // n_cols), (idx % n_cols)]
            else:
                ax = axs[idx]
            
            sns.histplot(data=data, x=feat, ax=ax, bins=20, kde=True)  # Add KDE for better distribution visualization
            ax.set_title(f"Hist Plot of {feat!r}")

        plt.tight_layout()
        st.pyplot(fig)

    # Call the histogram plot function
    st.write("#### Univariate Analysis - Histogram Plots")
    hist_plots(data=train_df_cleaned, features=features_to_plot, n_rows=2, n_cols=5, figsize=(25, 15))


    # Multivariate Analysis - Box Plots for Target Variables
    st.write("#### Multivariate Analysis - Heating and Cooling Load vs Overall Height")

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    sns.boxplot(data=train_df_cleaned, x="Overall Height", y="Heating Load (kWh/year)", ax=ax[0])
    ax[0].set_title("Heating Load Vs Overall Height")  # Set title for axis 0

    sns.boxplot(data=train_df_cleaned, x="Overall Height", y="Cooling Load (kWh/year)", ax=ax[1])
    ax[1].set_title("Cooling Load Vs Overall Height")  # Set title for axis 1

    plt.tight_layout()
    st.pyplot(fig)

    # Additional Multivariate Analysis for Glazing Area and Orientation
    st.write("#### Multivariate Analysis - Heating and Cooling Load vs Glazing Area Distribution")

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    sns.boxplot(data=train_df_cleaned, x="Glazing Area Distribution", y="Heating Load (kWh/year)", ax=ax[0])
    ax[0].set_title("Heating Load Vs Glazing Area Distribution")  # Set title for axis 0

    sns.boxplot(data=train_df_cleaned, x="Glazing Area Distribution", y="Cooling Load (kWh/year)", ax=ax[1])
    ax[1].set_title("Cooling Load Vs Glazing Area Distribution")  # Set title for axis 1

    plt.tight_layout()
    st.pyplot(fig)

    st.write("#### Multivariate Analysis - Heating and Cooling Load vs Orientation")

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    sns.boxplot(data=train_df_cleaned, x="Orientation", y="Heating Load (kWh/year)", ax=ax[0])
    ax[0].set_title("Heating Load Vs Orientation")  # Set title for axis 0

    sns.boxplot(data=train_df_cleaned, x="Orientation", y="Cooling Load (kWh/year)", ax=ax[1])
    ax[1].set_title("Cooling Load Vs Orientation")  # Set title for axis 1

    plt.tight_layout()
    st.pyplot(fig)

    # Feature Engineering
    st.subheader("Feature Engineering")

    # Create interaction term and log transformations
    train_df_cleaned['Relative_Compactness_Height'] = train_df_cleaned['Relative Compactness'] * train_df_cleaned['Overall Height']
    train_df_cleaned['Log_Heating_Load'] = np.log1p(train_df_cleaned['Heating Load (kWh/year)'])
    train_df_cleaned['Log_Cooling_Load'] = np.log1p(train_df_cleaned['Cooling Load (kWh/year)'])

    # Display the newly engineered features
    st.write("New Features: Relative_Compactness_Height, Log_Heating_Load, Log_Cooling_Load")
    st.write(train_df_cleaned.head())

    # Drop the 'Surface Area' column from both training and testing datasets
    train_df.drop("Surface Area", inplace=True, axis=1)
    test_df.drop("Surface Area", inplace=True, axis=1)

    # Set the target variable and features for Cooling Load
    X_train = train_df.drop(["Heating Load (kWh/year)", "Cooling Load (kWh/year)"], axis=1)
    y_train = train_df["Cooling Load (kWh/year)"]

    X_test = test_df.drop(["Heating Load (kWh/year)", "Cooling Load (kWh/year)"], axis=1)
    y_test = test_df["Cooling Load (kWh/year)"]

    # Scaling the numerical features
    var_to_scale = X_train.select_dtypes(include=["float64", "int64"]).columns  # Variables to apply MinMaxScaler
    scaler = MinMaxScaler()

    # Column transformer instantiation
    col_transformer = ColumnTransformer(
        transformers=[("scaler", scaler, var_to_scale)],
        remainder="passthrough",
    )

    # Apply transformation to training set and test set
    X_train_trans = col_transformer.fit_transform(X_train) 
    X_test_trans = col_transformer.transform(X_test)   

    # Convert transformed data back to DataFrame
    df_train_trans = pd.DataFrame(data=X_train_trans, columns=X_train.columns)
    df_test_trans = pd.DataFrame(data=X_test_trans, columns=X_test.columns)

    # Display transformed training data
    st.subheader("Transformed Training Data")
    st.write(df_train_trans.head())

    # Display transformed testing data
    st.subheader("Transformed Testing Data")
    st.write(df_test_trans.head())

    # Linear Regression Model for Cooling Load
    st.subheader("Linear Regression Model for Cooling Load")
    # Instantiate the model
    model_cooling = LinearRegression()

    # Fit the model to the training data
    model_cooling.fit(X_train_trans, y_train)

    # Make predictions on the test set
    y_pred_cooling = model_cooling.predict(X_test_trans)

    # Evaluation for Cooling Load
    st.write("#### Cooling Load Model Evaluation")
    mse_cooling = mean_squared_error(y_test, y_pred_cooling)
    mae_cooling = mean_absolute_error(y_test, y_pred_cooling)
    r2_cooling = r2_score(y_test, y_pred_cooling)

    # Display evaluation metrics for Cooling Load
    st.write(f"Mean Squared Error (MSE): {mse_cooling:.2f}")
    st.write(f"Mean Absolute Error (MAE): {mae_cooling:.2f}")
    st.write(f"R-squared (R2): {r2_cooling:.2f}")

    # Visualizing predictions vs actual values for Cooling Load
    st.write("#### Cooling Load Predictions vs Actual Values")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred_cooling)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # line for perfect predictions
    ax.set_title("Cooling Load Predictions vs Actual Values")
    ax.set_xlabel("Actual Cooling Load (kWh/year)")
    ax.set_ylabel("Predicted Cooling Load (kWh/year)")
    st.pyplot(fig)

    # Optional: Displaying residuals for Cooling Load
    st.write("#### Cooling Load Residuals Analysis")
    residuals_cooling = y_test - y_pred_cooling
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(residuals_cooling, kde=True, bins=30)
    ax.set_title("Cooling Load Residuals Distribution")
    ax.set_xlabel("Residuals")
    st.pyplot(fig)

    # Linear Regression Model for Heating Load
    st.subheader("Linear Regression Model for Heating Load")
    # Set the target variable for Heating Load
    y_heating_train = train_df_cleaned["Heating Load (kWh/year)"]
    y_heating_test = test_df["Heating Load (kWh/year)"]

    # Instantiate the model for Heating Load
    model_heating = LinearRegression()

    # Fit the model to the training data
    model_heating.fit(X_train_trans, y_heating_train)

    # Make predictions on the test set for Heating Load
    y_pred_heating = model_heating.predict(X_test_trans)

    # Evaluation for Heating Load
    st.write("#### Heating Load Model Evaluation")
    mse_heating = mean_squared_error(y_heating_test, y_pred_heating)
    mae_heating = mean_absolute_error(y_heating_test, y_pred_heating)
    r2_heating = r2_score(y_heating_test, y_pred_heating)

    # Display evaluation metrics for Heating Load
    st.write(f"Mean Squared Error (MSE): {mse_heating:.2f}")
    st.write(f"Mean Absolute Error (MAE): {mae_heating:.2f}")
    st.write(f"R-squared (R2): {r2_heating:.2f}")

    # Visualizing predictions vs actual values for Heating Load
    st.write("#### Heating Load Predictions vs Actual Values")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=y_heating_test, y=y_pred_heating)
    ax.plot([y_heating_test.min(), y_heating_test.max()], [y_heating_test.min(), y_heating_test.max()], color='red', linestyle='--')  # line for perfect predictions
    ax.set_title("Heating Load Predictions vs Actual Values")
    ax.set_xlabel("Actual Heating Load (kWh/year)")
    ax.set_ylabel("Predicted Heating Load (kWh/year)")
    st.pyplot(fig)

    # Optional: Displaying residuals for Heating Load
    st.write("#### Heating Load Residuals Analysis")
    residuals_heating = y_heating_test - y_pred_heating
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(residuals_heating, kde=True, bins=30)
    ax.set_title("Heating Load Residuals Distribution")
    ax.set_xlabel("Residuals")
    st.pyplot(fig)

# Tab 2: Time Series Analysis
with tab2:
    st.title('Household Power Consumption Forecasting')
    st.header("Time Series Analysis")
    st.write("""
    This application forecasts household power consumption using historical data.
    The dataset used is the Individual Household Electric Power Consumption dataset.
    """)

    train_data_url = r"..\TS1\train_household_power_consumption.txt"
    test_data_url = r"..\TS1\test_household_power_consumption.txt"

    train_data = load_data(train_data_url) 
    test_data = load_data(test_data_url) 

    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)

    st.write("### Training Data Sample")
    st.write(train_data.head())

    st.write("### Testing Data Sample")
    st.write(test_data.head())

    st.write("### Distribution of Global Active Power")
    chart = alt.Chart(train_data).mark_bar().encode(  
        alt.X("Global_active_power:Q", bin=True),
        y='count()',
    ).properties(
        width=600,
        height=400
    )
    st.altair_chart(chart)


    st.write("### Global Active Power over Time")
    chart = alt.Chart(train_data.reset_index()).mark_line().encode(  
        x='Datetime:T',
        y='Global_active_power:Q',
        tooltip=['Datetime:T', 'Global_active_power:Q']
    ).properties(
        width=800,
        height=400
    )
    st.altair_chart(chart)


    st.write("### Correlation Heatmap")
    corr_matrix = train_data.corr()
    corr_df = corr_matrix.stack().reset_index()
    corr_df.columns = ['Variable 1', 'Variable 2', 'Correlation']

    heatmap = alt.Chart(corr_df).mark_rect().encode(
        x='Variable 1:O',
        y='Variable 2:O',
        color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='viridis')),  # Use 'viridis', 'blues', 'reds', etc.
        tooltip=['Variable 1', 'Variable 2', 'Correlation']
    ).properties(
        width=600,
        height=600
    ).configure_axis(
        labelFontSize=10,
        titleFontSize=12
    )

    st.altair_chart(heatmap)

    # Prophet
    # Initialize the model with 95% uncertainty interval
    my_model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.1  # Increase complexity
    )

    # Prepare the data for Prophet
    train_data_prepared = train_data.reset_index()[['Datetime', 'Consumption']]
    train_data_prepared.rename(columns={'Datetime': 'ds', 'Consumption': 'y'}, inplace=True)

    # Fit the model
    my_model.fit(train_data_prepared)

    # Create future dates DataFrame for the next 365 days
    future_dates = my_model.make_future_dataframe(periods=60, freq='D')

    # Make predictions
    forecast = my_model.predict(future_dates)

    # Display the forecast
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())

    # Plot the forecast
    st.write("#### Forecast Plot")
    fig1 = my_model.plot(forecast)
    st.pyplot(fig1)

    # Plot the forecast components
    st.write("#### Forecast Components")
    fig2 = my_model.plot_components(forecast)
    st.pyplot(fig2)

    # Align the forecast with the test data index
    forecast.set_index('ds', inplace=True)
    aligned_forecast = forecast.reindex(test_data.index).dropna()

    # Evaluate metrics only on the timestamps present in both test data and forecast
    test_data = test_data.loc[aligned_forecast.index]
    test_data['yhat'] = aligned_forecast['yhat']

    mae, mse, rmse, r2 = calculate_metrics(test_data['Consumption'], test_data['yhat'])

    st.write("### Model Evaluation on Test Data")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.3f}")
    st.write(f"**Mean Squared Error (MSE):** {mse:.3f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.3f}")
    st.write(f"**R-squared (R2):** {r2:.3f}")
