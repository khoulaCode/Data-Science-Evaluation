import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from prophet import Prophet
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import altair as alt
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns
import plotly.express as px
from prophet.plot import plot_plotly,  plot_components_plotly


# Suppress global plot warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# Function to rename columns
def rename_columns(df):
    df.columns = [
        "Relative Compactness", "Surface Area", "Wall Area", "Roof Area", 
        "Overall Height", "Orientation", "Glazing Area", "Glazing Area Distribution", 
        "Heating Load", "Cooling Load"
    ]
    return df
def load_data(file_path):
    data = pd.read_csv(file_path, sep=';', parse_dates={'Datetime': ['Date', 'Time']}, infer_datetime_format=True, na_values=['?'])
    data.set_index('Datetime', inplace=True)
    return data

# Function to create a boxplot for a specific feature
def create_boxplot(feature, df):
    chart = alt.Chart(df).mark_boxplot(extent='min-max').encode(
        y=alt.Y(f'{feature}:Q', title=feature),
    )
    return chart


def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, rmse, r2

# Function to summarize numerical columns
def num_summary(dataframe, num_col, plot=False):
    quantiles = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    stats = dataframe[num_col].describe(quantiles).T
    stats = stats.reset_index()
    stats.columns = ['Metric', 'Value']
    stats = stats.set_index('Metric').T
    
    st.write(stats)

    if plot:
        chart = alt.Chart(dataframe).mark_bar().encode(
            alt.X(f'{num_col}:Q', bin=alt.Bin(maxbins=20), title=num_col),
            alt.Y('count():Q', title='Count')
        ).properties(
            title=f'Histogram of {num_col}'
        )
        st.altair_chart(chart, use_container_width=True)
## Time Series Functions:
def exploratory_data_analysis(daily_df):
    # Exploratory Data Analysis
    st.subheader("Exploratory Data Analysis")
    
    features = daily_df.columns

    # Displaying box plots in a grid layout
    st.subheader("Box Plots for All Features")

    # Calculate the number of rows needed
    num_features = len(features)
    num_cols = 2  # Number of columns per row
    num_rows = (num_features + num_cols - 1) // num_cols  # Ceiling division

    # Create a container for the plots
    for i in range(num_rows):
        cols = st.columns(num_cols)
        for j in range(num_cols):
            index = i * num_cols + j
            if index < num_features:
                feature = features[index]
                with cols[j]:
                    st.altair_chart(create_boxplot(feature, daily_df), use_container_width=True)
    # Heatmap of correlation matrix
    st.subheader("Heatmap of Feature Correlations")

    st.write("Correlation heatmap to understand relationships between numeric features.")
    numeric_df = daily_df.select_dtypes(include=['number'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", center=0)
    plt.title('Correlation Heatmap')
    st.pyplot()
    
    st.subheader("Explanation of Correlation Matrix")
    # Markdown box with updated content
    st.markdown("""
        <style>
            .markdown-text {
                background-color: #f5f5f5;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #d0d0d0;
            }
        </style>
        <div class="markdown-text">
        **Correlation Matrix Explanation:**

        - **Global_active_power and Global_intensity** have an almost perfect correlation (0.999), indicating redundancy. Consider dropping one of them.
        - **Global_active_power and power_consumption** have a strong positive correlation (0.941), making Global_active_power a key predictor of power consumption.
        - **Sub_metering_3** also shows a high correlation with **Global_active_power** (0.760) and **power_consumption** (0.875), indicating it should be considered in the model.
        - **Global_reactive_power** shows moderate correlation with **Voltage** (0.348) but low correlation with **power_consumption** (0.259), suggesting it might be less useful for prediction.

        These findings indicate which features might be most important for predicting power consumption and which might be redundant or less useful.
        </div>
        """, unsafe_allow_html=True)
    # Additional plots for features
    st.subheader("Exploratory Plots")
   
   
    st.subheader("Altair Pair Plot of Global Active Power and Global Intensity")
    alt_chart = alt.Chart(daily_df).mark_circle(size=60).encode(
        x='Global_active_power',
        y='Global_intensity',
        color='Global_active_power',
        tooltip=['Global_active_power', 'Global_intensity']
    ).interactive()
    st.altair_chart(alt_chart, use_container_width=True)

    st.subheader("Scatter Plot of Global Active Power vs. Voltage")
    fig, ax = plt.subplots()
    ax.scatter(daily_df['Global_active_power'], daily_df['Voltage'], color='blue')
    ax.set_xlabel('Global Active Power')
    ax.set_ylabel('Voltage')
    st.pyplot(fig)


    # 3D Scatter Plot: Global_reactive_power, Voltage, and power_consumption
    st.subheader('3D Scatter Plot: Global_reactive_power, Voltage, and power_consumption')

    scatter_3d = alt.Chart(daily_df).mark_circle(size=60).encode(
        x='Global_reactive_power',
        y='Voltage',
        color='power_consumption',
        tooltip=['Global_reactive_power', 'Voltage', 'power_consumption']
    ).interactive()

    st.altair_chart(scatter_3d, use_container_width=True)

    # Line chart for power consumption over time
    st.subheader('Line chart for power consumption over time')
    st.line_chart(daily_df['power_consumption'], use_container_width=True)
    

    #Pair plot using Altair
    st.subheader("Pair Plot")
    numeric_features = daily_df[['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'power_consumption']]
    pair_plot = alt.Chart(numeric_features.reset_index()).mark_circle(size=9).encode(
        x=alt.X(alt.repeat("column"), type='quantitative'),
        y=alt.Y(alt.repeat("row"), type='quantitative'),
        color=alt.Color('power_consumption', scale=alt.Scale(scheme='plasma')),
        tooltip=['Datetime']
    ).properties(
        width=80,
        height=80
    ).repeat(
        row=['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity'],
        column=['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']
    )
    st.altair_chart(pair_plot, use_container_width=True)

    # Markdown box indicating feature selection
    st.markdown("""
    <style>
        .markdown-text {
            background-color: #f0f0f0;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #d0d0d0;
        }
    </style>
    <div class="markdown-text">
    **Feature Selection:**
    After analyzing the heatmap and pair plot, it is observed that `Global_intensity` has an almost perfect correlation (0.999) with the target variable. Therefore, we will use `Global_intensity` for further analysis and modeling.
    </div>
    """, unsafe_allow_html=True)



# Function to process the data
def process_data(df, flag):
    # Ensure 'Datetime' is in datetime format
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d/%m/%Y %H:%M:%S')
    
    # Set 'Datetime' as index
    df.set_index('Datetime', inplace=True)
    
    # Show null values
    
    
    # Drop rows where all values are null
    df.dropna(how='all', inplace=True)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # Handle missing values for 'Sub_metering_3'
    # df['Sub_metering_3'].fillna(df['Sub_metering_3'].mean(), inplace=True)
    
    # Add a new column for power consumption
    df['power_consumption'] = ((df['Global_active_power']*1000/60) - (df['Sub_metering_1'] - df['Sub_metering_2'] - df['Sub_metering_3']))
    
    # Resample data daily
    daily_df = df.resample('H').sum()
    if flag:
        st.subheader("Null Values in the Dataset")
        st.write(df.isnull().sum())
        # Explanation and action inside a colored box
        st.markdown(
            """
            <div style=".markdown-text {
            background-color: #f0f0f0;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #d0d0d0;
        }">
                <p style="color: #333333;">
                    Since the null values are present in the same rows across all columns, we will drop all rows containing null values.
                </p>
            </div>
            """, unsafe_allow_html=True
        )
        # Display the processed data
        st.subheader("Processed Data Overview")
        st.write(daily_df.head())
        
        # Statistical summary
        st.subheader("Statistical Summary")
        st.write(daily_df.describe())
    
    return daily_df
    
def plot_acf_pacf(series):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(series, ax=ax[0])
    ax[0].set_title('ACF')
    plt.tight_layout()
    st.pyplot(fig)

def analyze_time_series(df):
    daily_df = df
    # daily_df.set_index('Datetime', inplace=True)

    # Resample to daily frequency, summing up the power consumption for each day
    daily_df = daily_df.resample('D').sum()
    # Decompose the power consumption into seasonal components
    st.subheader("Seasonal Decomposition of Power Consumption")
    decomposition = seasonal_decompose(daily_df['power_consumption'], model='additive')
    
    trend = decomposition.trend.dropna()
    seasonal = decomposition.seasonal.dropna()
    residual = decomposition.resid.dropna()
    
    
    # Plotting Trend component
    trend_fig = px.line(trend, title='Trend')
    st.plotly_chart(trend_fig)

    # Plotting Seasonal component
    seasonal_fig = px.line(seasonal, title='Seasonal')
    st.plotly_chart(seasonal_fig)

    # Plotting Residual component
    residual_fig = px.line(residual, title='Residual')
    st.plotly_chart(residual_fig)

    st.markdown(
        """
        <style>
            .markdown-text {
                background-color: #f0f0f0;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #d0d0d0;
            }
        </style>
        <div class="markdown-text"">
            <p style="color: #333333;">
                The Trend component shows that there is no clear upward or downward trend over time. The data fluctuates without a consistent direction, indicating that the overall trend is relatively stable.
            </p>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown(
        """
        <style>
            .markdown-text {
                background-color: #f0f0f0;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #d0d0d0;
            }
        </style>
        <div class="markdown-text">
            <p style="color: #333333;">
                The Seasonal component is very pronounced, showing clear and recurring patterns. This indicates that there are significant seasonal effects influencing the data, which repeat at regular intervals.
            </p>
        </div>
        """, unsafe_allow_html=True
    )

    
            
    # ACF and PACF plots
    st.subheader("ACF and PACF Plots")
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(df['power_consumption'].dropna(), ax=ax[0])
    plot_pacf(df['power_consumption'].dropna(), ax=ax[1])
    st.pyplot(fig)
    
    # AutoRegression Analysis
    st.subheader("AutoRegression Analysis")
    lags = st.slider("Select the number of lags for AutoReg", min_value=1, max_value=50, value=30)
    model = AutoReg(df['power_consumption'].dropna(), lags=lags).fit()
    st.write(f"AutoRegression Model Summary for lags={lags}:")
    st.write(model.summary())

    st.write("")
    # Model Summary Box for AR lags=30
    st.markdown(
        """
        <style>
        .markdown-text {
            background-color: #f0f0f0;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #d0d0d0;
        }
        </style>
    
        <div class="markdown-text">
            <p style="color: #333333;">
                **AutoRegression Model Summary for lags=30:**
                <br><br>
                The AutoReg Model results indicate a good fit with an AIC of 456305.237 and a BIC of 456568.503. The coefficients for various lags show different levels of significance, with some lag terms having very small p-values, indicating a strong relationship with the lagged values.
                <br><br>
                The residuals exhibit considerable variability and noise, suggesting that the AutoRegression model alone might not capture all the complexities of the data. The presence of significant seasonal patterns and noise in the residuals indicates that more advanced models might be needed.
                <br><br>
                Given these observations, we will transition to more advanced model like Prophet. These models are better suited for handling complex seasonality and big data, and should provide improved forecasting performance.
            </p>
        </div>
        """, unsafe_allow_html=True
    )
    
    # Moving Average Analysis using Rolling Mean
    st.subheader("Moving Average Analysis using Rolling Mean")
    rolling_window = st.slider("Select the rolling window size", min_value=2, max_value=30, value=3)
    df['rolling_mean'] = df['power_consumption'].rolling(rolling_window).mean()
    st.line_chart(df[['power_consumption', 'rolling_mean']], use_container_width=True)

    return df

def prophet_analysis(daily_df, test_df, forecast_length):
    st.write('')
    st.header('Prophet Model Forecasting')
    st.write('')

    st.markdown("""
    ### Prophet Model


    #### Steps:
    - **Data Preparation**: The data is prepared by renaming columns and resetting indices. [ds, y]
    - **Model Initialization**: The Prophet model is initialized with yearly, weekly, and daily seasonality.
    - **Fitting the Model**: The model is trained using the prepared data.
    - **Forecasting**: Future data points are created, and forecasts are generated.
    - **Evaluation**: Model performance is evaluated using metrics such as Mean Squared Error (MSE) and R-squared (R2).

    ### Model Evaluation
    """)

    # Prepare data for Prophet
    daily_df = daily_df.drop(columns=['Global_intensity'])
    daily_df = daily_df.rename(columns={'power_consumption': 'y'})
    daily_df.reset_index(inplace=True)
    daily_df.rename(columns={'Datetime': 'ds'}, inplace=True)
    
    # Initialize and fit the Prophet model
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
    # model.add_regressor('Global_active_power')
    model.add_regressor('Global_reactive_power')
    model.add_regressor('Voltage')
    model.add_regressor('Sub_metering_1')
    model.add_regressor('Sub_metering_2')
    model.add_regressor('Sub_metering_3')
    model.fit(daily_df)
    
    # Prepare test data for Prophet
    test_df = test_df.rename(columns={'power_consumption': 'y'})
    test_df.reset_index(inplace=True)
    test_df.rename(columns={'Datetime': 'ds'}, inplace=True)
    
    
    # Create future DataFrame for forecasting
    start_date = daily_df['ds'].max() #+ pd.Timedelta(hours=1)
    end_date = test_df['ds'].max() 
    future = pd.DataFrame({
        'ds': pd.date_range(start=start_date, end=end_date, freq='H')
    })

    # Prepare the future DataFrame for forecasting
    combined_df = pd.concat([daily_df[['ds', 'y' ]], test_df[['ds', 'y']]])
    combined_df.set_index('ds', inplace=True)
    # future = model.make_future_dataframe(periods=len(combined_df), freq='H')
    # future['Global_active_power'] = test_df['Global_active_power'].values
    future['Global_reactive_power'] = test_df['Global_reactive_power'].values
    future['Voltage'] = test_df['Voltage'].values
    future['Sub_metering_1'] = test_df['Sub_metering_1'].values
    future['Sub_metering_2'] = test_df['Sub_metering_2'].values
    future['Sub_metering_3'] = test_df['Sub_metering_3'].values
    
    # Forecast using the fitted model
    forecast = model.predict(future)
    # test_df.set_index('ds', inplace=True)
    # Align the predicted values
    
    actual = combined_df.loc[forecast['ds']]
    predicted = forecast.set_index('ds')['yhat'][:len(test_df)]
    # Create a plot for predicted vs actual values

    # Plot forecast
    st.write("**Forecast**")
    forecast_fig = plot_plotly(model, forecast)
    st.plotly_chart(forecast_fig)

    st.subheader("Comparison of Predicted and Actual Values")


    # Plot actual vs predicted values
    fig, ax = plt.subplots(figsize=(12, 6))
    # Plot the actual values
    ax.plot(actual.index, actual, label='Actual', color='blue', linewidth=2.5)
    # Plot the predicted values
    ax.plot(predicted.index, predicted, label='Predicted', color='#1f77b4', linewidth=2.5)
    # Add labels and title
    ax.set_xlabel('Datetime', fontsize=14, color='#333333')
    ax.set_ylabel('Power Consumption', fontsize=14, color='#333333')
    ax.set_title('Comparison of Actual vs Predicted Power Consumption', fontsize=16, color='#333333')
    # Customize tick parameters
    ax.tick_params(axis='both', labelsize=12, colors='#666666')
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    # Add legend
    ax.legend(frameon=False, fontsize=12)
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Display the plot
    st.pyplot(fig)

    # Plot components
    st.write("**Components**")
    components_fig = plot_components_plotly(model, forecast)
    st.plotly_chart(components_fig)

    
    mae = mean_absolute_error(test_df['y'], predicted)
    mse = mean_squared_error(test_df['y'], predicted)
    r2 = r2_score(test_df['y'], predicted)
    
    # Display forecasted data and metrics
    st.subheader("Forecast Data")
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(len(actual)))
    # Display model evaluation metrics in a Markdown box
    st.markdown(f"""
    ### Model Evaluation
    - **Mean Absolute Error (MAE)**: {mae}
    - **Mean Squared Error (MSE)**: {mse}
    - **R-squared (R2)**: {r2}
    """)
    st.markdown("""
        <style>
            .markdown-text {
                background-color: #f0f0f0;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #d0d0d0;
            }
        </style>
        <div class="markdown-text">
            **Prophet Model Results:**

            We used the Prophet model to forecast power consumption due to its strengths in handling time series data with multiple seasonalities and its robustness to missing values. Prophet is well-suited for this dataset as it can incorporate seasonal effects and trend changes, which are essential for accurate forecasting of energy consumption.

            ### What We Did:
            - **Data Preparation:** We transformed the data to include necessary regressors and formatted it for Prophet.
            - **Model Training:** The model was trained with daily data and evaluated using various metrics.
            - **Forecasting:** We generated forecasts and compared them to actual values.

            ### Why Prophet?
            Prophet was chosen because of its flexibility and ease of use with time series data that exhibits strong seasonal patterns and trend changes. It provides intuitive parameter tuning and is robust to missing data, making it an excellent choice for this energy consumption forecasting task.

            **Summary:** The Prophet model has demonstrated good performance with a high R-squared value, indicating that it effectively captures the underlying patterns in the data. The results suggest that Prophet is a suitable tool for forecasting power consumption in this context.
        </div>
    """)


def main():

    with st.sidebar:
        st.header("Options Menu")
        selected = option_menu(
            'Datasets', ["Linear Regression", "Time Series"], 
            icons=['play-btn', 'search'], menu_icon='intersect', default_index=0
        )

    if selected == "Linear Regression":
        st.title("Energy Consumption Prediction")

        st.sidebar.header("Upload Datasets")
        train_file = pd.read_excel('LR2/Train_ENB2012_data.xlsx')
        df_train = rename_columns(train_file)
        test_file = pd.read_excel('LR2/Test_ENB2012_data.xlsx')
        df_test = rename_columns(test_file)

        st.write("# Training Dataset Overview")
        st.dataframe(df_train.head())

        st.write("# Testing Dataset Overview")
        st.dataframe(df_test.head())

        # Data Cleaning and Preprocessing
        st.write("# Data Cleaning and Preprocessing")

        st.write("## Checking for Missing Values")
        missing_values = df_train.isnull().sum()
        missing_values = missing_values.reset_index()
        missing_values.columns = ['Metric', 'Value']
        missing_values = missing_values.set_index('Metric').T
        st.write(missing_values)
        
        st.markdown(
            """
            <div style=".markdown-text {
                background-color: #f0f0f0;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #d0d0d0;
            }">
                    <p style="color: #333333;">
                        No missing values found.
                    </p>
                </div>
            """, unsafe_allow_html=True
        )
        # Outlier Detection with Boxplots
        st.write("# Outlier Detection with Boxplots")

        num_features = df_train.drop(columns=["Heating Load", "Cooling Load"]).columns

        num_cols = len(num_features)
        num_columns = 2
        num_rows = (num_cols + num_columns - 1) // num_columns

        for i in range(num_rows):
            cols = st.columns(num_columns)
            for j in range(num_columns):
                index = i * num_columns + j
                if index < num_cols:
                    feature = num_features[index]
                    with cols[j]:
                        st.altair_chart(create_boxplot(feature, df_train), use_container_width=True)
                else:
                    # Empty placeholder for unused column in the last row
                    st.write("")
        
        st.markdown(
            """
            <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; border: 1px solid #d0d0d0;">
                
                **Boxplot Analysis:**

                No significant outliers were detected, indicating that the data is well-behaved and ready for modeling.
            
            </div>
            """, 
            unsafe_allow_html=True
        )

        # Data Visualization
        st.write("# Exploratory Data Analysis")

        st.write("## Numerical Summary and Histograms")
        num_cols = df_train.select_dtypes(include=['number']).columns
        selected_col = st.selectbox("Select a column to view summary", num_cols)

        if selected_col:
            st.write(f"### Summary for {selected_col}")
            num_summary(df_train, selected_col, plot=True)

        # Pairplot to understand relationships
        st.write("# Pairplot for Visualizing Relationships between Features")

        st.write("#### Pairplot")
        sns.pairplot(df_train, hue="Heating Load")
        st.pyplot()
        st.write(
            "The pairplot visualizes the relationships between features with respect to the 'Heating Load'. "
            "The color hue represents different heating load categories, aiding in understanding feature interactions."
        )

        st.write("### Heatmap of Feature Correlations")
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_train.corr(), annot=True, cmap="YlGnBu", center=0)
        plt.title('Correlation Heatmap')
        st.pyplot()
        st.markdown("""
                <style>
                    .markdown-text {
                        background-color: #f5f5f5;
                        padding: 15px;
                        border-radius: 8px;
                        border: 1px solid #d0d0d0;
                    }
                </style>
                <div class="markdown-text">
                ### Feature Correlation Findings
                    
                    1. **Heating Load**:
                        - **Relative Compactness**: Positive correlation (0.63), indicating that as relative compactness increases, the heating load tends to increase.
                        - **Surface Area**: Negative correlation (-0.66), suggesting that larger surface areas are associated with lower heating loads.
                        - **Roof Area**: Negative correlation (-0.87), which implies that larger roof areas are related to lower heating loads.
                        - **Overall Height**: Strong positive correlation (0.89), indicating that higher overall heights are associated with increased heating loads.
                        - **Orientation**: Very weak negative correlation (-0.03), suggesting minimal effect of orientation on heating load.
                        
                    2. **Cooling Load**:
                        - **Surface Area**: Negative correlation (-0.68), suggesting that larger surface areas are related to lower cooling loads.
                        - **Roof Area**: Negative correlation (-0.87), indicating that larger roof areas are associated with lower cooling loads.
                        - **Overall Height**: Strong positive correlation (0.90), meaning that greater overall heights are associated with higher cooling loads.
                        - **Orientation**: Very weak negative correlation (-0.01), suggesting minimal impact of orientation on cooling load.
                        - **Glazing Area**: Positive correlation (0.22), showing a weak association between glazing area and cooling load.
                                
                </div>
                """, unsafe_allow_html=True)
        
        # Distribution of Overall Height
        st.write("## Distribution of Overall Height")
        plt.figure(figsize=(10, 5))
        sns.histplot(df_train['Overall Height'], kde=True, color='skyblue', bins=30)
        plt.title('Distribution of Overall Height')
        st.pyplot()

        st.markdown(
            """
            <div style="background-color: #f0f0f0; padding: 15px; border-radius: 8px; border: 1px solid #d0d0d0;">
                
                The histogram shows the distribution of 'Overall Height' with density represented in blue.

                **Overall Height Feature Analysis:**
                The distribution of the `Overall Height` feature shows clustering around two distinct values: 3.5 and 7. 
                This separation indicates that `Overall Height` can be treated as a categorical variable. By applying label encoding, we can convert this feature into categorical labels, potentially enhancing the model's ability to identify patterns related to these specific height values.
                
            </div>
            """, 
            unsafe_allow_html=True
        )

        
    
        st.write("## Distribution of Glazing Area")
    
        plt.figure(figsize=(10, 5))
        sns.histplot(df_train['Glazing Area'], kde=True, color='teal', bins=30)
        plt.title('Distribution of Glazing Area')
        st.pyplot()
        st.write(
            "The histogram shows the distribution of 'Glazing Area'. "
            "The teal color represents the density of data points across the glazing area range."
        )

        st.write("## Distribution of Glazing Area Distribution")
        st.write(
            "### After Changes"
        )
        plt.figure(figsize=(10, 5))
        sns.countplot(x='Glazing Area Distribution', data=df_train, palette='magma')
        plt.title('Distribution of Glazing Area Distribution')
        st.pyplot()

        st.markdown(
            """
            <div style="background-color: #f0f0f0; padding: 15px; border-radius: 8px; border: 1px solid #d0d0d0;">
                
                    **Glazing Area Feature Analysis:**
                    The distribution of the `Glazing Area Distribution` feature reveals that the count of `0` is significantly less than half of all other values, 
                    while the counts for values `1`, `2`, `3`, `4`, and `5` are relatively similar. 
                    This pattern suggests that treating `Glazing Area` as a binary variable (0 vs. 1-5) could help improve prediction accuracy, 
                    as it highlights the distinction between no glazing and any level of glazing, which might capture more meaningful relationships in the model.
                
            </div>
            """, 
            unsafe_allow_html=True
        )


        

        # Split features and target variables
        
        df_train_orig = df_train
        df_test_orig = df_test
        X_train_original = df_train_orig.drop(columns=["Heating Load", "Cooling Load"])
        y_train_heating_original = df_train_orig["Heating Load"]
        y_train_cooling_original = df_train_orig["Cooling Load"]

        X_test_original = df_test_orig.drop(columns=["Heating Load", "Cooling Load"])
        y_test_heating_original = df_test_orig["Heating Load"]
        y_test_cooling_original = df_test_orig["Cooling Load"]

    

        # feature engineering 
        # Simplify 'Glazing Area Distribution'
        
        df_train['Glazing Area Distribution'] = df_train['Glazing Area Distribution'].apply(lambda x: 0 if x == 0 else 1)
        df_test['Glazing Area Distribution'] = df_test['Glazing Area Distribution'].apply(lambda x: 0 if x == 0 else 1)

        X_train = df_train.drop(columns=["Heating Load", "Cooling Load"])
        y_train_heating = df_train["Heating Load"]
        y_train_cooling = df_train["Cooling Load"]

        X_test = df_test.drop(columns=["Heating Load", "Cooling Load"])
        y_test_heating = df_test["Heating Load"]
        y_test_cooling = df_test["Cooling Load"]
        # Apply Label Encoding
        label_encoder = LabelEncoder()
        X_train['Overall Height Encoded'] = label_encoder.fit_transform(X_train['Overall Height'])
        X_test['Overall Height Encoded'] = label_encoder.transform(X_test['Overall Height'])
        X_train = X_train.drop(columns=['Overall Height'])
        X_test = X_test.drop(columns=['Overall Height'])

        # Scale Features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Fit models and evaluate
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge()
        }

        
        for model_name, model in models.items():
            # Create a row with two columns for Heating and Cooling results
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"## {model_name} Evaluation - Heating Load")
                
                # With Original Features
                st.write("### With Original Features")
                mae, mse, rmse, r2 = evaluate_model(model, X_train_original, y_train_heating_original, X_test_original, y_test_heating_original)
                st.write(f"Mean Absolute Error: {mae:.2f}")
                st.write(f"Mean Squared Error: {mse:.2f}")
                st.write(f"Root Mean Squared Error: {rmse:.2f}")
                st.write(f"R-squared: {r2:.2f}")

                # With Encoded Overall Height

                X_train_encoded = X_train.copy()
                X_test_encoded = X_test.copy()
                X_train_encoded['Overall Height'] = X_train['Overall Height Encoded']
                X_test_encoded['Overall Height'] = X_test['Overall Height Encoded']

                st.write("### With Encoded Overall Height")
                mae, mse, rmse, r2 = evaluate_model(model, X_train_encoded, y_train_heating, X_test_encoded, y_test_heating)
                st.write(f"Mean Absolute Error: {mae:.2f}")
                st.write(f"Mean Squared Error: {mse:.2f}")
                st.write(f"Root Mean Squared Error: {rmse:.2f}")
                st.write(f"R-squared: {r2:.2f}")

                # With Scaled Features
                st.write("### With Scaled Features")
                mae, mse, rmse, r2 = evaluate_model(model, X_train_scaled, y_train_heating, X_test_scaled, y_test_heating)
                st.write(f"Mean Absolute Error: {mae:.2f}")
                st.write(f"Mean Squared Error: {mse:.2f}")
                st.write(f"Root Mean Squared Error: {rmse:.2f}")
                st.write(f"R-squared: {r2:.2f}")

            with col2:
                st.write(f"## {model_name} Evaluation - Cooling Load")
                
                # With Original Features
                st.write("### With Original Features")
                mae, mse, rmse, r2 = evaluate_model(model, X_train_original, y_train_cooling_original, X_test_original, y_test_cooling_original)
                st.write(f"Mean Absolute Error: {mae:.2f}")
                st.write(f"Mean Squared Error: {mse:.2f}")
                st.write(f"Root Mean Squared Error: {rmse:.2f}")
                st.write(f"R-squared: {r2:.2f}")

                # With Encoded Overall Height
                st.write("### With Encoded Overall Height")
                mae, mse, rmse, r2 = evaluate_model(model, X_train_encoded, y_train_cooling, X_test_encoded, y_test_cooling)
                st.write(f"Mean Absolute Error: {mae:.2f}")
                st.write(f"Mean Squared Error: {mse:.2f}")
                st.write(f"Root Mean Squared Error: {rmse:.2f}")
                st.write(f"R-squared: {r2:.2f}")

                # With Scaled Features
                st.write("### With Scaled Features")
                mae, mse, rmse, r2 = evaluate_model(model, X_train_scaled, y_train_cooling, X_test_scaled, y_test_cooling)
                st.write(f"Mean Absolute Error: {mae:.2f}")
                st.write(f"Mean Squared Error: {mse:.2f}")
                st.write(f"Root Mean Squared Error: {rmse:.2f}")
                st.write(f"R-squared: {r2:.2f}")
        st.markdown("""
                <style>
                    .markdown-text {
                        background-color: #f5f5f5;
                        padding: 15px;
                        border-radius: 8px;
                        border: 1px solid #d0d0d0;
                    }
                </style>
                <div class="markdown-text">
                    **Results and Findings**

                    After evaluating both Linear and Ridge Regression models across different feature engineering steps, the following insights were gathered:

                    1. **With Original Features:**
                    - **Heating Load:** The original features provided a good baseline, with Linear Regression achieving an R-squared of 0.91 and Ridge Regression at 0.90. The errors were relatively low, with MAE values around 2.12 for Linear and 2.21 for Ridge.
                    - **Cooling Load:** Similar to the heating load, the original features performed well but slightly worse, with R-squared values of 0.88 (Linear) and 0.86 (Ridge).

                    2. **After Changing 'Glazing Area Distribution' and Applying Label Encoding:**
                    - **Heating Load:** Introducing the binary encoding of 'Glazing Area Distribution' and label encoding 'Overall Height' led to slight improvements in R-squared for both models (up to 0.92 for Linear and Ridge Regression). The MAE slightly improved as well, suggesting that these transformations made the models slightly better at capturing relationships within the data.
                    - **Cooling Load:** Interestingly, these changes did not improve the Cooling Load predictions. The R-squared slightly dropped for Linear Regression (from 0.88 to 0.87) and remained the same for Ridge Regression. This indicates that these feature transformations were more beneficial for predicting Heating Load than Cooling Load.

                    3. **With Scaled Features:**
                    - **Heating Load:** Scaling the features maintained the improvements seen with the encoded data. The R-squared remained consistent at 0.92 for both models, with further slight reductions in MAE and RMSE, confirming that scaling was beneficial after encoding.
                    - **Cooling Load:** Again, scaling didn't significantly impact Cooling Load predictions. The performance remained consistent with the encoded step, suggesting that for this specific task, scaling was not as impactful for Cooling Load.

                    **Conclusion:**
                    - **Final Approach:** Based on these results, the optimal approach would involve using feature engineering steps, including encoding 'Glazing Area Distribution' and 'Overall Height,' followed by scaling the features. This combination consistently improved model performance, especially for predicting Heating Load.
                    - **Why Ridge Regression:** The Ridge Regression model generally provided slightly better or comparable results across different steps, particularly after scaling. Therefore, Ridge Regression is preferred for its stability and slightly better handling of multicollinearity in the features.

                </div>
                """, unsafe_allow_html=True)

    elif selected == "Time Series":
        train_file = 'TS1/train_household_power_consumption.txt'
        test_file = 'TS1/test_household_power_consumption.txt'
        if train_file is not None:
            st.header("Train Dataset Analysis")
            train_df = pd.read_csv(train_file, sep=';', parse_dates={'Datetime': ['Date', 'Time']}, infer_datetime_format=True, na_values=['?'])
            train_df = process_data(train_df, True)
            exploratory_data_analysis(train_df)

        if test_file is not None:
            st.header("Test Dataset Analysis")
            test_df = pd.read_csv(test_file, sep=';', parse_dates={'Datetime': ['Date', 'Time']}, infer_datetime_format=True, na_values=['?'])
            test_df = process_data(test_df,  False)
            analyze_time_series(train_df)
            # Forecast and evaluate using the Prophet model
            forecast_length = len(test_df)  # Number of periods to forecast
            prophet_analysis(train_df, test_df, forecast_length)

if __name__ == '__main__':
    main()
