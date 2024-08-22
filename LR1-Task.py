import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import streamlit as st
from fuzzywuzzy import process
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from prophet import Prophet
from math import sqrt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima


class Life_Expectancy:
    def __init__(self, df):
        self.df = df

    def display_head(self):
        """
         I use this function  to display the firsat few rows of the df.
        """
        st.subheader("DataFrame Head")
        st.write(self.df.head())

    def display_description(self):
        """
        To show and explore important stats about the data.
        """
        st.subheader("Descriptive Statistics")
        st.write(self.df.describe())

    def display_missing_values(self):
        """
        Check for Null Values
        """
        st.subheader("Missing Values Summary")
        # Calculate the number of missing values for each column
        missing_values = self.df.isnull().sum().sort_values(ascending=False)
        st.write(missing_values)

    def display_heatmap(self):
        """
        look for correlations between columns
        """
        st.subheader("Correlation Heatmap")
        numeric_df = self.df.select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
        if not numeric_df.empty:
            plt.figure(figsize=(15, 10))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
            st.pyplot(plt.gcf())
        else:
            st.write("No numeric columns available for correlation heatmap.")
        st.markdown(f"**Explanation:** The correlation heatmap displays the relationships between numeric features, helping identify which variables might be strongly related.")

    def display_histograms(self):
        """
        used to further understand the data
        """
        st.subheader("Histograms")
        numeric_columns = self.df.select_dtypes(include=['float64', 'int64']).columns
        if not numeric_columns.empty:
            plt.figure(figsize=(20, 15))
            for i, column in enumerate(numeric_columns):
                plt.subplot(len(numeric_columns) // 4 + 1, 4, i + 1)  # Adjust layout for 3 columns per row
                sns.histplot(self.df[column], bins=20, kde=True, color='skyblue')
                plt.title(f'Distribution of {column}')
                plt.xlabel(column)
                plt.ylabel('Frequency')
            plt.tight_layout()
            st.pyplot(plt.gcf())
        else:
            st.write("No numeric columns available for histograms.")
        st.markdown(f"**Explanation:** Histograms show the distribution of individual numeric features, providing insights into their range, central tendency, and spread.")

    def display_pairplots(self):
        """
        used to find relationships between data
        """
        st.subheader("Pair Plots of Selected Features")
        selected_features = [
            'life expectancy', 
            'adult mortality',
            'hiv/aids',
            'income composition of resources',
            'schooling',
        ]

        if all(feature in self.df.columns for feature in selected_features):
            sns.pairplot(self.df[selected_features], kind='scatter', plot_kws={'color': 'green'})
            st.pyplot(plt.gcf())
        else:
            st.write("Not all selected features are present in the dataset.")
        st.markdown(f"**Explanation:** Pair plots visualize relationships between pairs of features, providing a quick way to identify correlations and patterns across multiple variables.")

    def display_violin_plot(self):
        """
        box plot to find correlation between country status and life expectancy
        """
        st.subheader("Life Expectancy Based on Countries' Status")
        
        # Strip whitespace from column names to ensure correct references
        self.df.columns = self.df.columns.str.strip().str.lower()
        
        # Create the violin plot using the correct lowercase column name
        fig = px.violin(
            self.df,
            x='status',  # Use the lowercase 'status' here
            y='life expectancy',
            color='status',
            box=True,
            template='plotly_dark',
            title="Life Expectancy Based on Countries' Status"
        )
        st.plotly_chart(fig)
        st.markdown(f"**Explanation:** The violin plot shows the distribution of life expectancy in different types of countries (e.g., developed vs. developing), with a boxplot overlay to highlight summary statistics.")

    def display_boxplots(self):
        st.subheader("Boxplots of Selected Features")
        
        # Select relevant features (use lowercase and stripped versions)
        selected_features = [
            'life expectancy',  # Use the lowercase and stripped version
            'adult mortality',
            'hiv/aids',
            'income composition of resources',
            'schooling'
        ]
        
        # Generate compact boxplots for selected features
        cols = st.columns(2)  # Create two columns to make the boxplots more compact
        for i, feature in enumerate(selected_features):
            with cols[i % 2]:  # Alternate between the two columns
                plt.figure(figsize=(5, 4))
                sns.boxplot(x=self.df[feature], color='orange')
                plt.title(f'Boxplot of {feature}')
                plt.xlabel(feature)
                st.pyplot(plt.gcf())
        st.markdown(f"**Explanation:** The boxplots show the distribution of this variable, highlighting the median, quartiles, and potential outliers.")
        
    def fill_missing_values(self):
        """
        Fill missing values based on the mean of one row above and one row below,
        grouped by country.
        """

        def fill_value(group):
            """
            Apply the mean calculation for filling null values grouped by country.
            """
            for column in group.columns:
                if group[column].isnull().any():
                    for i in range(1, len(group) - 1): 
                        if pd.isnull(group.iloc[i][column]):
                            prev_value = group.iloc[i - 1][column]
                            next_value = group.iloc[i + 1][column]
                            if not pd.isnull(prev_value) and not pd.isnull(next_value):
                                group.at[group.index[i], column] = np.mean([prev_value, next_value])
                            elif not pd.isnull(prev_value):
                                group.at[group.index[i], column] = prev_value
                            elif not pd.isnull(next_value):
                                group.at[group.index[i], column] = next_value
            return group
        self.df = self.df.groupby('country').apply(fill_value).reset_index(drop=True)
        
    def fill_population_from_updated_data(self, update_df, country_name_mapping):
        """
        Fill the missing values in the 'Population' column.
        """
        # this is step is to make sure the population is the same format
        update_df['population_mln'] = update_df['population_mln'] * 1_000_000

        self.df['country'] = self.df['country'].str.strip().str.lower()
        update_df['country'] = update_df['country'].str.strip().str.lower()
        
        country_name_mapping = {k.lower(): v.lower() for k, v in country_name_mapping.items()}
        update_df['country'] = update_df['country'].replace(country_name_mapping)
        
        unmatched_countries = set(self.df['country']).difference(set(update_df['country']))
        for country in unmatched_countries:
            match, score = process.extractOne(country, update_df['country'].unique())
            if score >= 90:
                update_df.loc[update_df['country'] == match, 'country'] = country

        # merge OG df with updated df
        merged_df = pd.merge(self.df, update_df[['country', 'year', 'population_mln']], on=['country', 'year'], how='left')

        # Fill missing values 
        self.df['population'].fillna(merged_df['population_mln'], inplace=True)


    def fill_gdp_from_per_capita(self, update_df, population_column='population'):
        """
        Fill the missing values in the gdp.
        """
        # strip and lower
        self.df['country'] = self.df['country'].str.strip().str.lower()
        update_df['country'] = update_df['country'].str.strip().str.lower()

        # Merge the DataFrames
        merged_df = pd.merge(self.df[['country', 'year', population_column]], 
                            update_df[['country', 'year', 'gdp_per_capita']], 
                            on=['country', 'year'], how='left')

        merged_df['gdp'] = merged_df['gdp_per_capita']

        # Fill missing values 
        self.df['gdp'].fillna(merged_df['gdp'], inplace=True)


    def fill_schooling_from_update(self, update_df):
        """
        Fill the missing values in schooling
        """

        self.df['country'] = self.df['country'].str.strip().str.lower()
        update_df['country'] = update_df['country'].str.strip().str.lower()

        # Merge the original df 
        merged_df = pd.merge(self.df[['country', 'year', 'schooling']], 
                            update_df[['country', 'year', 'schooling']], 
                            on=['country', 'year'], how='left', suffixes=('_orig', '_updated'))

        # Fill missing values 
        self.df['schooling'].fillna(merged_df['schooling_updated'], inplace=True)


    def fill_hepatitis_b_from_update(self, update_df):
        """
        Fill the missing values in Hepatitis b
        """

        self.df['country'] = self.df['country'].str.strip().str.lower()
        update_df['country'] = update_df['country'].str.strip().str.lower()

        # Merge the original df 
        merged_df = pd.merge(self.df[['country', 'year', 'hepatitis b']], 
                            update_df[['country', 'year', 'hepatitis_b']], 
                            on=['country', 'year'], how='left', suffixes=('_orig', '_updated'))

        # Fill missing values
        self.df['hepatitis b'].fillna(merged_df['hepatitis_b'], inplace=True)

    
    def drop_column(self, column_name):
        if column_name in self.df.columns:
            self.df.drop(columns=[column_name], inplace=True)
            st.markdown(f"'{column_name}' column has been dropped from the dataset due to high correlation with under 5 deaths column.")
        else:
            st.markdown(f"'{column_name}' column not found in the dataset.")

    def fill_missing_with_mean(self):
        """
        Fill the leftover missing values the mean of each column.
        """
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_columns] = self.df[numeric_columns].apply(lambda x: x.fillna(x.mean()), axis=0)


    def train_regression_model(self, features, target):
        st.subheader("Training Linear Regression Model")
        
        X_train = self.df[features].dropna()
        y_train = self.df.loc[X_train.index, target]
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        st.markdown("Linear Regression model has been trained on the train dataset.")
        return model

    def train_random_forest_model(self, features, target, n_estimators=100, random_state=42):
        st.subheader("Training Random Forest Model")
        
        X_train = self.df[features].dropna()
        y_train = self.df.loc[X_train.index, target]
        
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)
        
        st.markdown("Random Forest model has been trained on the train dataset.")
        return model

    def train_ridge_model(self, features, target, alpha=1.0):
        st.subheader("Training Ridge Regression Model")
        
        X_train = self.df[features].dropna()
        y_train = self.df.loc[X_train.index, target]
        
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        
        st.markdown("Ridge Regression model has been trained on the train dataset.")
        return model

    def apply_regression_model(self, model, features, target, model_name="Model"):
        st.subheader(f"Applying {model_name} on Test Dataset")
        
        X_test = self.df[features].dropna()
        y_test = self.df.loc[X_test.index, target]
        
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        st.write(f"R² Score: {r2}")
        st.write(f"Mean Squared Error (MSE): {mse}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse}")
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, color='blue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Actual vs Predicted - {model_name}')
        st.pyplot(plt.gcf())

class TimeSeries:
    def __init__(self, train_df, stores_df, features_df):
        self.train_df = train_df
        self.stores_df = stores_df
        self.features_df = features_df
        self.merged_df = None

    def merge_data(self):
        """
        Merges the train DataFrame with the stores DataFrame and then with the features DataFrame
        based on the 'Store' and 'Date' columns.
        """
        # Ensure consistency in key columns by stripping whitespace and converting to title case
        self.train_df.columns = self.train_df.columns.str.strip().str.title()
        self.stores_df.columns = self.stores_df.columns.str.strip().str.title()
        self.features_df.columns = self.features_df.columns.str.strip().str.title()

        # Convert date columns to datetime format for proper merging
        self.train_df['Date'] = pd.to_datetime(self.train_df['Date'], errors='coerce')
        self.features_df['Date'] = pd.to_datetime(self.features_df['Date'], errors='coerce')

        # Merge train data with store information on Store column
        merged_store = pd.merge(self.train_df, self.stores_df, on='Store', how='left')

        # Merge the result with features on Store and Date
        self.merged_df = pd.merge(merged_store, self.features_df, on=['Store', 'Date'], how='left')

        # Display the merged DataFrame in Streamlit
        st.subheader("Merged DataFrame Head")
        st.dataframe(self.merged_df)

    def check_null_values(self):
        """
        count of null values in the merged DataFrame.
        """
        null_counts = self.merged_df.isnull().sum()
        st.subheader("Columns with Null Values")
        st.write(null_counts[null_counts > 0])


    def correlation_heatmap(self):
        """
        correlation heatmap of the merged DataFrame.
        """
        if self.merged_df is not None:
            st.subheader("Correlation Heatmap")
            
            # Select only numeric columns for correlation
            numeric_df = self.merged_df.select_dtypes(include=['float64', 'int64'])
            
            if not numeric_df.empty:
                plt.figure(figsize=(12, 8))
                sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
                plt.title("Correlation Heatmap")
                st.pyplot(plt.gcf())
            else:
                st.write("No numeric columns available to generate a correlation heatmap.")
        else:
            st.write("DataFrames are not merged yet. Please merge the DataFrames first.")

    def display_description(self):
        """
        important stats about the data.
        """
        st.subheader("Descriptive Statistics")
        st.write(self.merged_df.describe())

    def display_histograms(self):
        """
        Generates and displays histograms.
        """
        if self.merged_df is not None:
            st.subheader("Histograms of Numeric Features")
            
            # Select only numeric columns
            numeric_columns = self.merged_df.select_dtypes(include=['float64', 'int64']).columns
            
            if not numeric_columns.empty:
                plt.figure(figsize=(20, 15))
                for i, column in enumerate(numeric_columns):
                    plt.subplot(len(numeric_columns) // 4 + 1, 4, i + 1)  
                    sns.histplot(self.merged_df[column], bins=20, kde=True, color='skyblue')
                    plt.title(f'Distribution of {column}')
                    plt.xlabel(column)
                    plt.ylabel('Frequency')
                plt.tight_layout()
                st.pyplot(plt.gcf())
            else:
                st.write("No numeric columns available to generate histograms.")
        else:
            st.write("DataFrames are not merged yet. Please merge the DataFrames first.")

    def display_pairplots(self):
        """
        Generates and displays pair plots for the most relevant columns in the merged DataFrame.
        """
        relevant_columns = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'Cpi', 'Unemployment']
        st.subheader("Pair Plots of Selected Features")  
        sns.pairplot(self.merged_df[relevant_columns], kind='scatter', plot_kws={'alpha': 0.7, 'color': 'green'})
        st.pyplot(plt.gcf())

    def display_boxplots(self):
        """
        Generates and displays box plots
        """
        st.subheader("Box Plots of Selected Features")
        relevant_columns = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'Cpi', 'Unemployment', ]
            
        plt.figure(figsize=(15, 10))
        for i, column in enumerate(relevant_columns):
            plt.subplot(len(relevant_columns) // 3 + 1, 3, i + 1)  
            sns.boxplot(y=self.merged_df[column], color='orange')
            plt.title(f'Box Plot of {column}')
            plt.ylabel(column)
        plt.tight_layout()
        st.pyplot(plt.gcf())

    def display_pie_chart(self):
        """
        Generates and displays a pie chart
        """
        category_column = ('Type')
        st.subheader(f"Pie Chart of Store type Distribution")

        # Calculate the distribution of the category
        category_counts = self.merged_df[category_column].value_counts()

        # Plot the pie chart
        plt.figure(figsize=(4, 4))
        plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', colors=sns.color_palette("Set2"))
        plt.title(f"Distribution of {category_column}")
        st.pyplot(plt.gcf())

    def fill_nulls_with_zero(self):
        """
        Replaces null values with 0 in markedown columns
        """
        numeric_columns = ['Markdown1', 'Markdown2', 'Markdown3', 'Markdown4', 'Markdown5']
        self.merged_df[numeric_columns] = self.merged_df[numeric_columns].fillna(0)
        st.write("Null values in markdown columns have been replaced with 0.")

    def one_hot_encode_holiday(self):
        """
        One-hot encodes the 'IsHoliday_x' and 'IsHoliday_y'
        """
        holiday_columns = ['Isholiday_y', 'Isholiday_x']
            
        for col in holiday_columns:
            if col in self.merged_df.columns:
                # One-hot encode the column
                encoded_df = pd.get_dummies(self.merged_df[col], prefix=col)
                # Drop the original column a
                self.merged_df = pd.concat([self.merged_df.drop(columns=[col]), encoded_df], axis=1)
            
        st.write("One-hot encoding applied to 'IsHoliday_x' and 'IsHoliday_y'.")
        st.write(self.merged_df.head())  # Display the updated DataFrame

    def fill_nulls_cpi_unemployment(self):
        """
        Fills null values in 'CPI' and 'Unemployment'
        """
        for column in ['Cpi', 'Unemployment']:
            if column in self.merged_df.columns:
                self.merged_df[column] = self.merged_df.groupby('Store')[column].transform(
                    lambda x: x.fillna(x.mean())
                )
        st.write("Null values in 'CPI' and 'Unemployment' columns have been filled with the mean value for each corresponding store.")
        st.dataframe(self.merged_df)

    def aggregated_time_series_decomposition(self, model='multiplicative'):
        """
        Aggregates sales data 
        """
        aggregated_sales = self.merged_df.groupby('Date')['Weekly_Sales'].sum()
        aggregated_sales = aggregated_sales.sort_index()
        decomposition = seasonal_decompose(aggregated_sales, model=model, period=52)

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8))
        decomposition.observed.plot(ax=ax1)
        ax1.set_ylabel('Observed')
        decomposition.trend.plot(ax=ax2)
        ax2.set_ylabel('Trend')
        decomposition.seasonal.plot(ax=ax3)
        ax3.set_ylabel('Seasonal')
        decomposition.resid.plot(ax=ax4)
        ax4.set_ylabel('Residual')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown(f"**Explanation:** The decomposition shows the aggregated sales data, trend, seasonality, and residual components. This helps in understanding the overall patterns in the time series data across all stores.")

    def plot_acf(self, lags=52):
        """
        Plots the Autocorrelation Function
        """

        aggregated_sales = self.merged_df.groupby('Date')['Weekly_Sales'].sum()
        aggregated_sales = aggregated_sales.sort_index()

        fig, ax = plt.subplots(figsize=(12, 6))
        plot_acf(aggregated_sales, lags=lags, ax=ax)

        ax.set_title('Autocorrelation Function (ACF) of Aggregated Weekly Sales')
        ax.set_xlabel('Lags')
        ax.set_ylabel('Autocorrelation')

        plt.tight_layout()
        st.subheader("Autocorrelation Function (ACF) Plot")
        st.pyplot(fig)

        st.markdown(
            "**Interpretation:** The ACF plot shows the correlation of the time series with its own past values. "
            "Significant spikes at specific lags indicate strong correlation at those lags. For instance, a spike at lag 52 "
            "suggests annual seasonality in the data, meaning sales patterns repeat every year."
        )

    def plot_pacf(self, lags=52):
        """
        Plots the Partial Autocorrelation Function
        """

        aggregated_sales = self.merged_df.groupby('Date')['Weekly_Sales'].sum()
        aggregated_sales = aggregated_sales.sort_index()

        fig, ax = plt.subplots(figsize=(12, 6))
        plot_pacf(aggregated_sales, lags=lags, method='ywm', ax=ax)

        ax.set_title('Partial Autocorrelation Function (PACF) of Aggregated Weekly Sales')
        ax.set_xlabel('Lags')
        ax.set_ylabel('Partial Autocorrelation')

        plt.tight_layout()
        st.subheader("Partial Autocorrelation Function (PACF) Plot")
        st.pyplot(fig)

        st.markdown(
            "**Interpretation:** The PACF plot shows the correlation of the time series with its own past values "
            "after removing the effects of intermediate lags. This is useful for identifying the appropriate lag order "
            "in autoregressive models. Significant spikes indicate lags that contribute explanatory power to the model."
        )

    def moving_average_analysis(self, window=4):
        """
        Calculates and plots the moving average of weekly sales.
    
        """
        aggregated_sales = self.merged_df.groupby('Date')['Weekly_Sales'].sum()
        aggregated_sales = aggregated_sales.sort_index()

        # Calculate moving average
        moving_average = aggregated_sales.rolling(window=window).mean()

        # Plot actual vs moving average
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(aggregated_sales, label='Actual Sales', color='blue')
        ax.plot(moving_average, label=f'Moving Average (Window = {window})', color='orange')
        ax.set_title(f'Moving Average of Weekly Sales (Window = {window} weeks)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Weekly Sales')
        ax.legend()

        plt.tight_layout()
        st.subheader("Moving Average Analysis")
        st.pyplot(fig)

        st.markdown(
            f"**Interpretation:** The moving average smooths the time series data by averaging over a window of 4 weeks. "
            "This helps in identifying underlying trends by reducing short-term fluctuations or noise in the sales data." )

    def preprocess_for_prophet(self, is_train=True):
        """
        Prepares the data for Prophet modeling.
        If `is_train` is True, it includes the target variable ('y'), otherwise only includes the dates ('ds').
        """
        if is_train:
            df = self.merged_df.groupby('Date')['Weekly_Sales'].sum().reset_index()
            df.rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'}, inplace=True)
        else:
            df = self.merged_df[['Date']].drop_duplicates().reset_index(drop=True)
            df.rename(columns={'Date': 'ds'}, inplace=True)
        return df


    def fit_predict_prophet(self, train_df, test_df):
        """
        Fits a Prophet model on the training data and makes predictions on the test data.
        Calculates RMSE, MSE, and R² for the training data predictions.
        """
        prophet_model = Prophet()
        prophet_model.fit(train_df)

        # Forecasting within the training period for evaluation
        forecast_train = prophet_model.predict(train_df[['ds']])

        # Calculating metrics on the training data
        y_true_train = train_df['y'].values
        y_pred_train = forecast_train['yhat'].values

        mse_train = mean_squared_error(y_true_train, y_pred_train)
        rmse_train = sqrt(mse_train)
        r2_train = r2_score(y_true_train, y_pred_train)

        st.subheader("Prophet Model Training Evaluation Metrics")
        st.write(f"RMSE: {rmse_train}")
        st.write(f"MSE: {mse_train}")
        st.write(f"R²: {r2_train}")

        # Forecasting on the test data
        future = test_df[['ds']].copy()
        forecast_test = prophet_model.predict(future)

        # Visualize the forecast on the test data
        st.subheader("Prophet Model Forecast on Test Data")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_df['ds'], train_df['y'], label='Training Data', color='blue')
        ax.plot(forecast_train['ds'], forecast_train['yhat'], label='Fitted Values on Training Data', color='green')
        ax.plot(forecast_test['ds'], forecast_test['yhat'], label='Forecasted Sales', color='orange')
        ax.set_title('Prophet Model - Forecast')
        ax.set_xlabel('Date')
        ax.set_ylabel('Weekly Sales')
        ax.legend()
        st.pyplot(fig)

    def fit_auto_sarima(self, train_df, test_df):
        """
        Uses auto_arima to determine the best SARIMA parameters and fits the model.
        """
        train_series = train_df.set_index('ds')['y']

        # Fit the auto_arima model to find the best SARIMA parameters with optimizations
        auto_model = auto_arima(train_series, 
                                seasonal=True, 
                                m=52, 
                                trace=True, 
                                error_action='ignore', 
                                suppress_warnings=True, 
                                stepwise=True, 
                                n_jobs=-1,  
                                max_p=2, max_d=1, max_q=2,  
                                max_P=1, max_D=1, max_Q=1, 
                                max_order=5, 
                                max_runtime=100) 

        st.write(f'Best ARIMA Order: {auto_model.order}')
        st.write(f'Best Seasonal Order: {auto_model.seasonal_order}')

        # Fit the SARIMA model with the best-found parameters
        sarima_model = SARIMAX(train_series, order=auto_model.order, seasonal_order=auto_model.seasonal_order)
        sarima_result = sarima_model.fit(disp=False)

        # Forecasting within the training period for evaluation
        forecast_train = sarima_result.get_prediction(start=0, end=len(train_series)-1)
        y_pred_train = forecast_train.predicted_mean
        y_true_train = train_series.values

        # Calculating metrics on the training data
        mse_train = mean_squared_error(y_true_train, y_pred_train)
        rmse_train = sqrt(mse_train)
        r2_train = r2_score(y_true_train, y_pred_train)

        st.subheader("SARIMA Model Training Evaluation Metrics")
        st.write(f"RMSE: {rmse_train}")
        st.write(f"MSE: {mse_train}")
        st.write(f"R²: {r2_train}")

        # Forecasting on the test data
        test_series = pd.Series(index=test_df['ds'], dtype='float64')
        forecast_test = sarima_result.get_forecast(steps=len(test_series))
        y_pred_test = forecast_test.predicted_mean

        # Visualize the forecast on the test data
        st.subheader("SARIMA Model Forecast on Test Data")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_df['ds'], train_series, label='Training Data', color='blue')
        ax.plot(train_df['ds'], y_pred_train, label='Fitted Values on Training Data', color='green')
        ax.plot(test_df['ds'], y_pred_test, label='Forecasted Sales', color='orange')
        ax.set_title('SARIMA Model - Forecast')
        ax.set_xlabel('Date')
        ax.set_ylabel('Weekly Sales')
        ax.legend()
        st.pyplot(fig)


def main():
    train_file = './LR1/train Life Expectancy Data.csv'
    train_df = pd.read_csv(train_file)
    train_df.columns = train_df.columns.str.strip().str.lower()

    test_file = './LR1/test Life Expectancy Data.csv'
    test_df = pd.read_csv(test_file)
    test_df.columns = test_df.columns.str.strip().str.lower()

    update_file = 'Life-Expectancy-Data-Updated.csv'
    update_df = pd.read_csv(update_file)
    update_df.columns = update_df.columns.str.strip().str.lower()

    # Country name mapping 
    country_name_mapping = {
        "united states of america": "united states",
        "congo": "congo, rep.",
        "congo, dem. rep.": "democratic republic of the congo",
        "iran, islamic rep.": "iran (islamic republic of)",
        "united kingdom of great britain and northern ireland": "united kingdom",
        "republic of moldova": "moldova",
        "bahamas": "bahamas, the",
        "korea, dem. people’s rep.": "democratic people's republic of korea",
        "korea, rep": "republic of korea",
        "north macedonia": "the former yugoslav republic of macedonia",
        "egypt": "egypt, arab rep.",
        "sudan": "sudan",
        "south sudan": "south sudan",
        "yemen": "yemen, rep.",
        "saint vincent and the grenadines": "st. vincent and the grenadines",
        "nauru": "nauru",
        "bolivia (plurinational state of)": "bolivia",
        "côte d'ivoire": "cote d'ivoire",
        "venezuela, rb": "venezuela (bolivarian republic of)",
        "turkey": "turkiye",
        "viet nam": "vietnam",
        "cook islands": "cook islands",
        "kyrgyz republic": "kyrgyzstan",
        "monaco": "monaco",
        "st. lucia": "saint lucia",
        "palau": "palau",
        "swaziland": "eswatini",
        "lao pdr": "lao people's democratic republic",
        "united republic of tanzania": "tanzania",
        "st. kitts and nevis": "saint kitts and nevis",
        "slovak republic": "slovakia",
        "micronesia, fed. sts.": "micronesia (federated states of)",
        "gambia": "gambia, the",
        "dominica": "dominica",
    }

    train_eda = Life_Expectancy(train_df)
    test_eda = Life_Expectancy(test_df)

    tab1, tab2 = st.tabs(["Time Series", "Linear Regression"])

    with tab2:
        st.header("Linear Regression")
        
        st.subheader("Exploratory Data Analysis")
        train_eda.display_head()
        train_eda.display_description()
        train_eda.display_missing_values()
        train_eda.display_heatmap()
        train_eda.display_histograms()
        train_eda.display_pairplots()
        train_eda.display_violin_plot()
        train_eda.display_boxplots()

        st.subheader("Data Cleaning - Train Data")
        train_eda.fill_missing_values()
        st.write("As the first part of filling null values, where available, empty fields have been filled based on the mean of one row above and one row below grouped by country.")
        train_eda.fill_population_from_updated_data(update_df, country_name_mapping)
        st.write("Next step that just took place is, to further reduce null values in population, another dataset has been used to merge with to fill missing data, as well as mapping through a dictionary and a library called fuzzy wuzzy for word matching are used to merge the two columns and fill population properly.")
        train_eda.fill_missing_values()

        train_eda.fill_gdp_from_per_capita(update_df)
        st.write("Same process as population has now taken place for GDP column null field")

        train_eda.fill_schooling_from_update(update_df)
        st.write("Same process as population has now taken place for Schooling column null field")

        train_eda.fill_hepatitis_b_from_update(update_df)
        st.write("Same process as population has now taken place for hepitits b column null field")
        train_eda.fill_missing_values()
        train_eda.fill_missing_with_mean()
        st.write("Any leftover null values are filled based on full column mean value")
        train_eda.display_missing_values()
        st.write("Null Values handling end here.")

        train_eda.drop_column("infant deaths")

        st.subheader("Data Cleaning - Test Data")
        test_eda.display_head()
        test_eda.fill_missing_values()
        st.write("As the first part of filling null values, where available, empty fields have been filled based on the mean of one row above and one row below grouped by country.")

        test_eda.fill_population_from_updated_data(update_df, country_name_mapping)
        test_eda.fill_missing_values()
        st.write("Next step that just took place is, to further reduce null values in population, another dataset has been used to merge with to fill missing data, as well as mapping through a dictionary and a library called fuzzy wuzzy for word matching are used to merge the two columns and fill population properly.")

        test_eda.fill_gdp_from_per_capita(update_df)
        st.write("Same process as population has now taken place for GDP column null field")

        test_eda.fill_schooling_from_update(update_df)
        st.write("Same process as population has now taken place for Schooling column null field")

        test_eda.fill_hepatitis_b_from_update(update_df)
        st.write("Same process as population has now taken place for hepitits b column null field")

        test_eda.fill_missing_values()
        test_eda.fill_missing_with_mean()
        st.write("Any leftover null values are filled based on full column mean value")
        test_eda.display_missing_values()

        test_eda.drop_column("infant deaths")

        st.header("Regression Model")
        
        features = ['adult mortality', 'hiv/aids', 'income composition of resources', 'schooling', 'alcohol', 'percentage expenditure', 'bmi', 
                    'hepatitis b', 'measles', 'under-five deaths', 'polio', 'total expenditure', 'thinness  1-19 years',
                     'thinness 5-9 years', 'schooling' ]
        target = 'life expectancy'
        
        model = train_eda.train_regression_model(features, target)
        test_eda.apply_regression_model(model, features, target)

        st.subheader("Why use Linear Regression?")
        st.write("Linear regression is simple and provides clear results, showing how each factor like adult mortality or schooling affects life expectancy")
        st.write("changes in features cause a propertional change in life expectancy, making linear regression a nice choice.")
        st.subheader("Why use the chosen error metrics?")
        st.write("RMSE helps show large prediction errors more clearly.")
        st.write("R² measures how well your model explains the variance in the target variable. It ranges from 0 to 1, where 1 means the model perfectly explains the data.")
        st.write("Helps catch large errors more effectively, similar to RMSE")

        # Train and apply Random Forest model
        rf_model = train_eda.train_random_forest_model(features, target)
        test_eda.apply_regression_model(rf_model, features, target, model_name="Random Forest")
        st.subheader("why use Random forest?")
        st.write("it builds multiple decision trees and averages their predictions. This averaging process reduces the risk of overfitting, which is a common problem in single decision trees.")


        # Train and apply Ridge Regression model
        ridge_model = train_eda.train_ridge_model(features, target)
        test_eda.apply_regression_model(ridge_model, features, target, model_name="Ridge Regression")
        st.subheader("Why use Ridge model?")
        st.write("elps prevent the model from fitting the noise in the data, which is especially important when dealing with datasets that have many features or when the features are highly correlated.")

        st.subheader("Comparison of Models")
        st.write("Based on above models, Random forest appears to be the best regression model. The reason for this is: builds multiple decision trees during training. Each tree makes predictions, and the final output is the average of all tree predictions. ")

    with tab1:
        st.header("Time Series")
        train_df = pd.read_csv('./TS2/train.csv')
        test_df = pd.read_csv('./TS2/test.csv')
        stores_df = pd.read_csv('./TS2/stores.csv')
        features_df = pd.read_csv('./TS2/features.csv')
        ts_analysis = TimeSeries(train_df, stores_df, features_df)
        ts_analysis2 = TimeSeries(test_df, stores_df, features_df)

        st.header("Store and Features Data Merging")
        ts_analysis.merge_data()
        ts_analysis.display_description()

        st.header("Explatory Analysis")
        ts_analysis.check_null_values()
        ts_analysis.correlation_heatmap()
        st.markdown(f"**Explanation:** The correlation heatmap displays the relationships between numeric features, helping identify which variables might be strongly related.")
        ts_analysis.display_histograms()
        st.markdown(f"**Explanation:** Histograms show the distribution of individual numeric features, providing insights into their range, central tendency, and spread.")
        ts_analysis.display_pairplots()
        st.markdown(f"**Explanation: **Generates and displays pair plots for the most relevant columns in the merged DataFrame.")
        ts_analysis.display_boxplots()
        st.markdown("**Explanation: ** Generates and displays box plots whihc help find outliers")
        ts_analysis.display_pie_chart()
        st.markdown("**Explanation: ** Generates and displays a pie chart to see store type split")
        st.subheader("Time Series Decomposition")
        ts_analysis.aggregated_time_series_decomposition()
        st.markdown("**Explanation: ** Helps find trends and seasonalities in the data")
        ts_analysis.plot_acf()
        st.markdown("**Explanation: ** Helps find relationships between a time series and its lags")
        ts_analysis.plot_pacf()
        st.markdown("**Explanation: ** Helps find relationships between a time series and its lags")
        ts_analysis.moving_average_analysis()


        st.header("Preprocessing - Train Data")
        ts_analysis.fill_nulls_with_zero()
        ts_analysis.check_null_values()
        ts_analysis.one_hot_encode_holiday()

        st.header("Preprocessing - Test Data")
        ts_analysis2.merge_data()
        ts_analysis2.check_null_values()
        ts_analysis2.fill_nulls_with_zero()
        ts_analysis2.fill_nulls_cpi_unemployment()
        ts_analysis2.check_null_values()

        st.header("Prophet Model - Prediction")
        train_prophet_df = ts_analysis.preprocess_for_prophet(is_train=True)
        test_prophet_df = ts_analysis2.preprocess_for_prophet(is_train=False)
        ts_analysis.fit_predict_prophet(train_prophet_df, test_prophet_df)

        st.header("SARIMA Model - Prediction")
        ts_analysis.fit_auto_sarima(train_prophet_df, test_prophet_df)

        st.header("Model Comparison")
        st.write(f"Prophet model has shown much higher accuracy comapred to other models, such as, SARIMA. Prophet has the ability"
                 "to handle non stationary data (data that has seasonality and trends), this meant that it was the most suitable model"
                 "for the dataset. Prophet is less sensitive to outliers, as it can automatically detect and adjust for them, making it more robust in real-world scenarios where anomalies are common."
                 )
        st.subheader("Why use the chosen error metrics?")
        st.write("RMSE helps show large prediction errors more clearly.")
        st.write("R² measures how well your model explains the variance in the target variable. It ranges from 0 to 1, where 1 means the model perfectly explains the data.")
        st.write("Helps catch large errors more effectively, similar to RMSE")



if __name__ == "__main__":
    main()
