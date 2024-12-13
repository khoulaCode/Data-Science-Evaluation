import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.subplots as sp
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Sidebar with options menu
with st.sidebar:
    st.header("Menu")
    selected = option_menu(
        menu_title="Evaluation Tasks",
        options=["Linear Regression", "Time Series"],
        icons=['bar-chart', 'bar-chart'],
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#08172e"},
            "icon": {"color": "white", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#061121"},
            "nav-link-selected": {"background-color": "#08172e"}
        }
    )

def load_data_lr(train_file_lr, test_file_lr):
    try:
        train_data_lr = pd.read_csv(train_file_lr)
        test_data_lr = pd.read_csv(test_file_lr)
        
        # Clean column names by stripping any extra spaces
        train_data_lr.columns = train_data_lr.columns.str.strip()
        test_data_lr.columns = test_data_lr.columns.str.strip()

        return train_data_lr, test_data_lr
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def load_and_clean_ts(train_df_ts, test_df_ts):
    try:   
        # Load the datasets
        train_df_ts = pd.read_csv('Train_Coffee_Sales.csv')
        test_df_ts = pd.read_csv('Test_Coffee_Sales.csv')

        # Feature Engineering
        train_df_ts['datetime'] = pd.to_datetime(train_df_ts['datetime'])
        test_df_ts['datetime'] = pd.to_datetime(test_df_ts['datetime'])

        # Additional feature engineering (add more if necessary)
        train_df_ts['month'] = train_df_ts['datetime'].dt.month
        train_df_ts['day'] = train_df_ts['datetime'].dt.day
        train_df_ts['hour'] = train_df_ts['datetime'].dt.hour

        test_df_ts['month'] = test_df_ts['datetime'].dt.month
        test_df_ts['day'] = test_df_ts['datetime'].dt.day
        test_df_ts['hour'] = test_df_ts['datetime'].dt.hour

        # Define the features and target
        features_ts = ['month', 'day', 'hour', 'cash_type', 'card', 'coffee_name']
        target_ts = 'money'

        X_train_ts = train_df_ts[features_ts]
        y_train_ts = train_df_ts[target_ts]

        X_test_ts = test_df_ts[features_ts]
        y_test_ts = test_df_ts[target_ts]

        return train_df_ts, test_df_ts
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def perform_eda_lr(train_data_lr):
    st.title('Exploratory Data Analysis')
    st.title("Linear Regression")

    # Sample the data if it's too large
    sample_size = min(1000, len(train_data_lr))
    train_data_lr = train_data_lr.sample(sample_size, random_state=42)

    # Numeric columns
    numeric_columns_lr = train_data_lr.select_dtypes(include=[np.number]).columns
    non_numeric_columns_lr = train_data_lr.select_dtypes(exclude=[np.number]).columns

    # Histograms
    st.title('Histograms of Numeric Features')
    st.header('Purpose:')
    st.write('Histograms display the distribution of a single numeric variable by dividing the data into bins and counting the number of observations in each bin.')
    st.header('Explanation')
    st.write('What it Shows: The shape of the distribution of each numeric feature, such as whether it’s skewed or normally distributed.')
    st.write('Why it’s Useful: Helps in understanding the frequency distribution of numeric variables, identifying outliers, and checking for normality or skewness in the data.')

    num_histograms_lr = len(numeric_columns_lr)
    num_rows_histograms_lr = (num_histograms_lr + 1) // 2

    fig_lr_histograms_lr = sp.make_subplots(
        rows=num_rows_histograms_lr, cols=2,
        subplot_titles=numeric_columns_lr,
        vertical_spacing=0.05
    )

    for i, col in enumerate(numeric_columns_lr):
        row = i // 2 + 1
        col_pos = i % 2 + 1
        fig_lr_histograms_lr.add_trace(
            go.Histogram(x=train_data_lr[col], nbinsx=20, name=col),
            row=row, col=col_pos
        )

    fig_lr_histograms_lr.update_layout(
        title_text='Histograms of Numeric Features',
        height=350 * num_rows_histograms_lr,  # Increase height
        width=1000,  # Set a fixed width for the entire fig_lrure
        showlegend=False
    )
    st.plotly_chart(fig_lr_histograms_lr)

    # Box Plots
    st.title('Box Plots of Numeric Features')
    st.header('Purpose:')
    st.write('Box plots provide a visual summary of the distribution of a numeric variable, including its median, quartiles, and potential outliers.')
    st.header('Explanation')
    st.write('What it Shows: The spread and skewness of the data, including the median, quartiles, and any potential outliers.')
    st.write('Why it’s Useful: Helps in identifying outliers and understanding the variability of the data. It is especially useful for detecting differences in distributions across different groups if combined with categorical variables.')

    num_box_plots_lr = len(numeric_columns_lr)
    num_rows_box_plots_lr = (num_box_plots_lr + 1) // 2

    fig_lr_box_plots_lr = sp.make_subplots(
        rows=num_rows_box_plots_lr, cols=2,
        subplot_titles=numeric_columns_lr,
        vertical_spacing=0.05
    )

    for i, col in enumerate(numeric_columns_lr):
        row = i // 2 + 1
        col_pos = i % 2 + 1
        fig_lr_box_plots_lr.add_trace(
            go.Box(y=train_data_lr[col], name=col),
            row=row, col=col_pos
        )

    fig_lr_box_plots_lr.update_layout(
        title_text='Box Plots of Numeric Features',
        height=300 * num_rows_box_plots_lr,
        showlegend=False
    )
    st.plotly_chart(fig_lr_box_plots_lr)

    # Correlation Matrix
    st.title('Correlation Matrix')
    st.header('Purpose:')
    st.write('A correlation matrix displays the pairwise correlations between numeric features.')
    st.header('Explanation')
    st.write('What it Shows: The strength and direction of linear relationships between pairs of numeric variables.')
    st.write('Why it’s Useful: Helps in identifying multicollinearity (high correlation between features) and understanding the relationships between different variables.')

    corr_lr = train_data_lr[numeric_columns_lr].corr()
    fig_lr_corr_lr = go.Figure(data=go.Heatmap(
        z=corr_lr.values,
        x=corr_lr.columns,
        y=corr_lr.columns,
        colorscale='Viridis',
        colorbar=dict(title='Correlation')
    ))
    fig_lr_corr_lr.update_layout(title='Correlation Matrix')
    st.plotly_chart(fig_lr_corr_lr)

    # Scatter Plot Matrix
    st.title('Scatter Plot Matrix')
    st.header('Purpose:')
    st.write('A scatter plot matrix shows scatter plots for all pairs of numeric features.')
    st.header('Explanation')
    st.write('What it Shows: The relationships between each pair of numeric features, allowing for easy identification of patterns, correlations, and potential outliers.')
    st.write('Why it’s Useful: Useful for examining interactions between features and understanding how they relate to each other.')

    if len(numeric_columns_lr) > 4:
        subset_columns_lr = numeric_columns_lr[:4]
    else:
        subset_columns_lr = numeric_columns_lr

    if len(subset_columns_lr) >= 2:
        fig_lr_scatter_lr = px.scatter_matrix(
            train_data_lr,
            dimensions=subset_columns_lr,
            title='Scatter Plot Matrix'
        )
        st.plotly_chart(fig_lr_scatter_lr)

    # Hexbin Plot (using Plotly)
    st.title('Hexbin Plot')
    st.header('Purpose:')
    st.write('A hexbin plot visualizes the density of data points in a two-dimensional space using hexagonal bins.')
    st.header('Explanation')
    st.write('What it Shows: The density of data points in a scatter plot format, where colors indicate the number of points in each bin.')
    st.write('Why it’s Useful: Useful for visualizing the density of data points and identifying areas of high or low concentration.')

    if len(numeric_columns_lr) >= 2:
        x_col, y_col = numeric_columns_lr[:2]
        fig_lr_hexbin_lr = px.density_heatmap(
            train_data_lr,
            x=x_col,
            y=y_col,
            nbinsx=30,
            nbinsy=30,
            color_continuous_scale='Viridis',
            title=f'Hexbin Plot of {x_col} vs {y_col}'
        )
        st.plotly_chart(fig_lr_hexbin_lr)

    # Violin Plot (if there are categorical features)
    st.title('Violin Plots')
    st.header('Purpose:')
    st.write('Violin plots combine aspects of box plots and density plots, showing the distribution of data across different categories.')
    st.header('Explanation')
    st.write('Why it’s Useful: Provides a detailed view of the distribution of a numeric variable within categorical groups, highlighting differences between categories.')

    if non_numeric_columns_lr.size > 0:
        for cat_col_lr in non_numeric_columns_lr:
            # Ensure categorical data is treated correctly
            train_data_lr[cat_col_lr] = train_data_lr[cat_col_lr].astype(str)
            
            for num_col_lr in numeric_columns_lr:
                # Print to debug the data used in the plot
                st.header(f"Creating Violin Plot for {num_col_lr} by {cat_col_lr}")
                
                if train_data_lr[cat_col_lr].nunique() > 1:  # Ensure there's more than one unique category
                    fig_lr_violin_lr = px.violin(
                        train_data_lr,
                        y=num_col_lr,
                        x=cat_col_lr,
                        box=True,
                        points="all",
                        title=f'Violin Plot of {num_col_lr} by {cat_col_lr}'
                    )
                    st.plotly_chart(fig_lr_violin_lr)


def eda_coffee_sales(train_file, test_file):
    # Load the datasets
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    st.title("Time Series Analysis")
    st.title('Exploratory Data Analysis')

    # Ensure datetime columns are not included in numeric calculations
    if 'datetime' in train_df.columns:
        train_df['datetime'] = pd.to_datetime(train_df['datetime'], errors='coerce')
    if 'datetime' in test_df.columns:
        test_df['datetime'] = pd.to_datetime(test_df['datetime'], errors='coerce')
    
    # Histograms for numeric features
    numeric_columns = train_df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        fig = px.histogram(train_df, x=col, title=f'Histogram of {col}')
        st.plotly_chart(fig)
    
    # Bar charts for categorical features
    categorical_columns = train_df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        fig = px.bar(train_df[col].value_counts(), title=f'Bar Chart of {col}')
        st.plotly_chart(fig)
    
    # Scatter plot matrix for numeric columns only
    if len(numeric_columns) > 1:
        st.write("Scatter Plot Matrix for Train Dataset:")
        fig = px.scatter_matrix(train_df[numeric_columns], title="Scatter Plot Matrix")
        st.plotly_chart(fig)

def clean_data_lr(df_lr):
    # Drop 'infant deaths' column
    df_lr.drop(columns=['infant deaths'], inplace=True, errors='ignore')
    df_lr.drop(columns=['percentage expenditure'], inplace=True,errors='ignore')
    
    # Handle missing values
    for col_lr in df_lr.columns:
        if df_lr[col_lr].dtype == np.number:
            df_lr[col_lr].fillna(df_lr[col_lr].mean(), inplace=True)
        else:
            df_lr[col_lr].fillna(df_lr[col_lr].mode()[0], inplace=True)
    
    # Remove duplicates
    df_lr.drop_duplicates(inplace=True)
    
    # Convert categorical columns to string type for consistency
    categorical_columns_lr = df_lr.select_dtypes(include=['object']).columns
    df_lr[categorical_columns_lr] = df_lr[categorical_columns_lr].astype(str)
    
    # Standardize date columns if any
    date_columns_lr = [col_lr for col_lr in df_lr.columns if 'date' in col_lr]
    for col_lr in date_columns_lr:
        df_lr[col_lr] = pd.to_datetime(df_lr[col_lr], errors='coerce')
    
    return df_lr

def preprocess_and_train_lr(train_data_lr, test_data_lr):
    # Clean data
    train_data_lr = clean_data_lr(train_data_lr)
    test_data_lr = clean_data_lr(test_data_lr)

    # Handle non-numeric columns
    non_numeric_columns_lr = train_data_lr.select_dtypes(exclude=[np.number]).columns
    numeric_columns_lr = train_data_lr.select_dtypes(include=[np.number]).columns

    # Define preprocessing for non-numeric columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='mean'), numeric_columns_lr),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), non_numeric_columns_lr)
        ]
        
    )

    # Apply preprocessing
    X_train_lr = preprocessor.fit_transform(train_data_lr)
    X_test_lr = preprocessor.transform(test_data_lr)

    # Standardize features
    scaler = StandardScaler()
    X_train_lr_scaled = scaler.fit_transform(X_train_lr)
    X_test_lr_scaled = scaler.transform(X_test_lr)

    # Obtain feature names from ColumnTransformer
    predict_and_evaluate_lr = numeric_columns_lr.tolist()
    categorical_feature_names_lr = list(preprocessor.transformers_[1][1].get_feature_names_out())
    
    feature_names_lr = predict_and_evaluate_lr + categorical_feature_names_lr

    # Create DataFrame with correct feature names
    X_train_lr_df_lr = pd.DataFrame(X_train_lr_scaled, columns=feature_names_lr)
    X_test_lr_df_lr = pd.DataFrame(X_test_lr_scaled, columns=feature_names_lr)
    
    y = train_data_lr['Life expectancy']

    # Split the data into training and validation sets
    X_train_lr_split, X_val_lr, y_train_split, y_val_lr = train_test_split(X_train_lr_df_lr, y, test_size=0.2, random_state=42)

    # Initialize and train models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Decision Tree Regressor': DecisionTreeRegressor()
    }
    
    # Dictionary to store model evaluation metrics
    metrics = {}

    for name, model in models.items():
        # Train the model
        model.fit(X_train_lr_split, y_train_split)
        
        # Predict on the validation set
        y_pred_lr = model.predict(X_val_lr)
        
        # Calculate evaluation metrics
        mse = mean_squared_error(y_val_lr, y_pred_lr)
        mae = mean_absolute_error(y_val_lr, y_pred_lr)
        r2 = r2_score(y_val_lr, y_pred_lr)

        # Store metrics
        metrics[name] = {
            'MSE': mse,
            'MAE': mae,
            'R2': r2
        }

     
    return models, preprocessor, scaler, metrics

def predict_and_evaluate_lr(models, preprocessor, scaler, test_data_lr):
    # Clean data
    test_data_lr = clean_data_lr(test_data_lr)

    # Drop columns with high correlation 
    test_data_lr = test_data_lr.drop(columns=['Infant deaths'], errors='ignore')
    
    # Apply preprocessing
    X_test_lr = preprocessor.transform(test_data_lr)

    # Standardize features
    X_test_lr_scaled = scaler.transform(X_test_lr)

    # Obtain feature names from ColumnTransformer
    predict_and_evaluate_lr = test_data_lr.select_dtypes(include=[np.number]).columns.tolist()
    categorical_feature_names_lr = list(preprocessor.transformers_[1][1].get_feature_names_out())
    
    feature_names_lr = predict_and_evaluate_lr + categorical_feature_names_lr

    # Create DataFrame with correct feature names
    X_test_lr_df_lr = pd.DataFrame(X_test_lr_scaled, columns=feature_names_lr)

    # Store predictions for each model
    predictions = {}

    st.header("Model Explanations and Evaluations")
    st.header("We used the following models for the dataset:")
    st.write("We removed columns with high correlation to reduce redundancy and improve model performance. Specifically, the 'Infant deaths' and 'percentage expenditure' columns were dropped.")

    # Evaluate each model on the test data
    for name, model in models.items():
        st.header(f"* {name}")
        st.write(f"**Model Explanation:**")
        
        # Explanation for each model
        if name == 'Linear Regression':
            st.write("Linear Regression is used to model the relationship between the dependent variable and one or more independent variables by fitting a linear equation to observed data. It's useful for understanding the linear relationship and making predictions.")
        elif name == 'Ridge Regression':
            st.write("Ridge Regression includes a regularization term to handle multicollinearity and prevent overfitting. It is useful when the model has too many variables and needs to balance the trade-off between fitting the data and keeping the model simple.")
        elif name == 'Lasso Regression':
            st.write("Lasso Regression performs both variable selection and regularization. It can eliminate some variables completely, making it useful for simplifying the model and improving interpretability.")
        elif name == 'Decision Tree Regressor':
            st.write("Decision Tree Regressor creates a tree-like model of decisions and their consequences. It is useful for capturing non-linear relationships and interactions between features.")

        # Predict on the test set
        predictions[name] = model.predict(X_test_lr_df_lr)
        
        # Assuming we have the true values for comparison
        if 'Life expectancy' in test_data_lr.columns:
            y_true_lr = test_data_lr['Life expectancy']
            mae = mean_absolute_error(y_true_lr, predictions[name])
            mse = mean_squared_error(y_true_lr, predictions[name])
            r2 = r2_score(y_true_lr, predictions[name])

            # Display metrics
            st.write(f"**Evaluation Metrics:**")
            st.write(f"  - Mean Squared Error: {mse}")
            st.write(f"  - Mean Absolute Error: {mae}")
            st.write(f"  - R-squared: {r2}")

            # Plotting predictions vs. true values
            fig_lr = px.scatter(
                x=y_true_lr,
                y=predictions[name],
                labels={'x': 'True Values', 'y': 'Predictions'},
                title=f'Predictions vs True Values for {name}'
            )
            fig_lr.update_layout(xaxis_title='True Values', yaxis_title='Predictions')
            st.plotly_chart(fig_lr)

        else:
            st.write(f"Test data does not contain 'Life expectancy' for {name} evaluation.")
    
    return predictions

def preprocess_ts(train_df_ts, test_df_ts):
    # Load and clean the data
    train_df_ts, test_df_ts = load_and_clean_ts(train_df_ts, test_df_ts)
    
    if train_df_ts is None or test_df_ts is None:
        return None

    # Preprocessing pipeline
    numeric_features = ['month', 'day', 'hour']
    categorical_features = ['cash_type', 'card', 'coffee_name']

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Option 1: Stacking Regressor without cross-validation inside
    stacking_regressor = StackingRegressor(
        estimators=[
            ('ridge', Ridge()),
            ('dt', DecisionTreeRegressor())
        ],
        final_estimator=LGBMRegressor()
    )

    pipeline_stacking_ts = Pipeline(steps=[('preprocessor', preprocessor),
                                           ('model', stacking_regressor)])

    # Split data into features and target
    X_train_ts = train_df_ts.drop(columns=['money'])
    y_train_ts = train_df_ts['money']
    X_test_ts = test_df_ts.drop(columns=['money'])
    y_test_ts = test_df_ts['money']

    pipeline_stacking_ts.fit(X_train_ts, y_train_ts)
    y_pred_stacking_ts = pipeline_stacking_ts.predict(X_test_ts)

    r2_stacking = r2_score(y_test_ts, y_pred_stacking_ts)
    mae_stacking = mean_absolute_error(y_test_ts, y_pred_stacking_ts)
    rmse_stacking = np.sqrt(mean_squared_error(y_test_ts, y_pred_stacking_ts))

    # Option 2: Voting Regressor
    voting_regressor = VotingRegressor(
        estimators=[
            ('lgbm', LGBMRegressor()),
            ('xgb', XGBRegressor()),
            ('catboost', CatBoostRegressor(silent=True))
        ]
    )

    pipeline_voting = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('model', voting_regressor)])

    pipeline_voting.fit(X_train_ts, y_train_ts)
    y_pred_voting = pipeline_voting.predict(X_test_ts)

    r2_voting = r2_score(y_test_ts, y_pred_voting)
    mae_voting = mean_absolute_error(y_test_ts, y_pred_voting)
    rmse_voting = np.sqrt(mean_squared_error(y_test_ts, y_pred_voting))

    stacking_results = {
        'predictions': y_pred_stacking_ts,
        'r2': r2_stacking,
        'mae': mae_stacking,
        'rmse': rmse_stacking
    }
    
    voting_results = {
        'predictions': y_pred_voting,
        'r2': r2_voting,
        'mae': mae_voting,
        'rmse': rmse_voting
    }

    return {
        'stacking': stacking_results,
        'voting': voting_results
    }

def plot_ts_results(y_test_ts, y_pred_stacking_ts, y_pred_voting, r2_stacking, mae_stacking, rmse_stacking, r2_voting, mae_voting, rmse_voting):
    st.header('Time Series prediction is done using Stacking Regressor & Voting Regressor regressors')
    st.write('I used these two regressors because they gave me the most acurate results. I have tried multiple models. However these two had the best scores.')
    
    # st.header('Summary of steps:')
    # st.write('**Feature Engineering**: Extracted hour, day_of_week, and month from the datetime column.')

    st.header('Stacking Regressor Results')
    st.write(f"**Stacking Regressor - R-squared:** {r2_stacking:.4f}")
    st.write(f"**Stacking Regressor - Mean Absolute Error:** {mae_stacking:.2f}")
    st.write(f"**Stacking Regressor - Root Mean Squared Error:** {rmse_stacking:.2f}")
    
    st.header('Voting Regressor Results')
    st.write(f"**Voting Regressor - R-squared:** {r2_voting:.4f}")
    st.write(f"**Voting Regressor - Mean Absolute Error:** {mae_voting:.2f}")
    st.write(f"**Voting Regressor - Root Mean Squared Error:** {rmse_voting:.2f}")
    
    # Predicted vs. Actual Values Plot
    st.subheader('Predicted vs Actual Values')
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_test_ts, y_pred_stacking_ts, alpha=0.5)
    plt.plot([y_test_ts.min(), y_test_ts.max()], [y_test_ts.min(), y_test_ts.max()], '--r', linewidth=2)
    plt.title('Stacking Regressor: Predicted vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test_ts, y_pred_voting, alpha=0.5)
    plt.plot([y_test_ts.min(), y_test_ts.max()], [y_test_ts.min(), y_test_ts.max()], '--r', linewidth=2)
    plt.title('Voting Regressor: Predicted vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    
    st.pyplot(plt)
    
    # Residuals Plot
    st.subheader('Residuals Plot')
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    residuals_stacking = y_test_ts - y_pred_stacking_ts
    plt.scatter(y_pred_stacking_ts, residuals_stacking, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--', linewidth=2)
    plt.title('Stacking Regressor: Residuals')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    
    plt.subplot(1, 2, 2)
    residuals_voting = y_test_ts - y_pred_voting
    plt.scatter(y_pred_voting, residuals_voting, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--', linewidth=2)
    plt.title('Voting Regressor: Residuals')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    
    st.pyplot(plt)
    
    # Error Distribution Plot
    st.subheader('Error Distribution Plot')
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(residuals_stacking, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Stacking Regressor: Error Distribution')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.hist(residuals_voting, bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.title('Voting Regressor: Error Distribution')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    
    st.pyplot(plt)

def main():
    if selected == "Linear Regression":
        train_file_lr = 'train_Life_Expectancy_Data.csv'
        test_file_lr = 'test_life_Expectancy_Data.csv'
        
        train_data_lr, test_data_lr = load_data_lr(train_file_lr, test_file_lr)
        
        if train_data_lr is not None and test_data_lr is not None:
            # Perform EDA before any data cleaning
            perform_eda_lr(train_data_lr)
            
            # Preprocess data and train models
            models, preprocessor, scaler, metrics = preprocess_and_train_lr(train_data_lr, test_data_lr)
            
            # Predict and evaluate on the test data
            predictions = predict_and_evaluate_lr(models, preprocessor,scaler,test_data_lr)
    
    elif selected == "Time Series":
        train_df_ts = pd.read_csv('Train_Coffee_Sales.csv')
        test_df_ts = pd.read_csv('Test_Coffee_Sales.csv')

        eda_coffee_sales('Train_Coffee_Sales.csv', 'Test_Coffee_Sales.csv')
        
        # Preprocess time series data and train models
        ts_results = preprocess_ts(train_df_ts, test_df_ts)

        if ts_results is not None:
            plot_ts_results(
                y_test_ts=test_df_ts['money'],
                y_pred_stacking_ts=ts_results['stacking']['predictions'],
                y_pred_voting=ts_results['voting']['predictions'],
                r2_stacking=ts_results['stacking']['r2'],
                mae_stacking=ts_results['stacking']['mae'],
                rmse_stacking=ts_results['stacking']['rmse'],
                r2_voting=ts_results['voting']['r2'],
                mae_voting=ts_results['voting']['mae'],
                rmse_voting=ts_results['voting']['rmse']
            )


if __name__ == "__main__":
    main()
    

