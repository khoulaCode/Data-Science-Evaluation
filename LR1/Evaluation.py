import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.subplots as sp
import plotly.express as px
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer

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

def load_data(train_file, test_file):
    try:
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)
        
        # Clean column names by stripping any extra spaces
        train_data.columns = train_data.columns.str.strip()
        test_data.columns = test_data.columns.str.strip()

        return train_data, test_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def perform_eda(train_data):
    st.title('Exploratory Data Analysis')

    # Sample the data if it's too large
    sample_size = min(1000, len(train_data))
    train_data = train_data.sample(sample_size, random_state=42)

    # Numeric columns
    numeric_columns = train_data.select_dtypes(include=[np.number]).columns
    non_numeric_columns = train_data.select_dtypes(exclude=[np.number]).columns

    # Histograms
    st.title('Histograms of Numeric Features')
    st.header('Purpose:')
    st.write('Histograms display the distribution of a single numeric variable by dividing the data into bins and counting the number of observations in each bin.')
    st.header('Explanation')
    st.write('What it Shows: The shape of the distribution of each numeric feature, such as whether it’s skewed or normally distributed.')
    st.write('Why it’s Useful: Helps in understanding the frequency distribution of numeric variables, identifying outliers, and checking for normality or skewness in the data.')

    num_histograms = len(numeric_columns)
    num_rows_histograms = (num_histograms + 1) // 2

    fig_histograms = sp.make_subplots(
        rows=num_rows_histograms, cols=2,
        subplot_titles=numeric_columns,
        vertical_spacing=0.05
    )

    for i, col in enumerate(numeric_columns):
        row = i // 2 + 1
        col_pos = i % 2 + 1
        fig_histograms.add_trace(
            go.Histogram(x=train_data[col], nbinsx=20, name=col),
            row=row, col=col_pos
        )

    fig_histograms.update_layout(
        title_text='Histograms of Numeric Features',
        height=350 * num_rows_histograms,  # Increase height
        width=1000,  # Set a fixed width for the entire figure
        showlegend=False
    )
    st.plotly_chart(fig_histograms)

    # Box Plots
    st.title('Box Plots of Numeric Features')
    st.header('Purpose:')
    st.write('Box plots provide a visual summary of the distribution of a numeric variable, including its median, quartiles, and potential outliers.')
    st.header('Explanation')
    st.write('What it Shows: The spread and skewness of the data, including the median, quartiles, and any potential outliers.')
    st.write('Why it’s Useful: Helps in identifying outliers and understanding the variability of the data. It is especially useful for detecting differences in distributions across different groups if combined with categorical variables.')

    num_box_plots = len(numeric_columns)
    num_rows_box_plots = (num_box_plots + 1) // 2

    fig_box_plots = sp.make_subplots(
        rows=num_rows_box_plots, cols=2,
        subplot_titles=numeric_columns,
        vertical_spacing=0.05
    )

    for i, col in enumerate(numeric_columns):
        row = i // 2 + 1
        col_pos = i % 2 + 1
        fig_box_plots.add_trace(
            go.Box(y=train_data[col], name=col),
            row=row, col=col_pos
        )

    fig_box_plots.update_layout(
        title_text='Box Plots of Numeric Features',
        height=300 * num_rows_box_plots,
        showlegend=False
    )
    st.plotly_chart(fig_box_plots)

    # Correlation Matrix
    st.title('Correlation Matrix')
    st.header('Purpose:')
    st.write('A correlation matrix displays the pairwise correlations between numeric features.')
    st.header('Explanation')
    st.write('What it Shows: The strength and direction of linear relationships between pairs of numeric variables.')
    st.write('Why it’s Useful: Helps in identifying multicollinearity (high correlation between features) and understanding the relationships between different variables.')

    corr = train_data[numeric_columns].corr()
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='Viridis',
        colorbar=dict(title='Correlation')
    ))
    fig_corr.update_layout(title='Correlation Matrix')
    st.plotly_chart(fig_corr)

    # Scatter Plot Matrix
    st.title('Scatter Plot Matrix')
    st.header('Purpose:')
    st.write('A scatter plot matrix shows scatter plots for all pairs of numeric features.')
    st.header('Explanation')
    st.write('What it Shows: The relationships between each pair of numeric features, allowing for easy identification of patterns, correlations, and potential outliers.')
    st.write('Why it’s Useful: Useful for examining interactions between features and understanding how they relate to each other.')

    if len(numeric_columns) > 4:
        subset_columns = numeric_columns[:4]
    else:
        subset_columns = numeric_columns

    if len(subset_columns) >= 2:
        fig_scatter = px.scatter_matrix(
            train_data,
            dimensions=subset_columns,
            title='Scatter Plot Matrix'
        )
        st.plotly_chart(fig_scatter)

    # Hexbin Plot (using Plotly)
    st.title('Hexbin Plot')
    st.header('Purpose:')
    st.write('A hexbin plot visualizes the density of data points in a two-dimensional space using hexagonal bins.')
    st.header('Explanation')
    st.write('What it Shows: The density of data points in a scatter plot format, where colors indicate the number of points in each bin.')
    st.write('Why it’s Useful: Useful for visualizing the density of data points and identifying areas of high or low concentration.')

    if len(numeric_columns) >= 2:
        x_col, y_col = numeric_columns[:2]
        fig_hexbin = px.density_heatmap(
            train_data,
            x=x_col,
            y=y_col,
            nbinsx=30,
            nbinsy=30,
            color_continuous_scale='Viridis',
            title=f'Hexbin Plot of {x_col} vs {y_col}'
        )
        st.plotly_chart(fig_hexbin)

    # Violin Plot (if there are categorical features)
    st.title('Violin Plots')
    st.header('Purpose:')
    st.write('Violin plots combine aspects of box plots and density plots, showing the distribution of data across different categories.')
    st.header('Explanation')
    st.write('Why it’s Useful: Provides a detailed view of the distribution of a numeric variable within categorical groups, highlighting differences between categories.')

    if non_numeric_columns.size > 0:
        for cat_col in non_numeric_columns:
            # Ensure categorical data is treated correctly
            train_data[cat_col] = train_data[cat_col].astype(str)
            
            for num_col in numeric_columns:
                # Print to debug the data used in the plot
                st.header(f"Creating Violin Plot for {num_col} by {cat_col}")
                # st.write(train_data[[num_col, cat_col]].dropna().head())  # Displaying a sample of the data
                
                if train_data[cat_col].nunique() > 1:  # Ensure there's more than one unique category
                    fig_violin = px.violin(
                        train_data,
                        y=num_col,
                        x=cat_col,
                        box=True,
                        points="all",
                        title=f'Violin Plot of {num_col} by {cat_col}'
                    )
                    st.plotly_chart(fig_violin)

def clean_data(df):
    # Drop 'infant deaths' column
    df.drop(columns=['infant deaths'], inplace=True)
    df.drop(columns=['percentage expenditure'], inplace=True)
    
    # Handle missing values
    for col in df.columns:
        if df[col].dtype == np.number:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # Convert categorical columns to string type for consistency
    categorical_columns = df.select_dtypes(include=['object']).columns
    df[categorical_columns] = df[categorical_columns].astype(str)
    
    # Standardize date columns if any
    date_columns = [col for col in df.columns if 'date' in col]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    return df

def preprocess_and_train(train_data, test_data):
    # Clean data
    train_data = clean_data(train_data)
    test_data = clean_data(test_data)

    # Handle non-numeric columns
    non_numeric_columns = train_data.select_dtypes(exclude=[np.number]).columns
    numeric_columns = train_data.select_dtypes(include=[np.number]).columns

    # Define preprocessing for non-numeric columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='mean'), numeric_columns),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), non_numeric_columns)
        ]
        
    )

    # Apply preprocessing
    X_train = preprocessor.fit_transform(train_data)
    X_test = preprocessor.transform(test_data)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Obtain feature names from ColumnTransformer
    numeric_feature_names = numeric_columns.tolist()
    categorical_feature_names = list(preprocessor.transformers_[1][1].get_feature_names_out())
    
    feature_names = numeric_feature_names + categorical_feature_names

    # Create DataFrame with correct feature names
    X_train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    y = train_data['Life expectancy']

    # Split the data into training and validation sets
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train_df, y, test_size=0.2, random_state=42)

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
        model.fit(X_train_split, y_train_split)
        
        # Predict on the validation set
        y_pred = model.predict(X_val)
        
        # Calculate evaluation metrics
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        # Store metrics
        metrics[name] = {
            'MSE': mse,
            'MAE': mae,
            'R2': r2
        }

     
    return models, preprocessor, scaler, metrics

def predict_and_evaluate(models, preprocessor, scaler, test_data):
    # Clean data
    test_data = clean_data(test_data)

    # Drop columns with high correlation (if applicable)
    # Assuming 'Infant deaths' column was dropped
    test_data = test_data.drop(columns=['Infant deaths'], errors='ignore')
    
    # Apply preprocessing
    X_test = preprocessor.transform(test_data)

    # Standardize features
    X_test_scaled = scaler.transform(X_test)

    # Obtain feature names from ColumnTransformer
    numeric_feature_names = test_data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_feature_names = list(preprocessor.transformers_[1][1].get_feature_names_out())
    
    feature_names = numeric_feature_names + categorical_feature_names

    # Create DataFrame with correct feature names
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)

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
        predictions[name] = model.predict(X_test_df)
        
        # Assuming we have the true values for comparison
        if 'Life expectancy' in test_data.columns:
            y_true = test_data['Life expectancy']
            mae = mean_absolute_error(y_true, predictions[name])
            mse = mean_squared_error(y_true, predictions[name])
            r2 = r2_score(y_true, predictions[name])

            # Display metrics
            st.write(f"**Evaluation Metrics:**")
            st.write(f"  - Mean Squared Error: {mse}")
            st.write(f"  - Mean Absolute Error: {mae}")
            st.write(f"  - R-squared: {r2}")

            # Plotting predictions vs. true values
            fig = px.scatter(
                x=y_true,
                y=predictions[name],
                labels={'x': 'True Values', 'y': 'Predictions'},
                title=f'Predictions vs True Values for {name}'
            )
            fig.update_layout(xaxis_title='True Values', yaxis_title='Predictions')
            st.plotly_chart(fig)

        else:
            st.write(f"Test data does not contain 'Life expectancy' for {name} evaluation.")
    
    return predictions

def main():
    if selected == "Linear Regression":
        train_file = 'train_Life_Expectancy_Data.csv'
        test_file = 'test_life_Expectancy_Data.csv'
        
        train_data, test_data = load_data(train_file, test_file)
        
        if train_data is not None and test_data is not None:
            # Perform EDA before any data cleaning
            perform_eda(train_data)
            
            # Preprocess data and train models
            models, preprocessor, scaler, metrics = preprocess_and_train(train_data, test_data)
            
            # Predict and evaluate on the test data
            predictions = predict_and_evaluate(models, preprocessor,scaler,test_data)
    
    elif selected == "Time Series":
        st.write("Time Series functionality is not yet implemented.")

if __name__ == "__main__":
    main()

