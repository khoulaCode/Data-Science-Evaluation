import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import joblib


def main():
    # Title and Description
    st.title("Linear Regression for Real Estate Price Prediction")
    st.write("""
    Welcome to the Real Estate Price Prediction application. This tool is designed to help you explore real estate data, perform in-depth data analysis, and build predictive models to forecast house prices. In this evaluation task, we will walk through the process of data cleaning, exploratory data analysis (EDA), model building, and making predictions with detailed explanations and insights.
    """)

    # Data Upload Section
    st.header("1. Upload Your Dataset")
    st.write("""
    In this section, you'll upload your training and test datasets. The application will automatically clean and prepare the data for analysis.
    """)
    train_file = st.file_uploader("Upload the training dataset (CSV)", type=["csv"])
    test_file = st.file_uploader("Upload the test dataset (CSV)", type=["csv"])

    if train_file is not None and test_file is not None:
        # Read the datasets
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)

        # Display the first few rows of the data
        st.subheader("Training Data Overview")
        st.write("""
        Below is a preview of the training data that will be used to build the model. We will analyze the features, clean the data, and prepare it for modeling.
        """)
        st.write(train_data.head())

        st.subheader("Test Data Overview")
        st.write("""
        Below is a preview of the test data that will be used to evaluate the model's performance. We will ensure this data is consistent with the training data.
        """)
        st.write(test_data.head())

        # Display the column names
        st.subheader("Column Names in Training Data")
        st.write(train_data.columns)

        st.subheader("Column Names in Test Data")
        st.write(test_data.columns)

        # Identify numeric and non-numeric columns
        numeric_columns = train_data.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_columns = train_data.select_dtypes(exclude=[np.number]).columns.tolist()

        st.subheader("Numeric Columns in Training Data")
        st.write(numeric_columns)

        st.subheader("Non-Numeric Columns in Training Data")
        st.write(non_numeric_columns)

        # Data Cleaning and Feature Engineering
        st.header("2. Data Cleaning and Feature Engineering")
        st.write("""
        In this section, we undertake several crucial steps to prepare the data for effective modeling. Proper data cleaning and feature engineering are fundamental to building a robust predictive model. Heres what well do:
        - **Handle Missing Values**: Missing data can introduce bias or inaccuracies in the model. We fill missing values with the mean of the respective columns to maintain the integrity of the dataset.
        - **Drop Non-Numeric Columns**: Non-numeric columns are excluded from the analysis at this stage to focus on features that directly contribute to the numerical prediction of house prices. This simplification ensures that the model can be trained efficiently.
        - **Scale Features**: Scaling the numeric features standardizes the range of independent variables or features of data. This step is essential for algorithms that calculate distances between data points, such as in regression models. It ensures that all features contribute equally to the model.
        """)

        # Handle non-numeric columns (For now, we'll drop them)
        if non_numeric_columns:
            st.write(f"**Dropped Non-Numeric Columns:** {non_numeric_columns}")
            train_data = train_data.drop(columns=non_numeric_columns)
            test_data = test_data.drop(columns=non_numeric_columns)

        # Handle missing values
        train_data.fillna(train_data.mean(), inplace=True)
        test_data.fillna(test_data.mean(), inplace=True)

        # Feature Scaling
        scaler = StandardScaler()
        train_data[numeric_columns] = scaler.fit_transform(train_data[numeric_columns])
        test_data[numeric_columns] = scaler.transform(test_data[numeric_columns])

        st.write("""
        **Data Cleaning and Feature Engineering completed.** The following steps have been successfully applied:
        - Missing values have been handled to ensure no gaps in the data.
        - Non-numeric columns have been dropped, allowing us to focus on the numerical aspects of the dataset.
        - All features have been scaled, ensuring they are on a common scale, which is crucial for the accuracy and performance of our regression model.

        The data is now pre-processed and ready for the next stage: Exploratory Data Analysis (EDA). This preparation sets a solid foundation for building a reliable and accurate predictive model.
        """)

        # Exploratory Data Analysis (EDA)
        st.header("3. Exploratory Data Analysis (EDA)")
        st.write("""
        In this section, we explore the relationships and distributions within the dataset. Understanding these patterns helps in making informed decisions during model building.
        """)

        # Interactive Correlation Heatmap
        st.subheader("Correlation Heatmap")
        st.write("""
        The correlation heatmap below shows the relationships between numeric features. A high correlation (close to 1 or -1) between features can indicate multicollinearity, which we need to address in the modeling stage.
        """)
        corr_matrix = pd.DataFrame(train_data, columns=numeric_columns).corr().stack().reset_index()
        corr_matrix.columns = ['Feature 1', 'Feature 2', 'Correlation']

        heatmap = alt.Chart(corr_matrix).mark_rect().encode(
            x='Feature 1:O',
            y='Feature 2:O',
            color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='blueorange')),
            tooltip=['Feature 1', 'Feature 2', 'Correlation']
        ).properties(
            width=600,
            height=600
        )
        st.altair_chart(heatmap, use_container_width=True)

        # Interactive Scatter Plots
        st.subheader("Pairwise Scatter Plots")
        st.write("""
        These scatter plots illustrate the relationships between each feature and the target variable, 'Y house price of unit area'. Analyzing these plots helps us understand which features are most influential in predicting house prices.
        """)
        for feature in numeric_columns:
            scatter_plot = alt.Chart(train_data).mark_circle(size=60).encode(
                x=alt.X(feature, scale=alt.Scale(zero=False)),
                y=alt.Y('Y house price of unit area', scale=alt.Scale(zero=False)),
                tooltip=[feature, 'Y house price of unit area']
            ).interactive().properties(
                title=f'Scatter plot of {feature} vs Y house price of unit area',
                width=600,
                height=400
            )
            st.altair_chart(scatter_plot, use_container_width=True)

        # Interactive Histogram
        st.subheader("Distribution of Target Variable")
        st.write("""
        The histogram below shows the distribution of the target variable, 'Y house price of unit area'. This analysis helps us understand the range and skewness of house prices in the dataset.
        """)
        hist = alt.Chart(train_data).mark_bar().encode(
            alt.X('Y house price of unit area:Q', bin=True),
            y='count()',
            tooltip=['count()']
        ).properties(
            title='Distribution of Y house price of unit area',
            width=600,
            height=400
        ).interactive()
        st.altair_chart(hist, use_container_width=True)

        # Interactive Box Plots
        st.subheader("Box Plots of Numeric Features")
        st.write("""
        The box plots below help in identifying the spread and outliers in the data. Outliers can sometimes distort model predictions and might need to be treated separately.
        """)
        for feature in numeric_columns:
            box_plot = alt.Chart(train_data).mark_boxplot().encode(
                x=alt.X('Y house price of unit area:Q'),
                y=alt.Y(feature + ':Q'),
                tooltip=[feature, 'Y house price of unit area']
            ).properties(
                title=f'Box plot of Y house price of unit area by {feature}',
                width=600,
                height=400
            )
            st.altair_chart(box_plot, use_container_width=True)

        # Model Building
        st.header("4. Model Building")
        st.write("""
        In this section, we build and evaluate different linear models: standard Linear Regression, Lasso Regression, and Ridge Regression. These models are chosen to explore how regularization techniques (Lasso and Ridge) affect the model's performance.
        """)

        X_train = train_data.drop(columns=['Y house price of unit area'])
        y_train = train_data['Y house price of unit area']

        X_test = test_data.drop(columns=['Y house price of unit area'])
        y_test = test_data['Y house price of unit area']

        X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # Create Polynomial Features
        poly = PolynomialFeatures(degree=2, include_bias=False)

        # Models to Evaluate
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso()
        }

        # Evaluate each model
        results = {}
        for name, model in models.items():
            pipeline = Pipeline([
                ('poly_features', poly),
                ('regression', model)
            ])

            pipeline.fit(X_train_split, y_train_split)
            y_val_pred = pipeline.predict(X_val)
            mse = mean_squared_error(y_val, y_val_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_val, y_val_pred)

            results[name] = {
                "RMSE": rmse,
                "R-squared": r2
            }

        # Display the results
        st.subheader("Model Evaluation Results")
        for model_name, metrics in results.items():
            st.write(f"**{model_name}:**")
            st.write(f"- Validation RMSE: {metrics['RMSE']:.4f}")
            st.write(f"- Validation R-squared: {metrics['R-squared']:.4f}")

        # Choose the best model based on RMSE
        best_model_name = min(results, key=lambda k: results[k]["RMSE"])
        best_model = models[best_model_name]
        pipeline = Pipeline([
            ('poly_features', poly),
            ('regression', best_model)
        ])
        pipeline.fit(X_train, y_train)

        # Save the best model
        joblib.dump(pipeline, 'trained_linear_model.pkl')

        # Prediction on the test set
        y_test_pred = pipeline.predict(X_test)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(y_test, y_test_pred)

        st.write(f"**Test Results for {best_model_name}:**")
        st.write(f"- Test RMSE: {test_rmse:.4f}")
        st.write(f"- Test R-squared: {test_r2:.4f}")

        # Actual vs Predicted
        st.header("5. Actual vs Predicted")
        st.write("""
        The following chart compares the actual house prices to the predicted prices on the test set. This comparison allows us to visually assess the accuracy of our model. A perfect model would have all points lying on the 45-degree line, indicating that the predicted values match the actual values exactly.
        """)

        actual_vs_predicted = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_test_pred
        })

        scatter_actual_vs_predicted = alt.Chart(actual_vs_predicted).mark_circle(size=60).encode(
            x=alt.X('Actual', scale=alt.Scale(zero=False)),
            y=alt.Y('Predicted', scale=alt.Scale(zero=False)),
            tooltip=['Actual', 'Predicted']
        ).interactive().properties(
            title='Actual vs Predicted House Prices',
            width=600,
            height=400
        )

        st.altair_chart(scatter_actual_vs_predicted, use_container_width=True)

        st.write("""
        The scatter plot above shows the relationship between actual and predicted house prices. The closer the points are to the diagonal line, the better the model's predictions. Deviations from this line indicate discrepancies between the actual and predicted values, which can be further analyzed to improve model performance.
        """)

        # Prediction Section
        st.header("6. Predict House Price")
        st.write("""
        Use the inputs below to predict the house price per unit area based on the trained model. This feature allows you to experiment with different inputs and see how the model responds.
        """)

        house_age = st.number_input("House Age", min_value=0, max_value=100)
        distance_to_mrt = st.number_input("Distance to MRT Station", min_value=0)
        convenience_stores = st.number_input("Number of Convenience Stores", min_value=0)
        latitude = st.number_input("Latitude")
        longitude = st.number_input("Longitude")

        if st.button("Predict House Price"):
            features = poly.transform(scaler.transform([[house_age, distance_to_mrt, convenience_stores, latitude, longitude]]))
            prediction = pipeline.predict(features)
            st.write(f"**Predicted House Price per Unit Area:** {prediction[0]:.2f}")

        st.write("""
        This section allows you to predict house prices using the model trained earlier. By inputting the relevant features (house age, distance to the nearest MRT station, number of convenience stores nearby, latitude, and longitude), the model will estimate the price per unit area of the house.
        """)

if __name__ == "__main__":
    main()
