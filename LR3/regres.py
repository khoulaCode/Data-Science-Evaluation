import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

st.title("Real Estate Price Prediction")

# Step 1: Data Loading
train_file = st.file_uploader("Upload the Training Dataset", type="csv")
test_file = st.file_uploader("Upload the Testing Dataset", type="csv")

if train_file is not None and test_file is not None:
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
    st.write("Training Data Preview:")
    st.write(train_data.head())
    
    st.write("Testing Data Preview:")
    st.write(test_data.head())

    # Step 2: Data Cleaning and Structuring
    st.subheader("Data Cleaning and Structuring")
    
    # Assuming there are no missing values, otherwise handle them
    train_data.dropna(inplace=True)
    test_data.dropna(inplace=True)
    
    # Step 3: Exploratory Data Analysis (EDA)
    st.subheader("Exploratory Data Analysis")
    
    # Basic Information
    st.write("Dataset Information:")
    st.write(train_data.info())
    
    st.write("Descriptive Statistics:")
    st.write(train_data.describe())
    
    # Distribution of the Target Variable
    st.write("Distribution of House Prices per Unit Area:")
    plt.figure(figsize=(10, 6))
    sns.histplot(train_data['Y house price of unit area'], kde=True)
    plt.title('Distribution of House Prices per Unit Area')
    st.pyplot(plt)
    
    # Boxplot of the Target Variable
    st.write("Boxplot of House Prices per Unit Area:")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=train_data['Y house price of unit area'])
    plt.title('Boxplot of House Prices per Unit Area')
    st.pyplot(plt)
    
    # Correlation Matrix
    st.write("Correlation Matrix:")
    corr = train_data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    st.pyplot(plt)
    
    # Pairplot to explore pairwise relationships
    st.write("Pairplot of Features:")
    sns.pairplot(train_data)
    st.pyplot(plt)
    
    # Distribution plots for each feature
    st.write("Distribution of Features:")
    features = ['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station', 
                'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']
    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.histplot(train_data[feature], kde=True)
        plt.title(f'Distribution of {feature}')
        st.pyplot(plt)
    
    # Scatter plots for feature relationships with the target variable
    st.write("Scatter Plots of Features vs. Target Variable:")
    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=train_data[feature], y=train_data['Y house price of unit area'])
        plt.title(f'{feature} vs. House Price per Unit Area')
        st.pyplot(plt)
    
    # Step 4: Feature Engineering
    st.subheader("Feature Engineering")
    
    X_train = train_data[features]
    y_train = train_data['Y house price of unit area']
    
    X_test = test_data[features]
    y_test = test_data['Y house price of unit area']
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Step 5: Model Building
    st.subheader("Model Building")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Step 6: Model Evaluation
    st.subheader("Model Evaluation")
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"R-squared: {r2}")
    
    # Display a comparison of the first few predictions vs. actual prices
    comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    st.write("Comparison of Actual and Predicted Prices:")
    st.write(comparison.head())
    
    # Step 7: Deployment - Input data and predict
    st.subheader("Predict Property Price")
    
    transaction_date = st.number_input("Transaction Date", min_value=0)
    house_age = st.number_input("House Age", min_value=0)
    distance_to_mrt = st.number_input("Distance to MRT Station", min_value=0)
    num_convenience_stores = st.number_input("Number of Convenience Stores", min_value=0)
    latitude = st.number_input("Latitude")
    longitude = st.number_input("Longitude")
    
    input_data = pd.DataFrame({
        'X1 transaction date': [transaction_date],
        'X2 house age': [house_age],
        'X3 distance to the nearest MRT station': [distance_to_mrt],
        'X4 number of convenience stores': [num_convenience_stores],
        'X5 latitude': [latitude],
        'X6 longitude': [longitude]
    })
    
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    
    st.write(f"Predicted House Price per Unit Area: {prediction[0]}")
